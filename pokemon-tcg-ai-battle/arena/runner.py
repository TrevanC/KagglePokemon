"""Local match runner for the Pokémon TCG AI Battle competition.

The Kaggle competition runs matches server-side; this reproduces that loop
locally so you can test agents against each other before submitting.

An *agent* is the exact same contract Kaggle uses: a callable

    def agent(obs_dict: dict) -> list[int]

that returns a list of option indices for the current selection. See
`cg/api.py` for the observation / option schema.

Typical use:

    from runner import run_match, load_submission
    from agents import random_agent, greedy_agent

    deck = read_deck("../sample_submission/sample_submission/deck.csv")
    result = run_match(greedy_agent, random_agent, deck, deck, verbose=True)
    print(result)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

# Make the bundled engine importable regardless of where this is run from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cg import game  # noqa: E402  (imports the native engine)
from cg.api import LogType, OptionType  # noqa: E402

Agent = Callable[[dict], list]

# Reason codes attached to the RESULT log (see cg/api.py LogType.RESULT).
RESULT_REASON = {
    1: "opponent took all prize cards",
    2: "started turn with an empty deck (deck-out)",
    3: "no Pokémon in the Active Spot",
    4: "a card effect",
}


@dataclass
class MatchResult:
    winner: int  # 0, 1, or 2 for a draw
    reason: int  # RESULT log reason code (0 if unknown)
    reason_text: str
    turns: int  # final turn counter
    steps: int  # number of selections made across both players
    error: Optional[str] = None  # set if an agent produced an illegal move
    error_player: Optional[int] = None

    def __str__(self) -> str:
        if self.error is not None:
            return (f"player {self.error_player} forfeited: {self.error} "
                    f"-> winner={self.winner}")
        who = "draw" if self.winner == 2 else f"player {self.winner} wins"
        return f"{who} ({self.reason_text}) in {self.turns} turns, {self.steps} steps"


def read_deck(path: str) -> list:
    """Read a Kaggle-format deck.csv: 60 integer card IDs, one per line."""
    with open(path) as f:
        deck = [int(x) for x in f.read().split() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(deck)}")
    return deck


def _validate(select: dict, choice) -> Optional[str]:
    """Return an error string if `choice` is not a legal selection, else None."""
    if not isinstance(choice, list) or not all(isinstance(i, int) for i in choice):
        return f"selection must be list[int], got {choice!r}"
    n = len(select["option"])
    lo, hi = select["minCount"], select["maxCount"]
    if not (lo <= len(choice) <= hi):
        return f"selected {len(choice)} options, must be in [{lo}, {hi}]"
    if len(set(choice)) != len(choice):
        return f"duplicate indices in {choice}"
    if any(not (0 <= i < n) for i in choice):
        return f"index out of range [0, {n}) in {choice}"
    return None


def _final_result(obs: dict) -> tuple:
    """Extract (winner, reason) from the observation once the battle is over."""
    reason = 0
    for log in obs.get("logs", []):
        if log.get("type") == int(LogType.RESULT):
            reason = log.get("reason", 0) or 0
    return obs["current"]["result"], reason


def run_match(
    agent0: Agent,
    agent1: Agent,
    deck0: list,
    deck1: list,
    *,
    max_steps: int = 20000,
    verbose: bool = False,
    on_illegal: str = "forfeit",  # "forfeit" or "raise"
) -> MatchResult:
    """Play one game between two agents and return the result.

    agent0/agent1 use the Kaggle contract: `agent(obs_dict) -> list[int]`.
    The engine owns all hidden information and RNG (shuffles, coin flips), so
    games are not seedable; run many matches for statistics.
    """
    agents = (agent0, agent1)
    obs, start = game.battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(
            f"battle failed to start (errorPlayer={start.errorPlayer}, "
            f"errorType={start.errorType}) — usually an illegal deck.")
    try:
        steps = 0
        while True:
            state = obs["current"]
            if state["result"] != -1:
                winner, reason = _final_result(obs)
                return MatchResult(winner, reason, RESULT_REASON.get(reason, "unknown"),
                                   state["turn"], steps)

            select = obs["select"]
            player = state["yourIndex"]
            choice = agents[player](obs)

            err = _validate(select, choice)
            if err is not None:
                msg = f"agent {player}: {err}"
                if on_illegal == "raise":
                    raise ValueError(msg)
                # Illegal move = forfeit; the other player wins.
                return MatchResult(1 - player, 4, "illegal move (forfeit)",
                                   state["turn"], steps, error=err, error_player=player)

            if verbose:
                ctx = select["context"]
                print(f"[step {steps:4d}] turn {state['turn']:3d}  "
                      f"P{player} ctx={ctx} picks {choice} "
                      f"of {len(select['option'])} options")

            obs = game.battle_select(choice)
            steps += 1
            if steps > max_steps:
                raise RuntimeError(f"match exceeded {max_steps} steps (stuck agent?)")
    finally:
        game.battle_finish()


# --------------------------------------------------------------------------
# Loading a real Kaggle submission folder (main.py + deck.csv)
# --------------------------------------------------------------------------

def load_submission(path: str) -> tuple:
    """Load a Kaggle submission folder as (agent_callable, deck).

    `path` is a directory containing `main.py` (defining `agent`) and
    `deck.csv`. The deck is obtained by calling the agent with the initial
    None-select observation, exactly as Kaggle does; falls back to deck.csv.
    """
    path = os.path.abspath(path)
    main_py = os.path.join(path, "main.py")
    spec = importlib.util.spec_from_file_location(f"submission_{abs(hash(path))}", main_py)
    module = importlib.util.module_from_spec(spec)

    cwd = os.getcwd()
    os.chdir(path)  # so the agent's relative deck.csv / file reads resolve
    try:
        spec.loader.exec_module(module)
        agent_fn = module.agent
        # Kaggle asks for the deck first via a None-select observation.
        deck = agent_fn({"select": None, "logs": [], "current": None,
                         "search_begin_input": None})
    finally:
        os.chdir(cwd)

    if not (isinstance(deck, list) and len(deck) == 60):
        deck = read_deck(os.path.join(path, "deck.csv"))

    # Wrap so the agent always runs with its own directory as cwd.
    def wrapped(obs_dict, _fn=agent_fn, _dir=path):
        here = os.getcwd()
        os.chdir(_dir)
        try:
            return _fn(obs_dict)
        finally:
            os.chdir(here)

    return wrapped, deck


if __name__ == "__main__":
    # Quick self-test: random vs random on the sample deck.
    from agents import random_agent

    sample = os.path.join(_HERE, "..", "sample_submission", "sample_submission", "deck.csv")
    deck = read_deck(sample)
    res = run_match(random_agent, random_agent, deck, deck, verbose=True)
    print(res)
