"""Baseline rule-based agents for the Pokémon TCG AI Battle competition.

Every agent is a plain callable with the Kaggle contract:

    def agent(obs_dict: dict) -> list[int]

It returns a list of *option indices* into `obs["select"]["option"]`, of
length between `minCount` and `maxCount`, with no duplicates.

These are deliberately simple opponents to benchmark your real agent against.
Import them straight into the runner:

    from agents import random_agent, greedy_agent, first_agent
"""

from __future__ import annotations

import random

from cg.api import OptionType, SelectContext

# Option type ints (from cg/api.py OptionType), for readable priorities.
PLAY = int(OptionType.PLAY)
ATTACH = int(OptionType.ATTACH)
EVOLVE = int(OptionType.EVOLVE)
ABILITY = int(OptionType.ABILITY)
RETREAT = int(OptionType.RETREAT)
ATTACK = int(OptionType.ATTACK)
END = int(OptionType.END)
YES = int(OptionType.YES)
NO = int(OptionType.NO)


def _pick_n(select: dict, indices: list) -> list:
    """Clamp a wishlist of indices to a legal [minCount, maxCount] selection."""
    lo, hi = select["minCount"], select["maxCount"]
    indices = list(dict.fromkeys(indices))[:hi]  # dedupe, cap at maxCount
    if len(indices) < lo:  # pad with any remaining legal indices
        extra = [i for i in range(len(select["option"])) if i not in indices]
        indices += extra[: lo - len(indices)]
    return indices


# --------------------------------------------------------------------------

def random_agent(obs: dict) -> list:
    """Pick a uniformly random legal selection. The dumbest baseline."""
    select = obs["select"]
    if select is None:  # initial deck request — handled by the runner, not here
        return []
    n = len(select["option"])
    lo, hi = select["minCount"], min(select["maxCount"], n)
    k = random.randint(lo, hi)
    return random.sample(range(n), k)


def first_agent(obs: dict) -> list:
    """Always take the first legal option(s). Deterministic, very passive."""
    select = obs["select"]
    if select is None:
        return []
    return _pick_n(select, list(range(select["maxCount"])))


def greedy_agent(obs: dict) -> list:
    """A simple aggressive heuristic: develop the board and attack every turn.

    - On the main turn menu, prefer, in order: attack, evolve, use ability,
      attach energy, play a card, then end the turn.
    - During setup / search / discard style prompts, take a sensible default
      (play a Basic to the Active Spot, fill the bench, keep the opening hand).
    """
    select = obs["select"]
    if select is None:
        return []

    options = select["option"]
    context = select["context"]

    # Main turn menu: choose one action by priority. Crucially, ATTACK comes
    # near the end: attacking ends the turn, so we develop the board first
    # (evolve, fire abilities, play cards, attach energy) and only then swing.
    # An attack-first ordering loses badly (~13% vs random); this wins ~86%.
    if context == int(SelectContext.MAIN):
        priority = [EVOLVE, ABILITY, PLAY, ATTACH, ATTACK, END]
        by_type = {}
        for i, o in enumerate(options):
            by_type.setdefault(o["type"], i)
        for want in priority:
            if want in by_type:
                return [by_type[want]]
        return [0]

    # Yes/No prompts: answer based on what is being asked.
    yes_idx = next((i for i, o in enumerate(options) if o["type"] == YES), None)
    no_idx = next((i for i, o in enumerate(options) if o["type"] == NO), None)
    if yes_idx is not None and no_idx is not None:
        yes_contexts = {
            int(SelectContext.IS_FIRST),   # take the first turn
            int(SelectContext.ACTIVATE),   # fire beneficial effects
            int(SelectContext.FIRST_EFFECT),
        }
        no_contexts = {
            int(SelectContext.MULLIGAN),   # keep the opening hand
        }
        if context in yes_contexts:
            return [yes_idx]
        if context in no_contexts:
            return [no_idx]
        return [yes_idx]  # default: say yes

    # Setup / placement / search: take up to maxCount to develop the board.
    develop_contexts = {
        int(SelectContext.SETUP_ACTIVE_POKEMON),
        int(SelectContext.SETUP_BENCH_POKEMON),
        int(SelectContext.TO_BENCH),
        int(SelectContext.TO_FIELD),
        int(SelectContext.TO_HAND),
    }
    if context in develop_contexts:
        return _pick_n(select, list(range(select["maxCount"])))

    # Fallback: satisfy the minimum with the first options.
    return _pick_n(select, list(range(max(select["minCount"], 1))))


# Registry so tools / CLIs can look agents up by name.
AGENTS = {
    "random": random_agent,
    "first": first_agent,
    "greedy": greedy_agent,
}
