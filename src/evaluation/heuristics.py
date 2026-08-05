"""Rule-based baseline agents -- and the fallback that plays half the game.

Baselines already exist in the vendored harness; re-export rather than rewrite,
so there is one implementation to keep correct:

    pokemon-tcg-ai-battle/arena/agents.py
        random_agent    uniformly random legal selection
        first_agent     always take the first option
        greedy_agent    develops the board (evolve -> abilities -> play ->
                        attach energy) and only then attacks

    pokemon-tcg-ai-battle/arena/agents/iono/     Iono's Bellibolt ex deck
    pokemon-tcg-ai-battle/arena/agents/lucario/  Mega Lucario ex deck
        The two official sample agents, as Kaggle submission folders
        (main.py + deck.csv). Load with cabt_wrapper's load_submission.

Rough strength order: lucario > iono > greedy > first > random.

THE NON-MAIN FALLBACK IS DELEGATION, NOT NEW CODE
-------------------------------------------------
The policy network scores SelectContext.MAIN only -- 50.6% of decisions
(spec §1). Everything else routes to an existing complete agent.

Do NOT write a per-context handler. Both baselines are legal *by
construction* for any of the 49 contexts, including the 34 that never appear
in lucario-vs-iono play:

    greedy_agent   terminates in `_pick_n(select, ...)`, which dedupes, caps at
                   maxCount, and pads up to minCount from the remaining legal
                   indices. It cannot fall off the end.
    lucario        scores every option generically and returns
                   `desc_indices[:select.maxCount]`; since maxCount <=
                   len(option), the result is always within [minCount, maxCount].

That makes multi-select correct for free too -- maxCount > 1 in 4.9% of
decisions (TO_BENCH 60% of the time, TO_HAND 26%, DISCARD 16%) and minCount is
0 in 8% -- which is exactly the arithmetic that is easy to get wrong by hand.

WHICH baseline to delegate to is a strength decision, not a legality one:

    lucario   much stronger, but its scoring references specific card IDs and
              its own decklist. Correct choice if the agent plays lucario's
              deck; degrades (though stays legal) on a different deck.
    greedy    deck-agnostic and safe, but crude where it matters most: for
              TO_HAND it just takes the first maxCount options, and "which card
              to fetch from the deck" is often the highest-leverage decision of
              the turn.

>>> Our first BC policy clones Yushin Ito and plays that team's Alakazam deck
>>> (data/processed/deck_yushin_ito.csv), NOT lucario's. lucario's card-ID
>>> specific scoring is therefore mismatched to our board, which is why
>>> DEFAULT_FALLBACK is `greedy`. This is an empirical question, not a
>>> settled one -- run both through scripts/run_arena.py and let win rate
>>> decide.

CONSEQUENCE WORTH INTERNALIZING: with lucario as the fallback and a policy
trained by behavioral cloning ON lucario, the agent is lucario everywhere
except MAIN, where it is an imperfect imitation of lucario. That is a strictly
worse lucario. The ML only earns its place when MAIN is trained to *beat*
lucario -- from stronger data (top Kaggle replays) or from self-play RL.

Worth knowing before writing another heuristic: attacking ends the turn, so an
attack-first agent swings before it is set up and loses to random ~87% of the
time. Develop the board first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.env.cabt_wrapper import arena_path

Agent = Callable[[dict], list]

# Deck-agnostic by default; see the note above.
DEFAULT_FALLBACK = "greedy"

# Submission-folder agents, resolved lazily -- loading one executes its main.py.
_SUBMISSION_AGENTS = {"lucario": "agents/lucario", "iono": "agents/iono"}
_cache: dict[str, Agent] = {}


def _load_baselines() -> dict[str, Agent]:
    arena_path()
    from agents import AGENTS  # noqa: PLC0415 - needs arena on sys.path
    return dict(AGENTS)


def get_baseline(name: str) -> Agent:
    """Resolve "random" | "first" | "greedy" | "lucario" | "iono" to an agent.

    Submission-folder agents are loaded through the harness's own
    load_submission so the Kaggle main.py + deck.csv contract is exercised the
    same way here as on Kaggle.
    """
    if name in _cache:
        return _cache[name]

    builtins = _load_baselines()
    if name in builtins:
        _cache[name] = builtins[name]
        return _cache[name]

    if name in _SUBMISSION_AGENTS:
        from src.env.cabt_wrapper import load_agent  # noqa: PLC0415
        agent, _deck = load_agent(str(arena_path() / _SUBMISSION_AGENTS[name]))
        _cache[name] = agent
        return agent

    raise KeyError(f"unknown baseline {name!r}; "
                   f"expected one of {sorted(builtins) + sorted(_SUBMISSION_AGENTS)}")


BASELINES = tuple(["random", "first", "greedy"] + sorted(_SUBMISSION_AGENTS))


def is_main(select: dict | None, main_context: int) -> bool:
    return select is not None and select.get("context") == main_context


def clamp_selection(indices: list[int], select: dict) -> list[int]:
    """Force any wishlist into a legal selection.

    The engine forfeits the match on an illegal selection -- not a bad move, a
    total loss -- so this runs on EVERY return path, including the policy's.
    Mirrors agents._pick_n: dedupe, cap at maxCount, pad up to minCount from
    whatever indices remain.
    """
    n = len(select["option"])
    lo, hi = select["minCount"], min(select["maxCount"], n)
    out = [i for i in dict.fromkeys(indices) if 0 <= i < n][:hi]
    if len(out) < lo:
        out += [i for i in range(n) if i not in out][: lo - len(out)]
    return out


def route(obs: dict,
          policy: Callable[[dict], list[int]] | None,
          baseline: Agent,
          main_context: int,
          k_max: int = 80) -> list[int]:
    """MAIN -> policy, everything else -> baseline.

    Falls through to the baseline whenever the policy cannot or should not
    answer: a non-MAIN context, more candidates than the network was built for,
    or the policy raising. Falling back is always legal; failing is not.
    """
    select = obs.get("select")
    if select is None:
        # The initial deck request. Callers own the decklist, not this router.
        raise ValueError("route() received the initial deck request "
                         "(select is None); the agent must answer it before "
                         "routing")

    if policy is not None and is_main(select, main_context) \
            and len(select["option"]) <= k_max:
        try:
            return clamp_selection(policy(obs), select)
        except Exception:                      # noqa: BLE001 - never forfeit
            pass
    return clamp_selection(baseline(obs), select)
