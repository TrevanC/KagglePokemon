"""Standardized wrapper around the Kaggle cabt engine.

Does NOT reimplement the engine. The vendored harness at
pokemon-tcg-ai-battle/arena already provides a working local match loop with
the native `cg` bindings; this module puts that directory on sys.path and
re-exports it behind a stable interface so the rest of src/ never depends on
the vendor directory's layout.

What to reuse (see pokemon-tcg-ai-battle/arena/README.md):

    arena/runner.py     run_match(agent0, agent1, deck0, deck1, verbose=False) -> MatchResult
                        read_deck(path) -> list[int]
                        load_submission(dir) -> (agent, deck)   # Kaggle main.py + deck.csv
    arena/tournament.py many-game driver reporting win rates with Wilson CIs
    arena/cards.py      CARDS / ATTACKS lookups, name(), describe_deck()
    arena/cg/api.py     observation & option schema, to_observation_class(obs_dict)

Agent contract, identical to Kaggle:

    def agent(obs_dict: dict) -> list[int]

returning option indices into obs["select"]["option"], of length between
minCount and maxCount, no duplicates. Illegal moves forfeit the match.

Caveats inherited from the harness: the engine's RNG is not seedable (run many
games for stable numbers), and both agents share one process -- avoid pitting
two agents that use the `search_*` planning API against each other locally.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

Agent = Callable[[dict], list]

_ARENA = (Path(__file__).resolve().parent.parent.parent
          / "pokemon-tcg-ai-battle" / "arena")


def arena_path() -> Path:
    """Resolve the vendored arena and make it importable. Idempotent."""
    if not _ARENA.is_dir():
        raise FileNotFoundError(f"vendored arena not found at {_ARENA}")
    p = str(_ARENA)
    if p not in sys.path:
        sys.path.insert(0, p)
    return _ARENA


def main_context() -> int:
    """int value of SelectContext.MAIN, the one context the policy scores."""
    arena_path()
    from cg.api import SelectContext  # noqa: PLC0415 - needs arena on sys.path
    return int(SelectContext.MAIN)


def read_deck(path: str | Path) -> list[int]:
    arena_path()
    from runner import read_deck as _read_deck  # noqa: PLC0415
    return _read_deck(str(path))


def play_match(agent0: Agent, agent1: Agent, deck0: list[int],
               deck1: list[int] | None = None, **kwargs):
    """Thin pass-through to runner.run_match. deck1 defaults to deck0 (mirror)."""
    arena_path()
    from runner import run_match  # noqa: PLC0415
    return run_match(agent0, agent1, deck0,
                     deck0 if deck1 is None else deck1, **kwargs)


def load_agent(spec) -> tuple[Agent, list[int] | None]:
    """Accept a submission dir, a baseline name, or a callable -> (agent, deck).

    `deck` is None for baselines, which are deck-agnostic; a submission folder
    supplies its own via the initial None-select call, exactly as Kaggle does.
    """
    if callable(spec):
        return spec, None

    arena_path()
    name = str(spec)

    from agents import AGENTS  # noqa: PLC0415
    if name in AGENTS:
        return AGENTS[name], None

    if Path(name).is_dir():
        from runner import load_submission  # noqa: PLC0415
        return load_submission(name)

    raise ValueError(f"cannot resolve agent {name!r}: not a baseline name, "
                     "a submission directory, or a callable")
