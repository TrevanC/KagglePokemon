"""Convert Kaggle episode replays into the shared record format.

A replay is a Kaggle environment dump (see scripts/fetch_kaggle_replays.py for
how to obtain them). This module turns one into a stream of records as defined
by src/dataset/records.py, so replay data and self-play data are
interchangeable downstream.

    from src.dataset.kaggle_replays import replay_to_records
    records = list(replay_to_records("data/replays/kaggle/episode-123-replay.json"))

THE OFF-BY-ONE
--------------
The action responding to the observation at ``steps[i][p]`` is stored at
``steps[i+1][p]["action"]`` -- NOT at ``steps[i][p]["action"]``.

Measured on a real replay: 120/120 actions legal under the +1 offset,
**79/120 under +0**. The naive alignment therefore does not crash and passes a
legality check on most rows, while silently mislabelling roughly a third of the
data. `validate_alignment()` below re-derives this on any replay; call it
rather than trusting the constant.

WHOSE DECISIONS TO KEEP
-----------------------
Every replay holds both players' decisions and ``replay["rewards"]`` identifies
the winner (e.g. ``[-1, 1]`` -> player 1 won). Cloning the loser teaches losing
play, so `replay_to_records(winners_only=True)` is the default. Each record
still carries ``reward``, so a later stage can weight rather than filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

# Populated lazily so importing this module never requires the native engine.
_MAIN_CONTEXT: int | None = None


def _main_context() -> int:
    """int value of SelectContext.MAIN, resolved from the engine on first use."""
    global _MAIN_CONTEXT
    if _MAIN_CONTEXT is None:
        from cg.api import SelectContext  # noqa: PLC0415 - optional heavy import
        _MAIN_CONTEXT = int(SelectContext.MAIN)
    return _MAIN_CONTEXT


def is_legal(action, select) -> bool:
    """Does `action` satisfy the engine's contract for `select`?

    Length within [minCount, maxCount], no duplicates, every index in range.
    A violation forfeits the match, so this is also what the agent must
    guarantee at inference time.
    """
    if not isinstance(action, list) or select is None:
        return False
    k = len(select["option"])
    return (
        select["minCount"] <= len(action) <= select["maxCount"]
        and len(set(action)) == len(action)
        and all(isinstance(a, int) and 0 <= a < k for a in action)
    )


def iter_decisions(replay: dict, offset: int = 1) -> Iterator[tuple[int, dict, list]]:
    """Yield (player_index, select_dict, action) for each decision in a replay.

    `offset` is the step lag between an observation and its action; it is 1 for
    Kaggle replays and is exposed only so validate_alignment() can compare.
    """
    steps = replay["steps"]
    for i, step in enumerate(steps):
        for p, slot in enumerate(step):
            if slot.get("status") != "ACTIVE":
                continue
            obs = slot.get("observation") or {}
            select = obs.get("select")
            if not select:
                continue
            j = i + offset
            if j >= len(steps):
                continue
            yield p, select, steps[j][p].get("action")


def validate_alignment(replay: dict) -> dict:
    """Re-derive the correct action offset instead of trusting a constant.

    Returns {offset: (legal, illegal)}. A correct offset scores 0 illegal;
    the wrong one still scores mostly-legal, which is exactly why this check
    exists.
    """
    out = {}
    for offset in (0, 1):
        legal = illegal = 0
        for _p, select, action in iter_decisions(replay, offset):
            if is_legal(action, select):
                legal += 1
            else:
                illegal += 1
        out[offset] = (legal, illegal)
    return out


def replay_to_records(
    path: str | Path,
    *,
    main_only: bool = True,
    winners_only: bool = True,
    strict: bool = True,
) -> Iterator[dict]:
    """Stream records out of one replay file.

    main_only     keep only SelectContext.MAIN decisions (spec §1); the other
                  48 contexts are served by the heuristic fallback, not learned
    winners_only  keep only the winning player's decisions
    strict        raise if the +1 alignment does not validate cleanly, rather
                  than silently emitting mislabelled records
    """
    path = Path(path)
    replay = json.loads(path.read_text())

    if strict:
        checks = validate_alignment(replay)
        if checks[1][1] != 0:
            raise ValueError(
                f"{path.name}: action alignment failed at offset +1 "
                f"({checks[1][1]} illegal of {sum(checks[1])}). "
                f"offset 0 scored {checks[0]}. Replay schema may have changed."
            )

    rewards = replay.get("rewards") or []
    winner = (
        rewards.index(max(rewards))
        if rewards and len(set(rewards)) > 1
        else None
    )
    episode_id = replay["info"]["EpisodeId"]
    teams = replay["info"].get("TeamNames") or []
    main_ctx = _main_context() if main_only else None

    for player, select, action in iter_decisions(replay, offset=1):
        if winners_only and (winner is None or player != winner):
            continue
        if main_only and select["context"] != main_ctx:
            continue
        if not is_legal(action, select):
            if strict:
                raise ValueError(f"{path.name}: illegal action {action} for k={len(select['option'])}")
            continue
        yield {
            "obs": {"select": select},
            "action": action,
            "player": player,
            "reward": rewards[player] if player < len(rewards) else 0,
            "source": f"kaggle:{episode_id}",
            "team": teams[player] if player < len(teams) else None,
        }
