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

# Fraction of illegal actions tolerated at the correct offset before we treat a
# replay as schema-broken rather than merely containing a forfeit. Measured: a
# clean replay scores 0%, an errored-agent replay ~2% (1 bad action of 56), and
# the WRONG offset scores ~30%+ -- so the threshold sits well clear of both.
MAX_ILLEGAL_RATE = 0.05

# Populated lazily so importing this module never requires the native engine.
_MAIN_CONTEXT: int | None = None


def _main_context() -> int:
    """int value of SelectContext.MAIN, resolved from the engine on first use."""
    global _MAIN_CONTEXT
    if _MAIN_CONTEXT is None:
        from cg.api import SelectContext  # noqa: PLC0415 - optional heavy import
        _MAIN_CONTEXT = int(SelectContext.MAIN)
    return _MAIN_CONTEXT


def winner_of(replay: dict) -> int | None:
    """Index of the winning player, or None if there isn't a clear one.

    `rewards` entries can be None: an agent that errored or timed out gets no
    reward while its opponent still scores, e.g. rewards=[1, None] with
    statuses=['DONE', 'ERROR']. Measured rate: 1 replay in 745. Naively calling
    max() on that list raises TypeError comparing None to int, so every reward
    read must go through here.

    A None reward means that agent forfeited, so its opponent won -- which is
    also a reminder that forfeits are real at the top of the leaderboard.
    """
    rewards = replay.get("rewards") or []
    if not rewards:
        return None
    numeric = [(i, r) for i, r in enumerate(rewards) if r is not None]
    if len(numeric) != len(rewards):          # someone errored out
        return numeric[0][0] if len(numeric) == 1 else None
    if len({r for _, r in numeric}) < 2:      # draw
        return None
    return max(numeric, key=lambda ir: ir[1])[0]


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
    """Yield (player_index, observation, action) for each decision in a replay.

    Yields the WHOLE observation, not just `select`. The board state lives in
    obs["current"], and the 984-dim board vector (spec §4) cannot be built
    without it -- storing only the option list would produce records that look
    fine and are useless for training.

    `offset` is the step lag between an observation and its action; it is 1 for
    Kaggle replays and is exposed only so validate_alignment() can compare.
    """
    steps = replay["steps"]
    for i, step in enumerate(steps):
        for p, slot in enumerate(step):
            if slot.get("status") != "ACTIVE":
                continue
            obs = slot.get("observation") or {}
            if not obs.get("select"):
                continue
            j = i + offset
            if j >= len(steps):
                continue
            yield p, obs, steps[j][p].get("action")


def validate_alignment(replay: dict) -> dict:
    """Re-derive the correct action offset instead of trusting a constant.

    Returns {offset: (legal, illegal)}. A correct offset scores 0 illegal;
    the wrong one still scores mostly-legal, which is exactly why this check
    exists.
    """
    out = {}
    for offset in (0, 1):
        legal = illegal = 0
        for _p, obs, action in iter_decisions(replay, offset):
            if is_legal(action, obs["select"]):
                legal += 1
            else:
                illegal += 1
        out[offset] = (legal, illegal)
    return out


DECK_SIZE = 60


def decks_of(replay: dict) -> dict[int, list[int]]:
    """{player_index: 60-card deck} for a replay, skipping malformed entries.

    On the very first call an agent gets ``select is None`` and must answer with
    its deck instead of option indices, so the deck lands in the action slot of
    the *next* step -- the same +1 lag as every other action (see THE OFF-BY-ONE
    above). That makes every replay a decklist source as well as a policy
    source: the top players' exact 60 cards are recoverable without owning a
    submission.
    """
    steps = replay.get("steps") or []
    if len(steps) < 2:
        return {}
    out: dict[int, list[int]] = {}
    for p, slot in enumerate(steps[1]):
        action = slot.get("action")
        if (isinstance(action, list) and len(action) == DECK_SIZE
                and all(isinstance(c, int) for c in action)):
            out[p] = action
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
        legal1, illegal1 = checks[1]
        total = legal1 + illegal1
        rate = illegal1 / total if total else 1.0
        # A handful of illegal actions is expected and benign: an agent that
        # errors out (statuses=[...,'ERROR'], reward None) leaves a malformed
        # final action. Those rows are skipped individually below. What must
        # fail loudly is SYSTEMATIC misalignment -- a schema change -- which
        # shows up as a high illegal rate, or as +1 being no better than +0.
        if total and (rate > MAX_ILLEGAL_RATE or illegal1 >= checks[0][1]):
            raise ValueError(
                f"{path.name}: action alignment failed at offset +1 "
                f"({illegal1} illegal of {total}, {rate:.0%}); "
                f"offset 0 scored {checks[0]}. Replay schema may have changed."
            )

    rewards = replay.get("rewards") or []
    winner = winner_of(replay)
    episode_id = replay["info"]["EpisodeId"]
    teams = replay["info"].get("TeamNames") or []
    main_ctx = _main_context() if main_only else None

    for player, obs, action in iter_decisions(replay, offset=1):
        select = obs["select"]
        if winners_only and (winner is None or player != winner):
            continue
        if main_only and select["context"] != main_ctx:
            continue
        if not is_legal(action, select):
            continue          # forfeited/malformed row; global check above guards the schema
        reward = rewards[player] if player < len(rewards) else None
        yield {
            # The FULL observation: obs["current"] carries the board state the
            # 984-dim feature vector is built from. Dropping it yields records
            # that look valid and cannot train anything.
            "obs": {"select": select, "current": obs.get("current"),
                    "logs": obs.get("logs")},
            "action": action,
            "player": player,
            "reward": 0 if reward is None else reward,
            "source": f"kaggle:{episode_id}",
            "team": teams[player] if player < len(teams) else None,
        }
