"""Self-play / heuristic experience collector.

Wraps a match loop, records every decision point, and writes trajectories to
data/replays/<run_name>/ (e.g. random_vs_heuristic/, self_play_v1/).

One record per decision the acting player made. The record format and its
gzipped-JSONL reader/writer already exist in src/dataset/records.py:

    {"obs": <raw observation dict>,
     "action": <list[int] option indices returned>,
     "player": 0 | 1,
     "reward": +1 won / -1 lost / 0 draw,
     "source": "selfplay:lucario" | "kaggle:89083821" | ...}

Write through records.write_records() rather than reimplementing it, so
locally generated self-play data and downloaded Kaggle episode replays stay
interchangeable. The featurizer and dataset then never care which source a
record came from.

Record obs *before* the action, and backfill `reward` from the final match
result once the game ends.

TODO: implement
    class TrajectoryBuffer:
        add(obs, action, player)
        finalize(winner)         # backfill rewards for both players
        flush(path)              # write .jsonl.gz

Consumer: scripts/generate_replays.py, then scripts/process_dataset.py.
"""
