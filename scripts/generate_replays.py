"""Step 2: run bots against each other and log their decisions.

    python scripts/generate_replays.py --num_games 1000 --bot1 greedy --bot2 random

Input:   the cabt engine via src/env/cabt_wrapper.py, agents from
         src/evaluation/heuristics.py (or a submission folder / checkpoint)
Output:  gzipped JSONL trajectories under data/replays/<run_name>/
Config:  configs/env_config.yaml

The strong sample agents (lucario, iono) are the useful expert-data sources for
behavioral cloning; random/greedy are mainly for pipeline sanity checks.

Uses src/dataset/trajectory_buffer.py to collect and write records.

The engine's RNG is not seedable, so games vary — generate a lot of them.
Local self-play is effectively free: measured 0.04 s/game, ~94 MAIN decisions
per game, ~2,200 MAIN decisions/sec single-process. 100k decisions takes under
a minute. Data *volume* is never the constraint; decision *quality* is.

KAGGLE REPLAYS (the reason to bother)
-------------------------------------
Self-play against lucario can only ever produce lucario-quality decisions, and
cloning the agent you already delegate to is a null operation (spec §1). Real
Kaggle episodes are the one source of MAIN decisions better than the fallback.

    kaggle competitions episodes <submission_id>       # your own submissions only
    kaggle competitions replay <episode_id> -p ./dir   # episode-<id>-replay.json

Each replay carries BOTH players' observations and actions, and
replay["rewards"] says who won — so filter to the winner's side. Measured on
episode 89083821: 120 decisions, ~60 MAIN, split roughly evenly per player,
3.0 MB per replay. Reaching 100k MAIN decisions from winners alone needs
~3,300 episodes (~10 GB), versus under a minute of self-play — the entire
value is that the decisions are stronger.

    >>> Actions are stored one step LATE: the action responding to
    >>> steps[i][p]["observation"] lives at steps[i+1][p]["action"].
    >>> Verified 120/120 legal at +1 vs 79/120 at +0 — the wrong alignment
    >>> looks mostly valid and silently corrupts the labels. Assert legality
    >>> per row: minCount <= len(action) <= maxCount, no duplicates, indices
    >>> in range.

Both sources normalize to the same record schema (src/dataset/records.py), so
they mix freely downstream.
"""

if __name__ == "__main__":
    raise NotImplementedError("scripts/generate_replays.py is a stub")
