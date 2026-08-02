"""Step 2: run bots against each other and log their decisions.

    python scripts/generate_replays.py --num_games 1000 --bot1 greedy --bot2 random

Input:   the cabt engine via src/env/cabt_wrapper.py, agents from
         src/evaluation/heuristics.py (or a submission folder / checkpoint)
Output:  gzipped JSONL trajectories under data/replays/<run_name>/
Config:  configs/env_config.yaml

The strong sample agents (lucario, iono) are the useful expert-data sources for
behavioral cloning; random/greedy are mainly for pipeline sanity checks.

Uses src/dataset/trajectory_buffer.py to collect and write records. Kaggle
episode replays downloaded as JSON (e.g. 89083821.json) convert into the same
record format, so the two sources mix freely downstream.

The engine's RNG is not seedable, so games vary — generate a lot of them.
"""

if __name__ == "__main__":
    raise NotImplementedError("scripts/generate_replays.py is a stub")
