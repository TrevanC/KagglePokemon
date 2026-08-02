"""Step 5: evaluate a candidate model against baselines in the local arena.

    python scripts/run_arena.py --candidate models/checkpoints/bc_epoch10.pt \\
                               --baseline heuristic

Input:   a checkpoint from models/checkpoints/, baselines from
         src/evaluation/heuristics.py
Output:  win rates with 95% Wilson confidence intervals; on a clear win,
         promotion of the checkpoint to models/registry/best_policy.pt
Config:  configs/env_config.yaml

Thin CLI over src/evaluation/arena.py, which builds on the vendored
tournament runner. Always swap sides (--swap) to cancel position bias, and run
enough games that the confidence interval is narrower than the margin being
claimed — the engine's RNG is not seedable and single games are noisy.

This, not validation loss, is the promotion gate for the model registry.
"""

if __name__ == "__main__":
    raise NotImplementedError("scripts/run_arena.py is a stub")
