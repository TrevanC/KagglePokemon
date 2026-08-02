"""Step 3: turn raw replay logs into PyTorch tensor datasets.

    python scripts/process_dataset.py

Input:   data/replays/**/*.jsonl.gz  plus
         data/processed/static_feature_table.pt
Output:  data/datasets/train_bc_v1.pt, data/datasets/val_bc_v1.pt
Config:  configs/feature_config.yaml

Each output holds states (N, 912), actions (N, k, 16), targets (N,), and
masks (N, k) — see src/dataset/dataset.py.

Featurization goes through src/env/state_extractor.py, the *same* code live
inference uses (technical specification §2.5). Reimplementing it here would
silently desync training from deployment.

Split train/val by game, not by decision point: consecutive decisions within
one match are highly correlated, and a random per-row split leaks state across
the boundary and inflates validation accuracy.
"""

if __name__ == "__main__":
    raise NotImplementedError("scripts/process_dataset.py is a stub")
