"""Step 4: train the policy network by behavioral cloning.

    python scripts/train_bc.py --config configs/train/bc_trainer.yaml
    python scripts/train_bc.py --epochs 2 --limit 2000        # smoke test

Input:   data/datasets/*.jsonl.gz records, featurized on the fly (see
         src/dataset/dataset.py for why there is no precomputed .pt stage),
         plus the tables from scripts/build_static_features.py
Output:  models/checkpoints/<experiment_name>.pt
         (promoted to models/registry/best_policy.pt only after run_arena.py)
Config:  configs/train/bc_trainer.yaml

Thin CLI over src/training/trainer_bc.py: parse args, load the YAML config,
build the encoder (mlp | transformer) and DynamicActionScorerNetwork, then hand
off to the training loop.

Promotion is NOT done here. A checkpoint earns promotion by beating the
benchmark in scripts/run_arena.py; validation loss is not the gate, because a
better imitation of a mediocre expert can still lose more matches.

Training may use a GPU; inference is CPU-only, so keep the model small enough
to meet the < 10 ms per-turn budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                      / "pokemon-tcg-ai-battle" / "arena"))

from src.card_db.static_table import (  # noqa: E402
    DEFAULT_ATTACK_PATH,
    DEFAULT_PATH,
    load_attack_table,
    load_static_table,
)
from src.training.trainer_bc import train_bc  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("configs/train/bc_trainer.yaml"))
    p.add_argument("--static-table", type=Path, default=Path(DEFAULT_PATH))
    p.add_argument("--attack-table", type=Path, default=Path(DEFAULT_ATTACK_PATH))
    p.add_argument("--epochs", type=int, help="override optimizer.epochs")
    p.add_argument("--limit", type=int, help="cap records, for smoke tests")
    p.add_argument("--team", help="override data.team")
    p.add_argument("--device", help="override device (cpu | mps | cuda)")
    p.add_argument("--num-workers", type=int, help="override data.num_workers")
    args = p.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        config["optimizer"]["epochs"] = args.epochs
    if args.limit is not None:
        config["data"]["limit"] = args.limit
    if args.team is not None:
        config["data"]["team"] = args.team
    if args.device is not None:
        config["device"] = args.device
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    for path in (args.static_table, args.attack_table):
        if not path.exists():
            print(f"missing {path} -- run scripts/build_static_features.py first",
                  file=sys.stderr)
            return 1

    train_bc(config,
             load_static_table(args.static_table),
             load_attack_table(args.attack_table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
