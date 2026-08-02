"""Step 4: train the policy network by behavioral cloning.

    python scripts/train_bc.py --config configs/train/bc_trainer.yaml

Input:   data/datasets/train_bc_v1.pt, val_bc_v1.pt
Output:  models/checkpoints/<experiment_name>_best.pt
         (promoted to models/registry/best_policy.pt only after run_arena.py)
Config:  configs/train/bc_trainer.yaml, plus configs/model/*.yaml for the encoder

Thin CLI over src/training/trainer_bc.py: parse args, load and merge the YAML
config, build the encoder (mlp | transformer) and DynamicActionScorerNetwork,
then hand off to the training loop.

Training runs on GPU; inference is CPU-only, so keep the model small enough to
meet the < 10 ms per-turn budget.
"""

if __name__ == "__main__":
    raise NotImplementedError("scripts/train_bc.py is a stub")
