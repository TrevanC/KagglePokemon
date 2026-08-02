"""Behavioral cloning training loop.

Optimizes cross-entropy over the k candidate options at each decision point
(technical specification §2.5):

    L_BC = -log P(a_y | s_t) = -log ( exp(z_y) / sum_j exp(z_j) )

where y is the index of the option the expert actually chose. Because scoring
is already a softmax over candidates, this is plain nn.CrossEntropyLoss on the
logits — but only after padded candidates are masked to -1e9.

Driven by configs/train/bc_trainer.yaml (see scripts/train_bc.py). Expected
behavior:
  - AdamW + CosineAnnealingLR, gradient clipping per config.
  - Track val loss plus top-1 and top-3 accuracy each epoch.
  - Save checkpoints to models/checkpoints/ containing model_state_dict,
    optimizer_state_dict, the resolved config (feature dims, num_cards,
    embedding sizes), and validation metrics — the metadata is what makes a
    checkpoint loadable months later.
  - Promote to models/registry/best_policy.pt only after the candidate beats
    the current benchmark in scripts/run_arena.py. Val loss alone is not
    promotion criteria; win rate is.
  - Log to W&B (project from config), with an offline/disabled switch.

TODO: implement
    train_bc(config) -> dict   # best metrics
    evaluate(model, loader, criterion, device) -> (loss, top1, top3)

A working reference trainer covering most of this lives in
docs/behavioral_cloning_trainer.py.
"""
