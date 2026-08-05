"""Behavioral cloning training loop.

Optimizes cross-entropy over the k candidate options at each decision point
(technical specification §2.5):

    L_BC = -log P(a_y | s_t) = -log ( exp(z_y) / sum_j exp(z_j) )

where y is the index of the option the expert actually chose. Because scoring
is already a softmax over candidates, this is plain nn.CrossEntropyLoss on the
logits -- but only after padded candidates are masked to -1e9.

Driven by configs/train/bc_trainer.yaml (see scripts/train_bc.py).

  - AdamW + CosineAnnealingLR, gradient clipping per config.
  - Track val loss plus top-1 and top-3 accuracy each epoch.
  - Checkpoints carry model_state_dict, optimizer_state_dict, the resolved
    config and validation metrics -- the metadata is what makes a checkpoint
    loadable months later.
  - Promotion to models/registry/best_policy.pt happens only after the
    candidate beats the benchmark in scripts/run_arena.py. Val loss is NOT
    promotion criteria; arena win rate is. This module therefore writes
    checkpoints and stops.

BASELINES TO BEAT, not just "loss went down": a uniform policy over k
candidates scores 1/k, and always taking the first option is what `first_agent`
does. Both are reported next to val accuracy each epoch, because a model that
merely matches them has learned nothing.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset.dataset import PTCGReplayDataset, bc_collate
from src.models.policy_net import build_policy


def resolve_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def episode_split(dataset: PTCGReplayDataset, val_fraction: float, seed: int
                  ) -> tuple[Subset, Subset]:
    """Hold out whole EPISODES, never individual decisions.

    ~42 MAIN decisions come from a single game and share most of their board
    state. Splitting per decision puts near-duplicates on both sides and
    inflates val accuracy by a wide margin -- the model gets credit for having
    memorized the same game.
    """
    by_episode: dict[str, list[int]] = {}
    for i, rec in enumerate(dataset.records):
        by_episode.setdefault(rec["source"], []).append(i)

    episodes = sorted(by_episode)
    random.Random(seed).shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_fraction))
    val_eps = set(episodes[:n_val])

    train_idx = [i for ep in episodes if ep not in val_eps for i in by_episode[ep]]
    val_idx = [i for ep in val_eps for i in by_episode[ep]]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """-> {loss, top1, top3, first_option_rate, uniform_baseline}."""
    model.eval()
    total_loss = n = top1 = top3 = chose_first = 0
    uniform = 0.0
    for board, actions, targets in loader:
        board, actions = board.to(device), actions.to(device)
        targets = targets.to(device)
        logits, _ = model(board, actions)
        total_loss += criterion(logits, targets).item() * targets.numel()

        k = actions.mask.sum(dim=1)                      # real candidates per row
        ranked = logits.argsort(dim=-1, descending=True)
        top1 += (ranked[:, 0] == targets).sum().item()
        top3 += (ranked[:, :3] == targets.unsqueeze(1)).any(dim=1).sum().item()
        chose_first += (targets == 0).sum().item()
        uniform += (1.0 / k.clamp(min=1).float()).sum().item()
        n += targets.numel()

    return {
        "loss": total_loss / max(n, 1),
        "top1": top1 / max(n, 1),
        "top3": top3 / max(n, 1),
        "first_option_rate": chose_first / max(n, 1),
        "uniform_baseline": uniform / max(n, 1),
        "n": n,
    }


def train_bc(config: dict, static_table: torch.Tensor,
             attack_table: torch.Tensor) -> dict:
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)

    dcfg, ocfg = config["data"], config["optimizer"]
    device = resolve_device(config.get("device", "auto"))

    dataset = PTCGReplayDataset(
        dcfg["records_path"], attack_table,
        team=dcfg.get("team"), k_max=dcfg.get("k_max", 80),
        limit=dcfg.get("limit"))
    if not len(dataset):
        raise ValueError(f"no records matched (skipped: {dataset.skipped})")
    print(f"dataset: {len(dataset)} decisions   skipped {dataset.skipped}")

    train_set, val_set = episode_split(dataset, dcfg.get("val_fraction", 0.1), seed)
    episodes = len({r['source'] for r in dataset.records})
    print(f"split by episode: {len(train_set)} train / {len(val_set)} val "
          f"over {episodes} episodes")

    common = dict(batch_size=dcfg.get("batch_size", 64), collate_fn=bc_collate,
                  num_workers=dcfg.get("num_workers", 0))
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)

    model = build_policy(config, static_table).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {n_params:,} trainable params on {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ocfg.get("lr", 3e-4),
                                  weight_decay=ocfg.get("weight_decay", 1e-4))
    epochs = ocfg.get("epochs", 25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    grad_clip = ocfg.get("grad_clip", 1.0)

    save_dir = Path(config["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    name = config.get("experiment_name", "bc")
    best = {"top1": -1.0}

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running = steps = 0
        for board, actions, targets in train_loader:
            board, actions = board.to(device), actions.to(device)
            targets = targets.to(device)
            logits, _ = model(board, actions)
            loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item()
            steps += 1
        scheduler.step()

        metrics = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch:3d}/{epochs}  train {running/max(steps,1):.4f}  "
              f"val {metrics['loss']:.4f}  top1 {metrics['top1']:.3f}  "
              f"top3 {metrics['top3']:.3f}  "
              f"(first-option {metrics['first_option_rate']:.3f}, "
              f"uniform {metrics['uniform_baseline']:.3f})  "
              f"{time.perf_counter()-t0:.0f}s")

        if metrics["top1"] > best["top1"]:
            best = dict(metrics, epoch=epoch)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "metrics": metrics,
                "epoch": epoch,
            }, save_dir / f"{name}.pt")

    print(f"\nbest: epoch {best['epoch']}  top1 {best['top1']:.3f}  "
          f"top3 {best['top3']:.3f}  val loss {best['loss']:.4f}")
    print(f"checkpoint: {save_dir / f'{name}.pt'}")
    print("NOT promoted -- promotion gate is arena win rate "
          "(scripts/run_arena.py), not val loss.")
    return best
