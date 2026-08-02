import os
import argparse
import yaml
import time
import math
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb


class StaticCardFeatureTable:
    """Precomputed lookup table for static card features (1267 cards x 32 explicit features)."""
    def __init__(self, table_tensor: torch.Tensor = None, num_cards: int = 1267, explicit_dim: int = 32):
        if table_tensor is not None:
            self.table = table_tensor
        else:
            # Fallback zero initialization if static table file is missing
            self.table = torch.zeros(num_cards, explicit_dim)

    def get_features(self, card_ids: torch.Tensor) -> torch.Tensor:
        return self.table[card_ids]


class HybridCardEncoder(nn.Module):
    """Encodes individual card instances by combining static stats, learned IDs, and dynamic stats."""
    def __init__(self, static_table: StaticCardFeatureTable, num_cards: int = 1267, 
                 id_embed_dim: int = 8, explicit_dim: int = 32, card_out_dim: int = 64):
        super().__init__()
        self.static_table = static_table
        self.id_embedding = nn.Embedding(num_cards, id_embed_dim)
        
        # 32 (explicit) + 8 (learned ID) + 8 (dynamic stats) = 48 -> 64
        total_in = explicit_dim + id_embed_dim + 8
        self.proj = nn.Sequential(
            nn.Linear(total_in, card_out_dim),
            nn.ReLU(),
            nn.Linear(card_out_dim, card_out_dim)
        )

    def forward(self, card_ids: torch.Tensor, dynamic_stats: torch.Tensor) -> torch.Tensor:
        # card_ids: (B, N), dynamic_stats: (B, N, 8)
        f_static = self.static_table.get_features(card_ids).to(dynamic_stats.device) # (B, N, 32)
        f_id = self.id_embedding(card_ids) # (B, N, 8)
        combined = torch.cat([f_static, f_id, dynamic_stats], dim=-1) # (B, N, 48)
        return self.proj(combined) # (B, N, 64)


class MLPBoardStateEncoder(nn.Module):
    """Baseline MLP Board State Encoder (Raw 912-dim -> 512 -> 256)."""
    def __init__(self, raw_dim: int = 912, state_out_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(raw_dim, 512),
            nn.ReLU(),
            nn.Linear(512, state_out_dim)
        )

    def forward(self, raw_state_tensor: torch.Tensor) -> torch.Tensor:
        # raw_state_tensor shape: (B, 912)
        return self.projector(raw_state_tensor) # Shape: (B, 256)


class TransformerBoardStateEncoder(nn.Module):
    """Transformer Board State Encoder modeling dynamic card interactions."""
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2, state_out_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.location_embedding = nn.Embedding(16, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, state_out_dim)
        )

    def forward(self, token_sequence: torch.Tensor) -> torch.Tensor:
        # token_sequence shape: (B, L, 128)
        batch_size = token_sequence.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, 128)
        full_seq = torch.cat([cls_tokens, token_sequence], dim=1) # (B, 1 + L, 128)
        
        transformed = self.transformer(full_seq)
        cls_out = transformed[:, 0, :] # Extract updated [CLS] token (B, 128)
        return self.out_projection(cls_out) # (B, 256)


class DynamicActionScorerNetwork(nn.Module):
    """Joint Action Scoring Policy Network."""
    def __init__(self, state_encoder: nn.Module, action_in_dim: int = 16, 
                 action_embed_dim: int = 64, state_embed_dim: int = 256):
        super().__init__()
        self.state_encoder = state_encoder
        
        # Action Encoder MLP (16 -> 64 -> 64)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_in_dim, action_embed_dim),
            nn.ReLU(),
            nn.Linear(action_embed_dim, action_embed_dim)
        )
        
        # Joint Scorer Head ( (256 + 64) -> 128 -> 1 )
        joint_in_dim = state_embed_dim + action_embed_dim
        self.joint_scorer = nn.Sequential(
            nn.Linear(joint_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state_input: torch.Tensor, valid_actions: torch.Tensor, 
                action_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state_input: Raw board tensor or sequence tensor
        valid_actions: Candidate action matrix (B, k, 16)
        action_mask: Optional boolean mask for padded actions (B, k)
        returns: Logits (B, k) and Probabilities (B, k)
        """
        batch_size, k_actions, _ = valid_actions.shape
        
        # 1. Encode State -> (B, 256)
        state_emb = self.state_encoder(state_input)
        
        # 2. Virtual Expansion to match candidate options k -> (B, k, 256)
        state_expanded = state_emb.unsqueeze(1).expand(-1, k_actions, -1)
        
        # 3. Encode Actions -> (B, k, 64)
        action_embs = self.action_encoder(valid_actions)
        
        # 4. Joint Feature Fusion -> (B, k, 320)
        joint_features = torch.cat([state_expanded, action_embs], dim=-1)
        
        # 5. Score State-Action Pairs -> (B, k)
        logits = self.joint_scorer(joint_features).squeeze(-1)
        
        # Apply mask for padded actions if provided
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
            
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities


class PTCGReplayDataset(Dataset):
    """Dataset for Behavioral Cloning trajectory batches."""
    def __init__(self, data_path: str = None, dummy_samples: int = 1000, 
                 k_actions: int = 10, raw_state_dim: int = 912):
        super().__init__()
        if data_path and os.path.exists(data_path):
            print(f"[Dataset] Loading dataset file: {data_path}")
            data = torch.load(data_path)
            self.states = data["states"]         # (N, 912)
            self.actions = data["actions"]       # (N, k, 16)
            self.targets = data["targets"]       # (N,) integer in [0, k-1]
            self.masks = data.get("masks", None)
        else:
            print(f"[Dataset] File '{data_path}' not found. Generating synthetic dummy dataset ({dummy_samples} samples)...")
            self.states = torch.randn(dummy_samples, raw_state_dim)
            self.actions = torch.randn(dummy_samples, k_actions, 16)
            self.targets = torch.randint(0, k_actions, (dummy_samples,))
            self.masks = torch.ones(dummy_samples, k_actions, dtype=torch.bool)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {
            "state": self.states[idx],
            "actions": self.actions[idx],
            "target": self.targets[idx]
        }
        if self.masks is not None:
            item["mask"] = self.masks[idx]
        return item


def load_default_config() -> Dict[str, Any]:
    """Returns baseline default configuration dictionary."""
    return {
        "experiment_name": "bc_mlp_baseline_v1",
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {
            "train_path": "data/datasets/train_bc_v1.pt",
            "val_path": "data/datasets/val_bc_v1.pt",
            "batch_size": 64,
            "num_workers": 2
        },
        "model": {
            "type": "mlp",  # Options: ["mlp", "transformer"]
            "num_cards": 1267,
            "state_embed_dim": 256,
            "action_in_dim": 16,
            "action_embed_dim": 64,
            "num_transformer_layers": 2,
            "num_heads": 4
        },
        "optimizer": {
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "epochs": 15,
            "grad_clip": 1.0
        },
        "checkpoint": {
            "save_dir": "models/checkpoints/",
            "registry_dir": "models/registry/"
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTCG Behavioral Cloning Trainer with W&B Logging")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML training config file")
    parser.add_argument("--dummy", action="store_true", help="Force synthetic dummy dataset for pipeline testing")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B cloud logging (offline mode)")
    return parser.parse_args()


def evaluate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float, float]:
    """Evaluates validation loss, Top-1 accuracy, and Top-3 accuracy."""
    model.eval()
    total_val_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            states = batch["state"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["target"].to(device)
            masks = batch.get("mask", None)
            if masks is not None:
                masks = masks.to(device)

            logits, probs = model(states, actions, action_mask=masks)
            loss = criterion(logits, targets)

            total_val_loss += loss.item() * targets.size(0)
            
            # Compute Top-1 & Top-3 Accuracies
            _, top_preds = probs.topk(k=min(3, probs.size(1)), dim=-1)
            correct_top1 += (top_preds[:, 0] == targets).sum().item()
            
            # Top-3 matches
            targets_expanded = targets.unsqueeze(1).expand_as(top_preds)
            correct_top3 += (top_preds == targets_expanded).any(dim=1).sum().item()

            total_samples += targets.size(0)

    avg_loss = total_val_loss / max(1, total_samples)
    top1_acc = (correct_top1 / max(1, total_samples)) * 100.0
    top3_acc = (correct_top3 / max(1, total_samples)) * 100.0

    return avg_loss, top1_acc, top3_acc


def train():
    args = parse_args()
    config = load_default_config()

    # Overwrite config from YAML if provided
    if args.config and os.path.exists(args.config):
        print(f"[Config] Loading parameters from YAML: {args.config}")
        with open(args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
            config.update(yaml_cfg)

    # Set Reproducibility Seed
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    print(f"[Device] Operating on hardware accelerator: {device}")

    # Initialize W&B Experiment Tracking
    wandb_mode = "disabled" if args.no_wandb else "online"
    wandb.init(
        project="ptcg-ai-challenge",
        name=config["experiment_name"],
        config=config,
        mode=wandb_mode
    )

    # 1. Instantiate Data Loaders
    train_path = None if args.dummy else config["data"]["train_path"]
    val_path = None if args.dummy else config["data"]["val_path"]

    train_dataset = PTCGReplayDataset(train_path, dummy_samples=1200)
    val_dataset = PTCGReplayDataset(val_path, dummy_samples=300)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=True, 
        num_workers=config["data"].get("num_workers", 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=False, 
        num_workers=config["data"].get("num_workers", 0)
    )

    # 2. Instantiate Model Architecture
    model_type = config["model"]["type"].lower()
    if model_type == "transformer":
        print("[Architecture] Building Transformer Board State Encoder...")
        state_encoder = TransformerBoardStateEncoder(
            d_model=128,
            nhead=config["model"].get("num_heads", 4),
            num_layers=config["model"].get("num_transformer_layers", 2),
            state_out_dim=config["model"]["state_embed_dim"]
        )
    else:
        print("[Architecture] Building Baseline MLP Board State Encoder...")
        state_encoder = MLPBoardStateEncoder(
            raw_dim=912,
            state_out_dim=config["model"]["state_embed_dim"]
        )

    model = DynamicActionScorerNetwork(
        state_encoder=state_encoder,
        action_in_dim=config["model"]["action_in_dim"],
        action_embed_dim=config["model"]["action_embed_dim"],
        state_embed_dim=config["model"]["state_embed_dim"]
    ).to(device)

    # Watch model gradients with W&B
    wandb.watch(model, log="all", log_freq=100)

    # 3. Optimizer & Loss Criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["optimizer"]["lr"], 
        weight_decay=config["optimizer"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["optimizer"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    # Prepare Checkpoint Directories
    save_dir = Path(config["checkpoint"]["save_dir"])
    registry_dir = Path(config["checkpoint"]["registry_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    print(f"\n================ STARTING TRAINING ({config['optimizer']['epochs']} Epochs) ================")

    for epoch in range(1, config["optimizer"]["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            states = batch["state"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["target"].to(device)
            masks = batch.get("mask", None)
            if masks is not None:
                masks = masks.to(device)

            optimizer.zero_grad()
            logits, probs = model(states, actions, action_mask=masks)
            loss = criterion(logits, targets)
            
            loss.backward()
            
            # Gradient Clipping
            if config["optimizer"].get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["grad_clip"])
                
            optimizer.step()

            total_train_loss += loss.item() * targets.size(0)

            # Step-level batch loss logging
            step = (epoch - 1) * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "global_step": step
                })

        avg_train_loss = total_train_loss / len(train_dataset)
        val_loss, val_top1, val_top3 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed_time = time.time() - start_time

        # Epoch-level logging to W&B
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss,
            "val/loss": val_loss,
            "val/accuracy_top1": val_top1,
            "val/accuracy_top3": val_top3,
            "train/epoch_time_sec": elapsed_time
        })

        print(f"Epoch {epoch:02d}/{config['optimizer']['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Top-1 Acc: {val_top1:.2f}% | "
              f"Time: {elapsed_time:.1f}s")

        # Checkpoint Management & W&B Artifact Upload
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_top1

            checkpoint_filename = f"{config['experiment_name']}_best.pt"
            checkpoint_path = save_dir / checkpoint_filename
            registry_path = registry_dir / "best_policy.pt"

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_top1,
                "config": config,
            }

            # Save locally
            torch.save(checkpoint_data, checkpoint_path)
            torch.save(checkpoint_data, registry_path)
            print(f"  --> Saved new best checkpoint: {checkpoint_path} (Val Loss: {val_loss:.4f})")

            # Upload Checkpoint to W&B Artifact Registry
            if not args.no_wandb:
                artifact = wandb.Artifact(
                    name=f"{config['experiment_name']}-checkpoint",
                    type="model",
                    metadata={
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_top1_acc": val_top1,
                        "val_top3_acc": val_top3,
                        "model_type": model_type
                    }
                )
                artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(artifact)
                print("  --> Uploaded checkpoint artifact to W&B Cloud")

    print(f"\n[Finished] Training complete. Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    train()