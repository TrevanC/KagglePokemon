"""Dynamic action-scoring policy network -- the joint scorer.

Ranks the k legal candidates the engine offers for a SelectContext.MAIN
decision (spec §7). The state encoder is injected, so the MLP and transformer
encoders stay interchangeable.

    state_emb  (B, 256) --unsqueeze(1).expand(-1,k,-1)--> (B, k, 256)
    A_emb      (B, k, 64)
        -> torch.cat(dim=-1)      -> joint (B, k, 320)
        -> Linear(320, 128) -> ReLU -> Linear(128, 1)
        -> .squeeze(-1)           -> logits (B, k)
        -> masked_fill(~mask, -1e9)
        -> F.softmax(dim=-1)      -> probabilities (B, k)

Masking is not optional: batching pads the candidate dimension, and an unmasked
softmax leaks probability onto padding and biases every gradient.

OUTPUT HEAD -- single-select (spec §7.2). MAIN is always
minCount == maxCount == 1, so training is plain cross-entropy against the
expert's chosen index and inference returns [argmax]. Contexts needing 0 or 2+
selections never reach this network; they belong to the fallback.

    L_BC = -log P(a_y | s_t)

If scope later expands past MAIN (spec §7.3), reuse this scorer with a
multi-hot target and BCEWithLogitsLoss, and select the top-m where
m = clamp(#(sigmoid > 0.5), minCount, maxCount).

REGARDLESS of what the network emits, the agent must apply a final clamp
before returning: length within [minCount, maxCount], no duplicate indices,
every index < len(option). A violation forfeits the match -- not a bad move,
a total loss.

K_MAX = 80 (observed max 50 in the Kaggle record set). Pad for batching; never
truncate -- dropping candidates can drop the expert's chosen move and leave
the training target undefined. Decisions with k > K_MAX are excluded from
training and handled by the fallback at inference.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.env.state_extractor import (
    ACTION_IN_DIM,
    ActionBatch,
    BoardBatch,
    assemble_action_matrix,
)
from src.models.action_encoder import ActionEncoder

K_MAX = 80

# Large negative rather than -inf: a row that is entirely padding would make
# softmax produce NaN over all -inf, which silently poisons the whole batch.
MASK_FILL = -1e9


class DynamicActionScorerNetwork(nn.Module):
    def __init__(self,
                 state_encoder: nn.Module,
                 action_in_dim: int = ACTION_IN_DIM,
                 action_embed_dim: int = 64,
                 state_embed_dim: int = 256,
                 hidden_dim: int = 128):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_encoder = ActionEncoder(action_in_dim, action_embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(state_embed_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, board: BoardBatch, actions: ActionBatch
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (logits (B, k), probabilities (B, k)), padding masked out."""
        state_emb, slot_vecs = self.state_encoder(board)

        # The action matrix is assembled by the featurizer, not re-derived here:
        # one implementation serves training and live inference (invariant 1).
        action_matrix = assemble_action_matrix(
            actions, slot_vecs, self.state_encoder.card_encoder)
        action_emb = self.action_encoder(action_matrix)          # (B, k, 64)

        k = action_emb.shape[1]
        joint = torch.cat(
            [state_emb.unsqueeze(1).expand(-1, k, -1), action_emb], dim=-1)
        logits = self.scorer(joint).squeeze(-1)                  # (B, k)
        logits = logits.masked_fill(~actions.mask, MASK_FILL)
        return logits, F.softmax(logits, dim=-1)


def build_policy(config: dict, static_table: torch.Tensor
                 ) -> "DynamicActionScorerNetwork":
    """Construct the network from a resolved config.

    Shared by the trainer and by StandalonePTCGAgent so a checkpoint is
    guaranteed to be rebuilt with the architecture it was trained with -- a
    mismatch here loads silently and plays badly.
    """
    from src.models.state_mlp import MLPBoardStateEncoder  # noqa: PLC0415

    mcfg = config["model"]
    if mcfg.get("type", "mlp") != "mlp":
        raise NotImplementedError(
            f"model.type={mcfg['type']!r}: only 'mlp' is implemented; "
            "src/models/state_transformer.py is still a stub")
    encoder = MLPBoardStateEncoder(
        static_table,
        hidden_dim=mcfg.get("hidden_dim", 512),
        state_embed_dim=mcfg.get("state_embed_dim", 256),
    )
    return DynamicActionScorerNetwork(
        encoder,
        action_in_dim=mcfg.get("action_in_dim", ACTION_IN_DIM),
        action_embed_dim=mcfg.get("action_embed_dim", 64),
        state_embed_dim=mcfg.get("state_embed_dim", 256),
        hidden_dim=mcfg.get("scorer_hidden_dim", 128),
    )
