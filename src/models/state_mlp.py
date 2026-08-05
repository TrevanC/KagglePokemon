"""Baseline MLP board state encoder.

Fast, deterministic dense projection (spec §6.1):

    raw state (B, 984) -> Linear(984, 512) -> ReLU -> Linear(512, 256) -> s (B, 256)

984 = 12*64 field slots + 64 hand + 64 self discard + 64 opp discard
      + 24 global scalars.  (The v1 figure of 912 predates the measured
      feature widths -- see spec §0.)

Implements the encoder interface shared with state_transformer.py:
forward(BoardBatch) -> (state_emb (B, 256), slot_vecs (B, 12, 64)). Design
rule 3 -- the two must stay interchangeable so swapping encoders never touches
the scorer or the submission code.

The encoder owns the learned pieces the 984-vector needs -- the shared
HybridCardEncoder and the two discard projections -- because a transformer
variant will consume the same raw board differently (attending over the 12 slot
vectors instead of flattening them) while presenting the same interface. It
returns slot_vecs alongside the embedding so the action matrix's target block
is a gather rather than a recomputation.
"""

from __future__ import annotations

import torch
from torch import nn

from src.card_db.static_table import TABLE_ROWS
from src.env.state_extractor import (
    BoardBatch,
    CARD_OUT_DIM,
    RAW_STATE_DIM,
    assemble_board_state,
)
from src.models.card_encoder import HybridCardEncoder


class MLPBoardStateEncoder(nn.Module):
    def __init__(self,
                 static_table: torch.Tensor,
                 raw_dim: int = RAW_STATE_DIM,
                 hidden_dim: int = 512,
                 state_embed_dim: int = 256,
                 discard_proj_dim: int = CARD_OUT_DIM):
        super().__init__()
        self.card_encoder = HybridCardEncoder(static_table)
        # Separate parameters per side: what my discard pile reveals about my
        # deck is not what the opponent's reveals about theirs.
        self.self_discard_proj = nn.Sequential(
            nn.Linear(TABLE_ROWS, discard_proj_dim), nn.ReLU())
        self.opp_discard_proj = nn.Sequential(
            nn.Linear(TABLE_ROWS, discard_proj_dim), nn.ReLU())
        self.state_embed_dim = state_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_embed_dim),
        )

    def forward(self, board: BoardBatch) -> tuple[torch.Tensor, torch.Tensor]:
        raw_state, slot_vecs = assemble_board_state(
            board, self.card_encoder,
            self.self_discard_proj, self.opp_discard_proj)
        return self.mlp(raw_state), slot_vecs
