"""Hybrid card encoder: static features + learned ID embedding + dynamic state.

Every card instance becomes a 64-dim vector (spec §3.1):

    e_card = [ f_static (72) ; e_id (16) ; f_dynamic (24) ]  in R^112
             --Linear(112 -> 64) -> ReLU -> Linear(64 -> 64)-->  R^64

    f_static    StaticCardFeatureTable[card_id]  (src/card_db/static_table.py),
                a (1268, 72) table indexed directly by card ID
    e_id        nn.Embedding(1268, 16)
    f_dynamic   24 dims: hp ratio, max hp, energy count + 12-wide energy
                histogram, tools, pre-evolution depth, appearThisTurn,
                is_active, and 5 status flags
                (src/env/state_extractor.py::extract_card_dynamics)

This same encoder is reused for the action encoder's source-card block
(spec §5.2) with dynamic features zeroed for cards not in play -- one set of
card semantics shared by the board and the actions, rather than two.

The static table is a registered BUFFER, not a parameter: it must not receive
gradients, and it must travel with the checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from src.card_db.static_table import FEATURE_DIM, TABLE_ROWS


class HybridCardEncoder(nn.Module):
    """(card_ids, dynamic) -> 64-dim card vectors.

    Card id 0 is "no card": its static row is zeroed by construction, so an
    empty field slot encodes to whatever the MLP maps the zero-ish input to,
    consistently everywhere. Callers never branch on presence.
    """

    def __init__(self,
                 static_table: torch.Tensor,
                 table_rows: int = TABLE_ROWS,
                 id_embed_dim: int = 16,
                 static_dim: int = FEATURE_DIM,
                 dynamic_dim: int = 24,
                 card_out_dim: int = 64):
        super().__init__()
        if static_table.shape != (table_rows, static_dim):
            raise ValueError(
                f"static_table must be {(table_rows, static_dim)}, "
                f"got {tuple(static_table.shape)}")

        # Buffer, not parameter: no gradients, but saved in the state dict so a
        # checkpoint is self-sufficient and the submission needs no rebuild.
        self.register_buffer("static_table", static_table.float(), persistent=True)

        self.id_embed = nn.Embedding(table_rows, id_embed_dim, padding_idx=0)
        self.card_in_dim = static_dim + id_embed_dim + dynamic_dim   # 112
        self.card_out_dim = card_out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.card_in_dim, card_out_dim),
            nn.ReLU(),
            nn.Linear(card_out_dim, card_out_dim),
        )

    def forward(self, card_ids: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        """(B, N) ids + (B, N, 24) dynamics -> (B, N, 64)."""
        card_ids = card_ids.long()
        static = self.static_table[card_ids]                 # (B, N, 72)
        ids = self.id_embed(card_ids)                        # (B, N, 16)
        return self.mlp(torch.cat([static, ids, dynamic], dim=-1))
