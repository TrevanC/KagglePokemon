"""Action encoder: projects assembled candidate vectors to action embeddings.

Spec §5:

    A_raw (B, k, 192) -> Linear(192, 64) -> ReLU -> Linear(64, 64) -> (B, k, 64)

The 192-dim input is three concatenated 64-dim blocks built by
src/env/state_extractor.py::assemble_action_matrix:

    structural (64)   option fields the engine actually populates -- type,
                      area/index, inPlayArea/inPlayIndex, attack stats,
                      energyIndex, toolIndex, number, targets_self
    source card (64)  the card the option acts WITH, resolved from
                      (area, index) and embedded via HybridCardEncoder
    target slot (64)  the field-slot vector for (inPlayArea, inPlayIndex),
                      gathered from the 12 slots already computed for the board

The last two blocks are what make the encoding injective. The v1 design
encoded only (type one-hot, cardId/1267, slot/12) into 16 dims; since the
engine never populates cardId on MAIN options, 50.2% of options collided with
a sibling. See spec §0 and the §5.4 invariant.

Applied per-candidate, so it batches naturally over (B, k, 192).
"""

from __future__ import annotations

import torch
from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, action_in_dim: int = 192, action_embed_dim: int = 64):
        super().__init__()
        self.action_in_dim = action_in_dim
        self.action_embed_dim = action_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(action_in_dim, action_embed_dim),
            nn.ReLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, k, 192) -> (B, k, 64). Padding rows pass through and are masked
        downstream by the scorer, never here."""
        return self.mlp(actions)
