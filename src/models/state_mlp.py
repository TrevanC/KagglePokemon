"""Baseline MLP board state encoder.

Fast, deterministic dense projection (technical specification §3.1):

    raw state (1, 912) -> Linear(912, 512) -> ReLU -> Linear(512, 256) -> s (1, 256)

Implements the encoder interface shared with state_transformer.py:
forward(obs_dict) -> Tensor[1, 256]. Design rule 3 — the two must stay
interchangeable so swapping encoders never touches the scorer or the
submission code.

TODO: implement
    class MLPBoardStateEncoder(nn.Module):
        __init__(raw_dim=912, hidden_dim=512, state_embed_dim=256)
        forward(raw_state: Tensor (B, 912)) -> Tensor (B, 256)
"""
