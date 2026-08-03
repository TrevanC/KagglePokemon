"""Baseline MLP board state encoder.

Fast, deterministic dense projection (spec §6.1):

    raw state (B, 984) -> Linear(984, 512) -> ReLU -> Linear(512, 256) -> s (B, 256)

984 = 12*64 field slots + 64 hand + 64 self discard + 64 opp discard
      + 24 global scalars.  (The v1 figure of 912 predates the measured
      feature widths -- see spec §0.)

Implements the encoder interface shared with state_transformer.py:
forward(obs_dict) -> Tensor[B, 256]. Design rule 3 -- the two must stay
interchangeable so swapping encoders never touches the scorer or the
submission code.

TODO: implement
    class MLPBoardStateEncoder(nn.Module):
        __init__(raw_dim=984, hidden_dim=512, state_embed_dim=256)
        forward(raw_state: Tensor (B, 984)) -> Tensor (B, 256)
"""
