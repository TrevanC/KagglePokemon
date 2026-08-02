"""Transformer board state encoder — drop-in replacement for the MLP.

Models relational interactions between cards via self-attention
(technical specification §3.2).

Sequence assembly, each token tagged with a learned zone embedding:

    token 0        [CLS]                zone 0
    tokens 1..12   12 field slots       zones 1..12  (1=self active, 2..6=self bench,
                                                      7=opp active, 8..12=opp bench)
    tokens 13..    N hand cards         zone 13
    ...            discard summary      zone 14
    last token     game scalars         zone 15

    sequence (1, L, 128)
        -> nn.TransformerEncoder(2 layers, d_model=128, nhead=4,
                                 dim_feedforward=256, activation="relu",
                                 batch_first=True)
        -> extract updated [CLS]  (1, 128)
        -> Linear(128, 256) -> ReLU -> Linear(256, 256)
        -> s (1, 256)

Components: nn.Parameter(torch.randn(1, 1, 128)) for [CLS],
nn.Embedding(16, 128) for zones.

Variable hand size means variable L — batched training needs padding plus a
src_key_padding_mask so padded tokens are excluded from attention.

Same interface as state_mlp.py: forward(obs_dict) -> Tensor[1, 256].

TODO: implement
    class TransformerBoardStateEncoder(nn.Module):
        __init__(d_model=128, nhead=4, num_layers=2, num_zones=16, state_embed_dim=256)
        forward(tokens: Tensor (B, L, 128), zone_ids, padding_mask=None) -> Tensor (B, 256)
"""
