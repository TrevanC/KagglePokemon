"""Action matrix MLP encoder.

Turns raw candidate action vectors into action embeddings
(technical specification §4.1):

    A_raw (k, 16) -> Linear(16, 64) -> ReLU -> Linear(64, 64) -> A_emb (k, 64)

Input rows come from src/env/state_extractor.py::extract_action_matrix.
Applied per-candidate, so it batches naturally over (B, k, 16).

TODO: implement
    class ActionEncoder(nn.Module):
        __init__(action_in_dim=16, action_embed_dim=64)
        forward(actions: Tensor (B, k, 16)) -> Tensor (B, k, 64)
"""
