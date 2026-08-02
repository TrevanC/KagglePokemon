"""Hybrid card encoder: static features + learned ID embedding + dynamic state.

Every card instance (on the field or in hand) becomes a 64-dim vector
(technical specification §2.2):

    e_card = [ f_static (32) ; e_id (8) ; f_dynamic (8) ]  in R^48
             --Linear(48 -> 64) + ReLU-->  R^64

    f_static    StaticCardFeatureTable[card_id]  (src/card_db/static_table.py)
    e_id        nn.Embedding(1267, 8)
    f_dynamic   hp ratio, energy ratio, 6 status flags
                (src/env/state_extractor.py::extract_card_dynamics)

The static table is a buffer, not a parameter — it must not receive gradients,
and should be registered so it travels with the checkpoint.

TODO: implement
    class HybridCardEncoder(nn.Module):
        __init__(static_table, num_cards=1267, id_embed_dim=8,
                 static_dim=32, dynamic_dim=8, card_out_dim=64)
        forward(card_ids: Tensor (B, N), dynamic: Tensor (B, N, 8)) -> Tensor (B, N, 64)
"""
