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

TODO: implement
    class HybridCardEncoder(nn.Module):
        __init__(static_table, table_rows=1268, id_embed_dim=16,
                 static_dim=72, dynamic_dim=24, card_out_dim=64)
        forward(card_ids: Tensor (B, N), dynamic: Tensor (B, N, 24)) -> Tensor (B, N, 64)
"""
