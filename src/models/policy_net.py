"""Dynamic action-scoring policy network — the joint scorer.

Ranks the k legal candidate options the engine offers this turn
(technical specification §4.2). The state encoder is injected, so the MLP and
transformer encoders are interchangeable.

    state_emb  (1, 256) --.expand(k, -1)--> (k, 256)   # a view: no extra memory
    A_emb      (k, 64)
        -> torch.cat(dim=-1)      -> joint (k, 320)
        -> Linear(320, 128) -> ReLU -> Linear(128, 1)
        -> .squeeze(-1)           -> logits (k,)
        -> F.softmax(dim=-1)      -> probabilities (k,)

Batched training pads the candidate dimension to a fixed k, so forward() takes
an optional boolean action_mask (B, k) and applies logits.masked_fill(~mask,
-1e9) before the softmax — otherwise padded slots leak probability mass.

Training loss (§2.5): cross-entropy over the k candidates against the index y
of the option the expert actually chose.

TODO: implement
    class DynamicActionScorerNetwork(nn.Module):
        __init__(state_encoder, action_in_dim=16, action_embed_dim=64, state_embed_dim=256)
        forward(state_input, valid_actions, action_mask=None) -> (logits, probabilities)

A near-complete reference implementation of this class lives in
docs/Technical Specification.md §6 and docs/behavioral_cloning_trainer.py.
"""
