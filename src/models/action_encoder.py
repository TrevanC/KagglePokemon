"""Action encoder: projects assembled candidate vectors to action embeddings.

Spec §5:

    A_raw (B, k, 192) -> Linear(192, 64) -> ReLU -> Linear(64, 64) -> (B, k, 64)

The 192-dim input is three concatenated 64-dim blocks built by
src/env/state_extractor.py::extract_action_matrix:

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

TODO: implement
    class ActionEncoder(nn.Module):
        __init__(action_in_dim=192, action_embed_dim=64)
        forward(actions: Tensor (B, k, 192)) -> Tensor (B, k, 64)
"""
