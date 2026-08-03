"""Dynamic action-scoring policy network -- the joint scorer.

Ranks the k legal candidates the engine offers for a SelectContext.MAIN
decision (spec §7). The state encoder is injected, so the MLP and transformer
encoders stay interchangeable.

    state_emb  (B, 256) --unsqueeze(1).expand(-1,k,-1)--> (B, k, 256)
    A_emb      (B, k, 64)
        -> torch.cat(dim=-1)      -> joint (B, k, 320)
        -> Linear(320, 128) -> ReLU -> Linear(128, 1)
        -> .squeeze(-1)           -> logits (B, k)
        -> masked_fill(~mask, -1e9)
        -> F.softmax(dim=-1)      -> probabilities (B, k)

Masking is not optional: batching pads the candidate dimension to K_MAX = 80,
and an unmasked softmax leaks probability onto padding and biases every
gradient.

OUTPUT HEAD -- single-select (spec §7.2). MAIN is always
minCount == maxCount == 1 (measured 2268/2268), so training is plain
cross-entropy against the expert's chosen index and inference returns
[argmax]. Contexts needing 0 or 2+ selections never reach this network; they
belong to the fallback.

    L_BC = -log P(a_y | s_t)

If scope later expands past MAIN (spec §7.3), reuse this scorer with a
multi-hot target and BCEWithLogitsLoss, and select the top-m where
m = clamp(#(sigmoid > 0.5), minCount, maxCount).

REGARDLESS of what the network emits, the agent must apply a final clamp
before returning: length within [minCount, maxCount], no duplicate indices,
every index < len(option). A violation forfeits the match -- not a bad move,
a total loss.

K_MAX = 80 (observed max 67, mean 11.0, p99 34). Pad for batching; never
truncate -- dropping candidates can drop the expert's chosen move and leave
the training target undefined. Decisions with k > K_MAX are excluded from
training and handled by the fallback at inference.

TODO: implement
    class DynamicActionScorerNetwork(nn.Module):
        __init__(state_encoder, action_in_dim=192, action_embed_dim=64, state_embed_dim=256)
        forward(state_input, valid_actions, action_mask=None) -> (logits, probabilities)

docs/behavioral_cloning_trainer.py has a working reference for the scorer and
training loop, but note it predates the measured dimensions: its action_in_dim
of 16 and raw_dim of 912 are both wrong (spec §0).
"""
