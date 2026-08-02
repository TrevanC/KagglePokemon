"""Self-contained CPU inference agent for the Kaggle submission bundle.

Copied verbatim into submission/build/standalone_agent.py by
scripts/package_submission.py, where submission/build/main.py imports it:

    from standalone_agent import StandalonePTCGAgent

Design rules this file must satisfy:
  - Rule 4: fully self-contained. No imports from src/ or anywhere outside the
    deployment folder — inline whatever encoder/extractor code it needs. This
    is the one place duplication is correct.
  - Rule 1: < 10 ms per turn on CPU. Load the model and static table once at
    construction; no per-turn allocations of any size, no torch.load, no CSV.
  - Rule 2: read static_table.pt, never EN_Card_Data.csv.

Also mind: torch.no_grad() (or inference_mode) around every forward pass,
model.eval(), torch.set_num_threads(1) — the Kaggle box has 2 vCPUs and thread
contention costs more than it buys at this model size.

The returned list must have length in [minCount, maxCount] with no duplicate
indices, and must be valid even when obs["select"] is None (the initial
selection, where the agent returns its 60-card deck instead).

TODO: implement
    class StandalonePTCGAgent:
        __init__(model_path, static_table_path)
        act(obs_dict) -> list[int]
"""
