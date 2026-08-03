"""PyTorch Dataset for behavioral cloning / imitation learning.

Loads the tensor files that scripts/process_dataset.py produces from raw
replays — data/datasets/train_bc_v1.pt, val_bc_v1.pt — each holding:

    states    (N, 984)      board vectors (spec §4)
    actions   (N, k, 192)   candidate action matrices, padded to K_MAX = 80
    targets   (N,)          index in [0, k-1] of the option the expert chose
    masks     (N, k)        bool; False marks padding, consumed by the scorer's
                            masked_fill before the softmax

Padding to a fixed k is what makes batching possible — k varies per decision
(measured 1..67 for MAIN, mean 11.0, p99 34). Pad to K_MAX = 80 and DROP the
rare k > K_MAX decisions; never truncate the candidate list, since that can
remove the expert's chosen move and leave the target index undefined.

Only SelectContext.MAIN decisions belong in this dataset (spec §1) — the other
48 contexts are served by the heuristic fallback and are not learned.

Reconstruction path (technical specification §2.5): a raw record's `obs` goes
through src/env/state_extractor.py to produce states/actions, and `action` maps
to the target index. Use the same extractor as live inference — divergence
between the two is silent and expensive.

TODO: implement
    class PTCGReplayDataset(torch.utils.data.Dataset):
        __init__(data_path)
        __len__()
        __getitem__(idx) -> {"state", "actions", "target", "mask"}
"""
