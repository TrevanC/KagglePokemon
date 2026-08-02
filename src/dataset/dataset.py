"""PyTorch Dataset for behavioral cloning / imitation learning.

Loads the tensor files that scripts/process_dataset.py produces from raw
replays — data/datasets/train_bc_v1.pt, val_bc_v1.pt — each holding:

    states    (N, 912)      board vectors
    actions   (N, k, 16)    candidate action matrices, padded to a fixed k
    targets   (N,)          index in [0, k-1] of the option the expert chose
    masks     (N, k)        bool; False marks padding, consumed by the scorer's
                            masked_fill before the softmax

Padding to a fixed k is what makes batching possible — k varies per decision
point (1..50 in practice). Pick k from the dataset's distribution and drop or
truncate the tail rather than padding everything to 50.

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
