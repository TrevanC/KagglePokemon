"""Trajectory storage and data loading for imitation / reinforcement learning.

Modules:
    records            The record format every stage speaks, plus its
                       gzipped-JSONL reader/writer.
    trajectory_buffer  Collects records from live matches into data/replays/.
    dataset            PyTorch Dataset over the tensor files built from them.
"""
