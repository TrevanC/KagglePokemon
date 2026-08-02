"""PyTorch neural architectures.

The board state encoders are interchangeable: state_mlp and state_transformer
implement the same interface, forward(obs_dict) -> Tensor[1, 256], so the
downstream action scoring and submission code never changes.
"""
