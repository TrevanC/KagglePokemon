"""Core source module for the Pokémon TCG AI Battle agent.

Subpackages:
    card_db     Static card feature precomputing and CSV parsing.
    env         Interface and wrapper layer over the cabt game engine.
    models      PyTorch neural architectures (encoders, policy network).
    dataset     Trajectory storage and data loading.
    training    Behavioral cloning and reinforcement learning loops.
    evaluation  Local ladder, arena, and baseline heuristic agents.

See docs/ for the architecture blueprint and technical specification.
"""
