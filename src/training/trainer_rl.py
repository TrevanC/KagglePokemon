"""PPO / self-play reinforcement learning loop.

Second training stage, warm-started from a behavioral cloning checkpoint (the
BC policy provides a sane action distribution; PPO from scratch on a 1..50-way
dynamic action space is slow to move).

Shape of the loop, driven by configs/train/rl_trainer.yaml:
  1. Roll out matches against an opponent pool (baseline heuristics + frozen
     past checkpoints) via src/env/cabt_wrapper.py, collecting decisions into
     src/dataset/trajectory_buffer.py.
  2. Compute GAE returns and advantages. Rewards are terminal (+1 win / -1
     loss), so credit assignment spans a whole match.
  3. PPO clipped policy update over minibatches; the value head is separate
     from the action scorer, over the state embedding.
  4. Evaluate in the arena; promote to models/registry/ on win-rate improvement.

Note the action space is dynamic — k differs every step — so old/new log-probs
must be computed over the *same* candidate ordering the engine produced at
collection time. Store the action matrix with each transition, not just the
chosen index.

TODO: implement
    train_rl(config) -> dict
    collect_rollouts(policy, opponents, num_steps)
    compute_gae(rewards, values, gamma, lam)
    ppo_update(policy, optimizer, batch, config)
"""
