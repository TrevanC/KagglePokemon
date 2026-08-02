"""Local ladder: multi-game bot-vs-bot tournaments.

Built on top of the vendored harness rather than a new match loop —
pokemon-tcg-ai-battle/arena/tournament.py already runs many games and reports
win rates with 95% Wilson confidence intervals, and arena/runner.py plays a
single match. Reach them through src/env/cabt_wrapper.py.

What this module adds on top:
  - Loading a candidate checkpoint (models/checkpoints/*.pt) as a playable
    agent and pitting it against the baselines in heuristics.py.
  - Round-robin ladders across several checkpoints, for RL opponent pools.
  - The promotion decision: does the candidate beat the incumbent in
    models/registry/ by enough margin to replace it?

Two things the harness README warns about, which the promotion logic depends
on:
  - The engine's RNG is not seedable, so single games are noisy. Use enough
    games that the confidence interval is narrower than the margin you care
    about, and always swap sides to cancel position bias.
  - Both agents run in one process. Fine for a model vs a baseline; avoid two
    agents that both use the engine's search_* planning API.

Reference baseline strength (from arena/README.md):
    lucario > iono > greedy > first > random

TODO: implement
    run_arena(candidate, baseline, num_games, swap=True) -> ArenaResult
    round_robin(agents, num_games) -> win-rate matrix
    should_promote(result, threshold) -> bool
"""
