"""Standardized wrapper around the Kaggle cabt engine.

Does NOT reimplement the engine. The vendored harness at
pokemon-tcg-ai-battle/arena already provides a working local match loop with
the native `cg` bindings; this module puts that directory on sys.path and
re-exports it behind a stable interface so the rest of src/ never depends on
the vendor directory's layout.

What to reuse (see pokemon-tcg-ai-battle/arena/README.md):

    arena/runner.py     run_match(agent0, agent1, deck0, deck1, verbose=False) -> MatchResult
                        read_deck(path) -> list[int]
                        load_submission(dir) -> (agent, deck)   # Kaggle main.py + deck.csv
    arena/tournament.py many-game driver reporting win rates with Wilson CIs
    arena/cards.py      CARDS / ATTACKS lookups, name(), describe_deck()
    arena/cg/api.py     observation & option schema, to_observation_class(obs_dict)

Agent contract, identical to Kaggle:

    def agent(obs_dict: dict) -> list[int]

returning option indices into obs["select"]["option"], of length between
minCount and maxCount, no duplicates. Illegal moves forfeit the match.

Caveats inherited from the harness: the engine's RNG is not seedable (run many
games for stable numbers), and both agents share one process — avoid pitting
two agents that use the `search_*` planning API against each other locally.

TODO: implement
    arena_path() -> Path                 # resolve + sys.path-insert the vendored arena
    play_match(agent0, agent1, ...)      # thin pass-through to run_match
    load_agent(path_or_callable)         # accept a submission dir, a baseline name, or a callable
"""
