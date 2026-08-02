"""Rule-based baseline agents.

These already exist in the vendored harness — re-export them here instead of
rewriting them, so there is one implementation to keep correct:

    pokemon-tcg-ai-battle/arena/agents.py
        random_agent    uniformly random legal selection
        first_agent     always take the first option
        greedy_agent    develops the board (evolve -> abilities -> play ->
                        attach energy) and only then attacks

    pokemon-tcg-ai-battle/arena/agents/iono/     Iono's Bellibolt ex deck
    pokemon-tcg-ai-battle/arena/agents/lucario/  Mega Lucario ex deck
        The two official sample agents, as Kaggle submission folders
        (main.py + deck.csv). Load with cabt_wrapper's load_submission.

Rough strength order: lucario > iono > greedy > first > random.

Worth knowing before writing another heuristic: attacking ends the turn, so an
attack-first agent swings before it is set up and loses to random ~87% of the
time. Develop the board first.

The strong sample agents (lucario, iono) are the interesting expert-data
sources for behavioral cloning; random/greedy are for sanity checks.

TODO: implement
    get_baseline(name: str) -> Agent    # resolve "random" | "greedy" | "lucario" | ...
    BASELINES: dict[str, Agent]
"""
