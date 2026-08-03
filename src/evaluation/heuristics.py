"""Rule-based baseline agents -- and the fallback that plays half the game.

Baselines already exist in the vendored harness; re-export rather than rewrite,
so there is one implementation to keep correct:

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

THE NON-MAIN FALLBACK
---------------------
The policy network scores SelectContext.MAIN only -- 50.6% of decisions
(spec §1). Everything else routes here:

    ATTACH_FROM  12.4%     TO_HAND  11.7%     ATTACH_TO  11.1%
    DISCARD       3.5%     TO_ACTIVE 2.6%     DISCARD_ENERGY_CARD 2.4%
    SWITCH, SETUP_*, TO_BENCH, IS_FIRST, ACTIVATE, DRAW_COUNT, ...

Two properties this code must have, in order of importance:

1. COMPLETE. Only 15 of the 49 SelectContext values appear in lucario-vs-iono
   games; the remaining 34 depend on cards outside those decks and WILL appear
   against other opponents. Any unhandled context yields an illegal selection,
   which forfeits the match. Terminate in a legal default for every context --
   never fall off the end.

2. MULTI-SELECT CORRECT. Unlike MAIN, these contexts routinely need other than
   exactly one index: maxCount > 1 in 4.9% of decisions overall (TO_BENCH 60%
   of the time, TO_HAND 26%, DISCARD 16%), and minCount is 0 in 468/5701 cases
   and 2-3 in 86 more. Return a list of length in [minCount, maxCount] with no
   duplicates. `_pick_n` in arena/agents.py already does this clamp -- reuse it.

Because the fallback plays half the game, the agent's ceiling is bounded by
its quality, not by the network's. TO_HAND in particular -- which card to fetch
from the deck -- is often the highest-leverage decision of the turn.

Worth knowing before writing another heuristic: attacking ends the turn, so an
attack-first agent swings before it is set up and loses to random ~87% of the
time. Develop the board first.

The strong sample agents (lucario, iono) are the interesting expert-data
sources for behavioral cloning; random/greedy are for sanity checks.

TODO: implement
    get_baseline(name: str) -> Agent    # resolve "random" | "greedy" | "lucario" | ...
    BASELINES: dict[str, Agent]
    fallback_select(obs) -> list[int]   # complete, legal handler for all 48 non-MAIN contexts
"""
