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

THE NON-MAIN FALLBACK IS DELEGATION, NOT NEW CODE
-------------------------------------------------
The policy network scores SelectContext.MAIN only -- 50.6% of decisions
(spec §1). Everything else routes to an existing complete agent:

    def agent(obs):
        sel = obs["select"]
        if sel is None:                       # initial deck request
            return read_deck()
        if sel["context"] == int(SelectContext.MAIN) and len(sel["option"]) <= K_MAX:
            return [policy.rank(obs).argmax()]
        return baseline(obs)                  # already legal for all 49 contexts

Do NOT write a per-context handler. Both baselines are legal *by
construction* for any of the 49 contexts, including the 34 that never appear
in lucario-vs-iono play:

    greedy_agent   terminates in `_pick_n(select, ...)`, which dedupes, caps at
                   maxCount, and pads up to minCount from the remaining legal
                   indices. It cannot fall off the end.
    lucario        scores every option generically and returns
                   `desc_indices[:select.maxCount]`; since maxCount <=
                   len(option), the result is always within [minCount, maxCount].

That makes multi-select correct for free too -- maxCount > 1 in 4.9% of
decisions (TO_BENCH 60% of the time, TO_HAND 26%, DISCARD 16%) and minCount is
0 in 8% -- which is exactly the arithmetic that is easy to get wrong by hand.

WHICH baseline to delegate to is a strength decision, not a legality one:

    lucario   much stronger, but its scoring references specific card IDs and
              its own decklist. Correct choice if the agent plays lucario's
              deck; degrades (though stays legal) on a different deck.
    greedy    deck-agnostic and safe, but crude where it matters most: for
              TO_HAND it just takes the first maxCount options, and "which card
              to fetch from the deck" is often the highest-leverage decision of
              the turn.

CONSEQUENCE WORTH INTERNALIZING: with lucario as the fallback and a policy
trained by behavioral cloning ON lucario, the agent is lucario everywhere
except MAIN, where it is an imperfect imitation of lucario. That is a strictly
worse lucario. The ML only earns its place when MAIN is trained to *beat*
lucario -- from stronger data (top Kaggle replays) or from self-play RL.

Worth knowing before writing another heuristic: attacking ends the turn, so an
attack-first agent swings before it is set up and loses to random ~87% of the
time. Develop the board first.

The strong sample agents (lucario, iono) are the interesting expert-data
sources for behavioral cloning; random/greedy are for sanity checks.

TODO: implement
    get_baseline(name: str) -> Agent    # resolve "random" | "greedy" | "lucario" | ...
    BASELINES: dict[str, Agent]
    route(obs, policy, baseline) -> list[int]   # MAIN -> policy, everything else -> baseline
"""
