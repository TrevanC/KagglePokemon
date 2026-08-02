# Local Arena — Pokémon TCG AI Battle

A self-contained local testing environment for the
[Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
Kaggle competition. Kaggle runs matches server-side; this reproduces that loop
on your machine so you can pit agents against each other and measure win rates
before submitting.

## What's here

| File | Purpose |
|------|---------|
| `cg/` | The competition's native game engine (copied from the sample submission) + its Python bindings. `libcg.dylib` is the arm64 macOS build. |
| `runner.py` | `run_match(...)` — plays one game between two agents. `load_submission(...)` — loads a Kaggle `main.py` + `deck.csv` folder. |
| `agents.py` | Baseline rule-based agents you can import: `random_agent`, `first_agent`, `greedy_agent`. |
| `agents/iono/`, `agents/lucario/` | The two official sample rule-based agents (extracted from the `.ipynb` notebooks), each as a Kaggle submission folder (`main.py` + `deck.csv`). |
| `tournament.py` | CLI to run many games and report win rates with confidence intervals. |
| `cards.py` | Look cards/attacks up by ID (`CARDS`, `ATTACKS`, `name()`, `describe_deck()`). |

## Sample agents

The two competition sample notebooks at the repo root ship as full heuristic
agents with their own decks. They're extracted here into runnable folders:

- **`agents/iono`** — Iono's Bellibolt ex deck. Loads a Pokémon with lots of
  Lightning Energy to swing big with Voltaic Chain.
- **`agents/lucario`** — Mega Lucario ex deck. Switches between Mega Lucario ex,
  Hariyama, and Solrock depending on the board.

Both are strong scoring-based agents and make good sparring partners. Each was
regenerated from its notebook: `main.py` is the notebook's `%%writefile` cell
and `deck.csv` is rebuilt from the decklist embedded in that cell.

## The agent contract

Identical to Kaggle. An agent is any callable:

```python
def agent(obs_dict: dict) -> list[int]:
    ...  # return option indices into obs["select"]["option"]
```

The returned list must have length between `minCount` and `maxCount`, with no
duplicate indices. The full observation schema (state, options, logs, enums) is
documented in `cg/api.py`. Convert the dict to typed dataclasses with
`cg.api.to_observation_class(obs_dict)`.

## Quick start

```bash
cd arena

# Watch one game step by step:
python runner.py

# Benchmark the baselines (sides swapped to cancel position bias):
python tournament.py greedy random -n 200 --swap

# Test a real Kaggle submission folder against a baseline:
python tournament.py ../sample_submission/sample_submission greedy -n 100

# The two official sample agents head-to-head:
python tournament.py agents/iono agents/lucario -n 200 --swap

# Two different decks head-to-head:
python tournament.py greedy greedy --deck0 a.csv --deck1 b.csv -n 100

# Inspect what's in a deck:
python cards.py ../sample_submission/sample_submission/deck.csv
```

Use it as a library:

```python
from runner import run_match, read_deck, load_submission
from agents import greedy_agent

deck = read_deck("../sample_submission/sample_submission/deck.csv")
my_agent, my_deck = load_submission("path/to/my_submission")

result = run_match(my_agent, greedy_agent, my_deck, deck)
print(result)   # e.g. "player 0 wins (opponent took all prize cards) in 22 turns"
```

## Baseline strength (≈ reference)

Measured over 100+ games on the sample deck:

- `agents/lucario` and `agents/iono` beat `greedy`/`random` ~80-98%
- `lucario` beats `iono` ~65% (sides swapped)
- `greedy` beats `random` ~85%, beats `first` ~68%
- `first` (take the first option) beats `random` ~79%
- `random` vs `random` ≈ 50/50 (no built-in player-0 advantage)

Rough strength order: **lucario > iono > greedy > first > random.**

`greedy` develops its board first (evolve → abilities → play → attach energy)
and only **then** attacks — attacking ends the turn, so an attack-first agent
swings before it's set up and loses badly (~13% vs random). This is a useful
gotcha to keep in mind when writing your own agent.

## Notes / limitations

- **RNG is not seedable.** The engine owns all shuffles and coin flips, so
  individual games vary. Run many games (`-n`) for stable win rates —
  `tournament.py` prints 95% Wilson confidence intervals.
- **Single process.** Both agents run in one Python process (fast, good for
  bulk games). Kaggle isolates each agent in its own process; if your agent
  uses the `search_*` planning API in `cg/api.py`, it shares one engine
  `agent_ptr` here, so avoid running two searching agents against each other
  locally. One searching agent vs a baseline is fine.
- **Decks.** The sample submission ships one `deck.csv`. To test a matchup of
  two different decks, pass `--deck0` / `--deck1` (each must be 60 legal cards;
  the engine rejects illegal decks at `battle_start`).
- **Illegal moves = forfeit**, mirroring Kaggle. `run_match` reports which
  agent forfeited and why via the `MatchResult.error` field.
```
