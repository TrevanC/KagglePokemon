# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Agent for the [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
Kaggle competition. The repo is currently **a scaffold, not a working pipeline**. Know which half
of the tree you are in:

- **Working code**: `pokemon-tcg-ai-battle/` (vendored Kaggle bundle — native engine, local arena,
  baseline agents) and `src/dataset/records.py`.
- **Documented stubs**: everything else in `src/` and all of `scripts/`. Modules are docstring-only;
  every script raises `NotImplementedError` under `__main__` **by design**, so an accidental run
  fails loudly. A script that raises is not a bug to fix in passing — it is unimplemented.

Each stub's docstring carries its contract (tensor shapes, feature index maps, I/O paths) drawn
from `docs/`. Read the stub before implementing it; the design decisions are already recorded there.

## Environment and commands

`.venv/` is the project venv (Python 3.12) with everything in `requirements.txt` installed —
`torch`, `numpy`, `PyYAML`, `wandb`, `tqdm`. Use `.venv/bin/python` explicitly.

There is **no test suite, no linter, and no build step** in this repo. Do not invent commands for
them. Verification is done by running the arena and by importing modules:

```bash
# Import check across all packages
.venv/bin/python -c "import src, src.card_db, src.env, src.models, src.dataset, src.training, src.evaluation"
```

### Local arena (the real way to test an agent)

The scripts run from any working directory (`runner.py` puts its own directory on `sys.path` to find
the native engine, and the default deck path is absolute). But **agent-spec paths are resolved
relative to your cwd**, so the relative paths below — and every example in the arena README — assume
you are in the arena directory.

```bash
cd pokemon-tcg-ai-battle/arena

python runner.py                                  # watch one game step by step
python tournament.py greedy random -n 200 --swap  # win rates + 95% Wilson CIs
python tournament.py ../sample_submission/sample_submission greedy -n 100   # a submission folder
python tournament.py agents/iono agents/lucario -n 200 --swap
python tournament.py greedy greedy --deck0 a.csv --deck1 b.csv -n 100       # differing decks
python cards.py ../sample_submission/sample_submission/deck.csv             # inspect a deck
```

Agent specs resolve to a baseline name (`random`, `first`, `greedy`) or a path to a Kaggle
submission folder containing `main.py` + `deck.csv`.

### Pipeline (all stubs today)

```bash
python scripts/build_static_features.py                                  # 1. CSV -> (1267,32) tensor
python scripts/generate_replays.py --num_games 1000 --bot1 greedy --bot2 random   # 2. replays
python scripts/process_dataset.py                                        # 3. replays -> .pt datasets
python scripts/train_bc.py --config configs/train/bc_trainer.yaml        # 4. behavioral cloning
python scripts/run_arena.py --candidate models/checkpoints/x.pt --baseline heuristic  # 5. eval
python scripts/package_submission.py                                     # 6. submission.tar.gz
```

## Architecture

### The agent contract

Identical to Kaggle's, and the same in the arena:

```python
def agent(obs_dict: dict) -> list[int]   # option indices into obs["select"]["option"]
```

The returned list must be within `[minCount, maxCount]` with no duplicates. **An illegal move
forfeits the match.** On the very first call `obs["select"]` is `None` — the agent must return its
60-card deck (a list of 60 card IDs) instead of option indices. The full observation/option schema
lives in `pokemon-tcg-ai-battle/arena/cg/api.py`; convert with `to_observation_class(obs_dict)`.

### The model: dynamic action scoring

The network ranks the *k* legal options the engine offers, rather than emitting a fixed action
space (`docs/Technical Specification.md`, revised against the live engine):

```
obs -> board state (B, 984) -> state encoder -> s (B, 256) -.
                                                             |- expand to (B,k,256), concat
obs -> action matrix (B,k,192) -> action encoder -> (k,64) -'  -> (k,320) -> MLP -> logits (k,)
                                                                            -> mask -> softmax
```

`984 = 12 slots x 64 + hand 64 + self discard 64 + opp discard 64 + scalars 24`. Each action row is
three 64-dim blocks: structural option fields, the **source card** resolved from `(area, index)`,
and the **target slot** gathered via `(inPlayArea, inPlayIndex)`. Training is cross-entropy over the
*k* candidates against the index the expert chose.

**The action encoding is load-bearing.** The engine never populates `cardId` on MAIN options — it
identifies targets positionally. An earlier design encoding only `(type, cardId, slot)` made 50.2%
of options byte-identical to a sibling, which is unlearnable rather than merely inaccurate. Spec
§5.4 makes zero encoding collisions a testable invariant; treat it as a permanent regression test,
since broken features and an undertrained model look identical on a loss curve.

Three invariants that span multiple files:

1. **One featurizer for both paths.** `src/env/state_extractor.py` must produce identical tensors
   for live inference (reading `obs.current` / `obs.select.option`) and for training (reconstructing
   from replay logs). Reimplementing extraction in `scripts/process_dataset.py` would silently
   desync training from deployment — the failure is invisible in metrics.
2. **Encoders are swappable.** `state_mlp.py` and `state_transformer.py` implement the same
   interface, `forward(obs_dict) -> Tensor[1, 256]`, so swapping never touches the scorer or the
   submission code.
3. **One record format.** `src/dataset/records.py` defines the gzipped-JSONL record
   (`obs`/`action`/`player`/`reward`/`source`) that both locally generated self-play and downloaded
   Kaggle episode replays produce. Write through `records.write_records()`; keeping the format in
   one place is what lets the two data sources mix freely.

### Deployment constraints

Kaggle evaluates on 2 vCPUs, CPU-only, with a **< 10 ms per-turn** budget.

- Never parse `EN_Card_Data.csv` in the turn loop — read the precomputed `static_feature_table.pt`.
- `submission/build/standalone_agent.py` must be **fully self-contained**: no imports reaching
  outside the deployment folder. This is the one place code duplication is correct — inline whatever
  encoder/extractor code it needs rather than importing from `src/`.
- Load the model and static table once at construction; no per-turn allocations, `torch.no_grad()`,
  `model.eval()`.

### `src/env/` and `src/evaluation/` wrap, not duplicate

These reach into `pokemon-tcg-ai-battle/arena/` (`run_match`, `read_deck`, `load_submission` from
`runner.py`; `random`/`first`/`greedy` from `agents.py`) instead of reimplementing a match loop.
Add the arena to `sys.path` rather than copying the engine bindings.

## Kaggle episode replays (expert data)

The official CLI has simulation-competition commands — they are not listed in the top-level
`kaggle --help`, only under `kaggle competitions --help`:

```bash
kaggle competitions episodes <submission_id>       # list episodes for one of YOUR submissions
kaggle competitions replay <episode_id> -p ./dir   # download episode-<id>-replay.json
kaggle competitions logs <episode_id> <agent_idx>  # agent stderr/stdout for debugging
```

`kaggle auth login` (OAuth, browser flow) is enough — no session cookie or scraping. The internal
`competitions.EpisodeService/*` HTTP endpoints reject both anonymous and OAuth-bearer requests;
don't go down that path.

**`episodes` takes a SUBMISSION id — and works for other people's submissions.** This is the whole
unlock, and it is easy to miss: a *team* id returns "No episodes found", which makes it look like
the command only works on your own account. Submission ids are exposed in leaderboard URLs when you
click a team on the competition leaderboard page:

```
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard?submissionId=55186239&episodeId=89625016
                                                                                  ^^^^^^^^
```

So no submission of your own is needed to harvest expert data. `scripts/fetch_kaggle_replays.py`
wraps this: give it submission ids, it enumerates episodes and downloads replays with rate
limiting and resume. One top-ranked submission yielded **178 episodes**, ~145 decisions each
(~3-5 MB per replay, so ~700 MB per submission — `data/replays/` is gitignored).

Meta Kaggle (`Episodes.csv` 6.3 GB, `EpisodeAgents.csv` 21.8 GB) remains the alternative global
index, but is not needed for this workflow.

**Each replay contains BOTH players' decisions**, and `replay["rewards"]` identifies the winner
(`[-1, 1]` → player 1 won). Measured on episode 89083821: 120 decisions total, ~60 MAIN, split
roughly evenly per player. So submitting even a mediocre agent yields training data from the
*strong opponents* it faces — filter to the winner's side.

> **Action alignment is off by one.** The action responding to the observation at `steps[i][p]`
> is stored at `steps[i+1][p]["action"]`. Verified on a real replay: 120/120 legal under `+1`,
> **79/120 under `+0`**. The naive alignment therefore looks *mostly* valid — it will not crash,
> will not fail a legality check on most rows, and will silently train the policy on two-thirds
> wrong labels. Any replay loader must use the `+1` offset and assert legality
> (`minCount <= len(action) <= maxCount`, no duplicates, all indices `< len(option)`) on every row.

## Gotchas

**Engine RNG is not seedable.** The engine owns all shuffles and coin flips, so individual games
vary and results are not reproducible. Always run many games and read the Wilson CIs; use `--swap`
to cancel position bias. A model's promotion gate is arena win rate, *not* validation loss.

**Both agents share one process** in the arena. Fine for a model vs a baseline, but two agents both
using the `search_*` planning API in `cg/api.py` share one engine `agent_ptr` — don't run those
against each other locally.

**Attacking ends the turn.** An attack-first agent swings before it is set up and loses to `random`
about 87% of the time. `greedy` develops the board (evolve → abilities → play → attach energy) and
only then attacks. Strength order: `lucario > iono > greedy > first > random`.

**Card data comes from the engine, not the CSV.** `all_card_data()` and `all_attack()` return typed
records; the static table is `(1268, 72)` indexed directly by card ID, with row 0 zeroed so 1-based
IDs need no off-by-one convention. `data/raw/EN_Card_Data.csv` is a human-readable reference only —
and note it is *not* one row per card: 2,022 rows for 1,267 IDs, one row per attack.

**The engine asks 49 kinds of question, and the model answers one.** `SelectContext.MAIN` is 50.6%
of decisions; the other 48 contexts route to the heuristic fallback in `src/evaluation/heuristics.py`,
which therefore plays half the game and caps the agent's ceiling. That fallback must be *complete* —
only 15 contexts appear in lucario-vs-iono play, and an unhandled one produces an illegal selection,
which forfeits. MAIN is always `minCount == maxCount == 1`, so multi-select (`maxCount > 1` in 4.9%
of decisions, `minCount == 0` in 8%) is entirely the fallback's problem.

**Never trust a dimension from the original spec draft.** Nearly all were wrong: 17 option types not
10, 12 energy types not 10, 7 card types not 3, 5 status flags not 6 (no Frozen, and they sit on
`PlayerState` so they apply to the Active only), max HP 380 not 340, max damage 350 not 300, k up to
67 not 50. Spec §0 tabulates the corrections and §9 says how to re-derive them after an engine
update.

**Native engine libraries are platform-specific.** `cg/` ships `libcg.dylib` (arm64 macOS),
`libcg.so`, `libcg-arm64.so`, and `cg.dll`. Anything cross-platform must account for this.

**`.gitignore` uses `data/**` + explicit negations.** Adding a file under `data/`, `models/`, or
`submission/build/` that should be tracked requires an explicit `!` line — otherwise it is silently
ignored. The paired `!data/**/` lines exist because git will not descend into an excluded directory,
so without them no negation inside could ever match. The two `data/raw/Card_ID List_*.pdf` files
(174 MB, 131 MB) must stay ignored: they exceed GitHub's 100 MB per-file hard limit.

## Reference docs

- `docs/PTCG Project Architecture & Workflow Specification.md` — layout, pipeline stages, packaging
- `docs/Technical Specification.md` — feature extraction, encoder architectures, tensor shape tables
- `docs/behavioral_cloning_trainer.py` — a working reference BC trainer (not wired into `src/`)
- `pokemon-tcg-ai-battle/arena/README.md` — arena usage, baseline strength, limitations
