# Pokémon TCG AI Challenge

Agent for the [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
Kaggle competition: a **Dynamic Action-Scoring Policy Network** that ranks the *k* legal turn
options the engine offers, trained by behavioral cloning on replay data and evaluated in a local
arena before packaging for submission.

Target environment: 2 vCPUs, 12.2 GiB RAM, **CPU-only** evaluation, **< 10 ms** per turn.

## Layout

| Path | What lives there |
|------|------------------|
| `configs/` | Experiment and hyperparameter YAML (env, features, model, training) |
| `data/` | `raw/` card databases (EN/JP CSVs + card-ID reference PDFs), precomputed feature table, raw replays, tensor datasets |
| `models/` | `checkpoints/` from training runs; `registry/` for validated weights |
| `notebooks/` | Research and feature-analysis notebooks |
| `src/` | Core module: `card_db`, `env`, `models`, `dataset`, `training`, `evaluation` |
| `scripts/` | Pipeline entrypoints (steps 1–6 below) |
| `submission/` | Kaggle deployment sandbox — `build/` staging area and the final tarball |
| `docs/` | Architecture blueprint and technical specification |
| `pokemon-tcg-ai-battle/` | Vendored Kaggle bundle: native `cg` engine, sample submission, and a working local arena |

Everything under `data/` and `models/` is git-ignored except the card CSVs in
`data/raw/`, which are the tracked pipeline inputs. The two `Card_ID List_*.pdf`
references are ignored too — at 174 MB and 131 MB they exceed GitHub's 100 MB
per-file limit, and the CSVs carry the same data in a usable form.

## Pipeline

| Stage | Command | Description |
|-------|---------|-------------|
| 1. Setup | `pip install -r requirements.txt` | Install engine dependencies and PyTorch |
| 2. Feature precompute | `python scripts/build_static_features.py` | Generate `static_feature_table.pt` from the card CSV |
| 3. Data generation | `python scripts/generate_replays.py --games 5000` | Run self-play / heuristic matches to log moves |
| 4. Process dataset | `python scripts/process_dataset.py` | Format raw logs into PyTorch tensor training splits |
| 5. Train model | `python scripts/train_bc.py --config configs/train/bc_trainer.yaml` | Train the behavioral cloning model |
| 6. Arena evaluation | `python scripts/run_arena.py --candidate models/checkpoints/bc_epoch10.pt --baseline heuristic` | Measure win rate against baselines |
| 7. Package agent | `python scripts/package_submission.py` | Build and verify `submission.tar.gz` |

Data flows: `EN_Card_Data.csv` → `static_feature_table.pt` → replays → `.pt` datasets →
trained checkpoint → `models/registry/best_policy.pt` → `submission/submission.tar.gz`.

## Design rules

1. **CPU latency budget** — forward passes in `src/models/standalone_agent.py` must run in
   < 10 ms per turn. No heavy allocations inside `agent(obs, config)`.
2. **Deterministic precomputing** — never parse `EN_Card_Data.csv` inside the live turn loop.
   Read `static_feature_table.pt` instead.
3. **Modular encoder swapping** — `state_mlp.py` and `state_transformer.py` implement the same
   interface: `forward(obs_dict) -> Tensor[1, 256]`.
4. **Standalone inference** — the code in `submission/build/standalone_agent.py` must be fully
   self-contained, with no relative imports outside its deployment folder.

## Local testing

The vendored arena reproduces Kaggle's match loop on your machine. See
[`pokemon-tcg-ai-battle/arena/README.md`](pokemon-tcg-ai-battle/arena/README.md):

```bash
cd pokemon-tcg-ai-battle/arena
python tournament.py greedy random -n 200 --swap
```

`src/env/cabt_wrapper.py` and `src/evaluation/` wrap that harness rather than duplicating it.

## Specifications

- [`docs/PTCG Project Architecture & Workflow Specification.md`](docs/PTCG%20Project%20Architecture%20%26%20Workflow%20Specification.md) — repository layout, pipeline stages, MLOps workflow
- [`docs/Technical Specification.md`](docs/Technical%20Specification.md) — feature extraction, encoder architectures, tensor shapes
