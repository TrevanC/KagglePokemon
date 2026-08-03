# **Technical Specification: Pokémon TCG AI Battle Model Architecture & Feature Pipeline**

**Document Status:** Revised against the live engine — ready for implementation
**Target Environment:** Kaggle Pokémon TCG AI Battle Challenge (2 vCPUs, 12.2 GiB RAM, CPU-only evaluation)
**Card Pool Size (`N_cards`):** 1,267 unique card IDs (1..1267)
**Inference Latency Budget:** < 10 ms per turn

---

## **0. Revision note: what changed and why**

The original draft of this document was written from the competition write-up rather than
from the engine. Every dimension below was re-derived by querying `cg.api` directly and by
instrumenting real lucario-vs-iono matches (30–40 games, ~5,700 decisions per run). The
reproduction scripts are described in §9.

| Original claim | Measured reality | Consequence |
| :---- | :---- | :---- |
| 10 option types | **17** (`OptionType`) | action one-hot widened |
| 10 element types | **12** (`EnergyType`), incl. `RAINBOW`, `TEAM_ROCKET` | energy one-hots widened |
| 3 card categories | **7** (`CardType`) | category one-hot widened |
| 6 status flags incl. *Frozen* | **5**, no Frozen; stored on `PlayerState`, so they apply to the **Active only** | status block is 5-wide and only meaningful for the active slot |
| Base HP ÷ 340 | max HP is **380** | normalizer corrected |
| Damage ÷ 300 | max attack damage is **350** | normalizer corrected |
| Parse `EN_Card_Data.csv` | `all_card_data()` / `all_attack()` return typed records | **no CSV parsing at all** (§2) |
| Action target identified by `card_id` | `cardId` is **never populated** on MAIN options | the entire action encoding was rebuilt (§5) |
| k ∈ [1, 50] | k reaches **67** | K_MAX raised; truncation forbidden |
| One decision type | **49** `SelectContext` values; MAIN is **50.6%** of decisions | scope narrowed explicitly (§1) |

**The critical defect.** Under the original action encoding — `[type one-hot(10) | cardId/1267 |
slot/12 | 4 pads]` — **60.5% of MAIN decisions contained two or more options that encode to
byte-identical vectors**, covering **50.2% of all options**. The cause is that the engine
identifies option targets *positionally* (`area`/`index` into a zone, plus
`inPlayArea`/`inPlayIndex` for the target in play), and never by card ID. A representative
collision:

```
{type: ATTACH, area: HAND, index: 0, inPlayArea: ACTIVE, inPlayIndex: 0}
{type: ATTACH, area: HAND, index: 0, inPlayArea: BENCH,  inPlayIndex: 0}
```

— attach this energy to the Active attacker, versus to a benched Pokémon. Two strategically
opposite moves, one vector.

This is not a tuning problem. In behavioral cloning the target is an index into the k rows;
when rows *i* and *j* are identical the network is asked to emit different logits for identical
inputs, so the loss floors at `log(#duplicates)` and no amount of data, depth, or training
resolves it. §5 replaces the encoding.

---

## **1. Scope: which decisions the model makes**

The engine asks for 49 different kinds of selection, not one. Measured frequency over
5,701 decisions:

| SelectContext | Share | k range | needs multi-select |
| :---- | ----: | :---- | ----: |
| `MAIN` | 50.6% | 1..67 | 0% |
| `ATTACH_FROM` | 12.4% | 2..6 | 0% |
| `TO_HAND` | 11.7% | 1..21 | 26% |
| `ATTACH_TO` | 11.1% | 1..10 | 7% |
| `DISCARD` | 3.5% | 1..8 | 16% |
| `TO_ACTIVE`, `DISCARD_ENERGY_CARD`, `SWITCH`, `SETUP_*`, `TO_BENCH`, `IS_FIRST`, `ACTIVATE`, `DRAW_COUNT`, … | 10.7% | — | varies |

**This specification covers `SelectContext.MAIN` only.** All other contexts are served by a
rule-based fallback (`src/evaluation/heuristics.py`). Two consequences follow, and both are
load-bearing:

1. **The fallback plays roughly half the game.** It is not an edge case. The agent's ceiling is
   bounded by the quality of that hand-written half.
2. **The fallback must be complete, not merely good.** Only 15 of the 49 contexts appear in
   lucario-vs-iono play; the other 34 depend on cards outside those decks. An unhandled context
   produces an illegal selection, which **forfeits the match**.

**Multi-select is out of scope for the network.** MAIN is always `minCount == maxCount == 1`
(measured: 2,268/2,268). Selections requiring 0, or 2+, indices occur only in non-MAIN contexts
and are therefore entirely the fallback's responsibility. §7.3 specifies the general head for
when scope expands.

---

## **2. Static card features (offline)**

**Source: the engine, not the CSV.** `cg.api.all_card_data()` returns typed `CardData`
(`cardId, name, cardType, retreatCost, hp, weakness, resistance, energyType, basic, stage1,
stage2, ex, megaEx, tera, aceSpec, evolvesFrom, skills, attacks`) and `all_attack()` returns
`Attack` (`attackId, name, text, damage, energies`) for all 1,556 attacks. This avoids parsing
`{G}` symbols, `50+`/`20x` damage suffixes, and the CSV's 2,022-rows-for-1,267-cards layout
(one row per attack), and is guaranteed consistent with the engine judging the match.

`EN_Card_Data.csv` remains tracked as a human-readable reference and as a fallback source; it is
not on the runtime path.

### **2.1 `StaticCardFeatureTable` — shape (1268, 72)**

Row index **is** the card ID. Row 0 is left zeroed so IDs 1..1267 index directly, eliminating
the off-by-one convention that must otherwise be kept in sync across the extractor, the table
builder, and the standalone agent.

| Index | Feature | Encoding |
| :---- | :---- | :---- |
| 0..6 | `cardType` one-hot | POKEMON, ITEM, TOOL, SUPPORTER, STADIUM, BASIC_ENERGY, SPECIAL_ENERGY |
| 7..9 | Stage one-hot | `basic`, `stage1`, `stage2` |
| 10..13 | Rule flags | `ex`, `megaEx`, `tera`, `aceSpec` |
| 14 | Base HP | `hp / 380.0` |
| 15 | Retreat cost | `retreatCost / 4.0` |
| 16..27 | `energyType` one-hot | 12 `EnergyType` values |
| 28..39 | Weakness one-hot | 12; all-zero when `None` |
| 40..51 | Resistance one-hot | 12; all-zero when `None` |
| 52 | Attack count | `len(attacks) / 2.0` (max 2) |
| 53..54 | Attack 0 | `damage / 350.0`, `len(energies) / 5.0` |
| 55..56 | Attack 1 | same; zeros when absent |
| 57..68 | Attack cost histogram | count of each `EnergyType` across all attacks, ÷ 5.0 |
| 69 | Has ability | `len(skills) > 0` |
| 70 | Evolves from something | `evolvesFrom is not None` |
| 71 | Reserved | 0.0 |

Built once by `scripts/build_static_features.py` → `data/processed/static_feature_table.pt`.
**Design rule: never call `all_card_data()` or read the CSV inside the turn loop.**

---

## **3. Dynamic card features — 24 dims**

Per Pokémon instance in play, from `Pokemon` and its owning `PlayerState`:

| Index | Feature | Encoding |
| :---- | :---- | :---- |
| 0 | Current HP ratio | `hp / maxHp` |
| 1 | Max HP | `maxHp / 380.0` |
| 2 | Attached energy count | `len(energies) / 5.0` |
| 3..14 | Attached energy histogram | count per `EnergyType` ÷ 5.0 |
| 15 | Tools attached | `len(tools)` (max 1 observed) |
| 16 | Pre-evolution depth | `len(preEvolution) / 2.0` |
| 17 | Played this turn | `appearThisTurn` |
| 18 | Is Active | 1.0 for the active slot |
| 19..23 | Status flags | `poisoned, burned, asleep, paralyzed, confused` — from `PlayerState`, so **non-zero only in the active slot** |

Empty bench slots are the zero vector.

### **3.1 Card instance vector**

```
e_card = [ f_static (72) ; e_id (16) ; f_dynamic (24) ]  ∈ R^112
         ── Linear(112 → 64) → ReLU → Linear(64 → 64) ──►  R^64
```

`e_id = nn.Embedding(1268, 16)`. The static table is a **registered buffer**, never a parameter —
it must not receive gradients, and it must travel with the checkpoint.

---

## **4. Board state — 984 dims**

`benchMax` is 5 for both players in every observed game, so the 12 fixed field slots are sound.

| Group | Dims | Content |
| :---- | ----: | :---- |
| Field slots | 12 × 64 = **768** | self Active, self Bench 0..4, opp Active, opp Bench 0..4; unoccupied slots are zeros |
| Hand | **64** | mean-pool of self hand card vectors; zeros when empty. `PlayerState.hand` is `None` for the opponent — only `handCount` is observable |
| Self discard | **64** | 1268-dim count histogram ÷ 45.0 → `Linear(1268, 64)` → ReLU |
| Opp discard | **64** | same projection, separate parameters |
| Global scalars | **24** | §4.1 |
| **Total** | **984** | |

### **4.1 Global scalars (24)**

`self_prizes/6`, `opp_prizes/6` — note prize cards are facedown, so the count is `len(prize)`,
**not** a count of non-`None` entries; `self_deck/60`, `opp_deck/60`; `self_hand/13`,
`opp_hand/13`; `turn/32`; `turnActionCount/72`; `yourIndex`; `firstPlayer == yourIndex`;
`supporterPlayed`; `stadiumPlayed`; `energyAttached`; `retreated`; `stadium present`;
`self_bench/5`; `opp_bench/5`; `self_discard/45`; `opp_discard/45`; `minCount/5`; `maxCount/5`;
`k/80`; two reserved slots. (22 populated + 2 reserved = 24.)

---

## **5. Action encoding — the corrected design**

Each of the k candidate options is encoded as three concatenated blocks, then projected:

```
a_i = [ structural (64) ; v_source_card (64) ; v_target_slot (64) ]  ∈ R^192
      ── Linear(192 → 64) → ReLU → Linear(64 → 64) ──►  R^64
```

The two learned blocks are what make the encoding injective. Both reuse machinery already
built for the board state — no new parameters beyond the projection.

### **5.1 Structural block (64)**

| Index | Field | Encoding |
| :---- | :---- | :---- |
| 0..16 | `type` one-hot | 17 `OptionType` values |
| 17..29 | `area` one-hot | 13 = 12 `AreaType` + "absent" at index 0 |
| 30 | `index` | `/ 48.0` (observed 0..45) |
| 31..43 | `inPlayArea` one-hot | 13, same convention |
| 44 | `inPlayIndex` | `/ 5.0` (observed 0..4) |
| 45 | Has attack | `attackId is not None` |
| 46..47 | Attack stats | `damage/350`, `len(energies)/5` via `ATTACKS[attackId]` |
| 48..59 | Attack cost histogram | per `EnergyType`, ÷ 5.0 |
| 60 | `energyIndex` | `/ 5.0` |
| 61 | `toolIndex` | as-is |
| 62 | `number` | `/ 8.0` |
| 63 | Targets self | `playerIndex == yourIndex` |

### **5.2 Source card block (64)**

Resolve `(area, index)` against `obs.current` to the card the option acts *with* — hand card,
discard card, deck card, stadium — and embed it with the **same** `HybridCardEncoder` used for
the board (§3.1), with dynamic features zeroed for cards not in play. Zeros when `area` is
absent or the reference cannot be resolved.

This is what tells the model *which* card is being played, information the original encoding
discarded entirely.

### **5.3 Target slot block (64)**

When `inPlayArea`/`inPlayIndex` are present, reuse the **already-computed field-slot vector**
for that target (one of the 12 slots from §4) — a gather, not a recomputation, so it costs
nothing at inference. Zeros when absent.

This is what distinguishes "attach to Active" from "attach to Bench 0", the exact collision that
broke the original encoding.

### **5.4 Injectivity requirement**

> Any two options within one decision that differ in engine semantics **must** differ in their
> encoded vector. This is a testable invariant, not an aspiration: assert a zero collision rate
> over a sample of real decisions before training. Regressions here are silent — they degrade
> play strength while training metrics continue to improve.

---

## **6. Board state encoders**

Both implement the same interface, `forward(obs_dict) -> Tensor[B, 256]`, so they remain
interchangeable without touching the scorer or the submission code.

### **6.1 Baseline MLP**

```
raw state (B, 984) ─► Linear(984, 512) ─► ReLU ─► Linear(512, 256) ─► s (B, 256)
```

### **6.2 Transformer**

Tokens, each summed with a learned zone embedding `nn.Embedding(16, 128)`:

```
token 0        [CLS]                zone 0
tokens 1..12   12 field slots       zones 1..12
tokens 13..    self hand cards      zone 13
               self/opp discard     zone 14
last           global scalars       zone 15
```

`nn.TransformerEncoder(num_layers=2, d_model=128, nhead=4, dim_feedforward=256,
activation="relu", batch_first=True)`; take the updated `[CLS]`, then
`Linear(128, 256) → ReLU → Linear(256, 256)`.

Card vectors are 64-dim and tokens are 128-dim, so a `Linear(64, 128)` lift precedes assembly.
Hand size varies (max 13, p99 11), so batched training requires padding plus a
`src_key_padding_mask` — without the mask, padding attends and silently corrupts `[CLS]`.

---

## **7. Joint scorer and output head**

### **7.1 Scoring**

```
s (B, 256) ──unsqueeze/expand──► (B, k, 256)
A_emb (B, k, 64)
        └─► cat(dim=-1) ─► (B, k, 320)
            ─► Linear(320, 128) → ReLU → Linear(128, 1) ─► (B, k, 1) ─► squeeze ─► logits (B, k)
```

Padded candidates are masked with `logits.masked_fill(~mask, -1e9)` **before** any softmax.
Omitting the mask leaks probability mass onto padding and biases every gradient.

### **7.2 MAIN head (single-select)**

`F.softmax(logits, dim=-1)`; training loss is cross-entropy against the expert's chosen index:

```
L_BC = -log P(a_y | s_t)
```

At inference return `[argmax]`. Valid because MAIN is always `minCount == maxCount == 1`.

### **7.3 General head (for future scope expansion)**

When contexts with `maxCount > 1` are brought into the model, the same scorer is reused with a
multi-hot target and per-candidate `BCEWithLogitsLoss`. Selection becomes: sort logits
descending, take the top *m*, where

```
m = clamp(#(sigmoid(logit) > 0.5), minCount, maxCount)
```

`minCount == 0` is legal and means the empty selection may be returned. **Returning a selection
outside `[minCount, maxCount]`, or containing duplicates, forfeits the match** — this must be
enforced by a final clamp in the agent, independent of what the network produces.

### **7.4 Candidate count**

`K_MAX = 80` (observed max 67; mean 11.0, p99 34). Padding for batching is fine.
**Truncation is not**: dropping candidates can remove the expert's chosen move, leaving the
target index undefined and injecting label noise. Decisions with `k > K_MAX` are excluded from
the training set and handled by the fallback at inference.

---

## **8. Tensor shape reference**

| Module | Input | Input shape | Transformation | Output shape |
| :---- | :---- | :---- | :---- | :---- |
| Static lookup | `card_id` | (N,) | `StaticCardFeatureTable[id]` | (N, 72) |
| Card instance | static+id+dynamic | (N, 112) | `Linear(112,64)→ReLU→Linear(64,64)` | **(N, 64)** |
| Discard projection | histogram | (B, 1268) | `Linear(1268,64)+ReLU` | (B, 64) |
| MLP encoder | raw state | (B, 984) | `Linear(984,512)→ReLU→Linear(512,256)` | **(B, 256)** |
| Transformer encoder | tokens | (B, L, 128) | `TransformerEncoder(2)`→`[CLS]`→proj | **(B, 256)** |
| Action structural | option fields | (B, k, 64) | — | (B, k, 64) |
| Action assembly | struct+source+target | (B, k, 192) | `Linear(192,64)→ReLU→Linear(64,64)` | **(B, k, 64)** |
| State expansion | `state_emb` | (B, 256) | `unsqueeze(1).expand(-1,k,-1)` | (B, k, 256) |
| Fusion | [state, action] | (B,k,256),(B,k,64) | `cat(dim=-1)` | **(B, k, 320)** |
| Scorer | joint | (B, k, 320) | `Linear(320,128)→ReLU→Linear(128,1)` | (B, k, 1) |
| Squeeze + mask | raw scores | (B, k, 1) | `squeeze(-1)`, `masked_fill` | **(B, k)** |
| Softmax | logits | (B, k) | `F.softmax(dim=-1)` | **(B, k)** |

---

## **9. Reproducing the measurements**

Every number in §0–§7 came from the live engine. To re-derive after an engine update:

- **Enum widths and card DB** — `all_card_data()`, `all_attack()`, and `len(list(OptionType))`
  etc. Yields 1,267 cards / 1,556 attacks / max HP 380 / max damage 350 / max retreat 4 /
  max 2 attacks per card / max 5 energies per attack.
- **Context distribution, k ranges, min/maxCount** — wrap both agents in `run_match` with a
  recorder that tallies `obs["select"]` per `SelectContext`, over ≥30 lucario-vs-iono games.
- **Encoding injectivity (§5.4)** — for each MAIN decision, hash each option under the candidate
  encoding and count decisions containing duplicate hashes. **The target is 0%.** The original
  encoding scored 60.5%.

Treat the injectivity check as a permanent regression test, not a one-off: it is the only cheap
signal that distinguishes "the features are broken" from "the model needs more training", and
the two look identical on a loss curve.
