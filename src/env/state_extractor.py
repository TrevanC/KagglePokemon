"""Convert a raw observation dict into model input tensors.

All dimensions here were measured against the live engine (spec §0, §9), not
inferred from the competition write-up. The v1 numbers were wrong in almost
every field; do not "restore" them.

SCOPE: SelectContext.MAIN only (50.6% of decisions). The other 48 contexts go
to the rule-based fallback in src/evaluation/heuristics.py. See spec §1.

Two outputs per MAIN decision:

  1. Board state vector, 984 dims (spec §4):

        12 field slots x 64      = 768   self Active + Bench 0..4, opp Active +
                                         Bench 0..4 (benchMax measured as 5 for
                                         both players); empty slots are zeros
        self hand mean-pool      =  64   PlayerState.hand is None for the
                                         opponent -- only handCount is visible
        self discard projection  =  64   1268-dim count histogram / 45.0
                                         -> Linear(1268, 64) -> ReLU
        opp discard projection   =  64   same shape, separate parameters
        global scalars           =  24   see spec §4.1
        ---------------------------------
        total                    = 984

     Prize count is len(prize) -- prize cards are facedown, so every entry is
     None and counting non-None yields 0.

  2. Candidate action matrix, (k, 192) for the k legal options (spec §5).
     Three concatenated blocks: structural (64), source card (64), target
     slot (64).

WHAT THE ENGINE ACTUALLY PUTS ON A MAIN OPTION
----------------------------------------------
Measured over 4,000 real MAIN decisions from data/datasets (7 option types
appear; DISCARD is legal but unobserved). `cardId` is populated on NONE of
them -- the engine identifies cards positionally:

    PLAY     16011 options   index                                  (hand index)
    ATTACH   13103 options   area, index, inPlayArea, inPlayIndex
    END       4000 options   -
    EVOLVE    2774 options   area, index, inPlayArea, inPlayIndex
    ABILITY   2513 options   area, index
    RETREAT   2510 options   -
    ATTACK    2307 options   attackId

That table is why the source-card and target-slot blocks exist. Encoding only
(type, cardId, slot) -- the v1 design -- made 50.2% of options byte-identical
to a sibling and 60.5% of MAIN decisions unlearnable. Two consequences:

  * PLAY carries no `area`; the index is implicitly into the hand.
  * ATTACK carries no card reference at all, only an attackId, so attack damage
    and cost must come from a precomputed attack table
    (src/card_db/static_table.py::build_attack_table). Resolving the source of
    an ATTACK or RETREAT to the player's own Active Pokémon is a deliberate
    refinement of the spec's literal "(area, index)": those option types have
    no (area, index), and the acting Pokémon is unambiguous.

INVARIANT (spec §5.4): any two options in one decision that differ in engine
semantics must differ in their encoded vector. `action_collision_report()`
checks this on the RAW features -- the pre-encoder quantities below. That is
the right place to check: the encoders are deterministic functions of the raw
features, so distinct raw rows cannot collide after encoding, and identical raw
rows certainly will. Assert 0% before training and keep it as a regression
test: broken features and an undertrained model look identical on a loss curve.

Per-card dynamic features (24 dims, spec §3):

        0       hp / maxHp
        1       maxHp / 380.0
        2       len(energies) / 5.0
        3..14   attached energy histogram per EnergyType, / 5.0
        15      len(tools)
        16      len(preEvolution) / 2.0
        17      appearThisTurn
        18      is_active
        19..23  poisoned, burned, asleep, paralyzed, confused
                -- these live on PlayerState, so they are non-zero ONLY in the
                   active slot, and there is no Frozen flag

The same code path must serve both modes: live inference reads obs["current"]
and obs["select"]["option"] from the engine; training reconstructs identical
tensors from replay records. Any divergence silently poisons training while
metrics keep improving -- keep exactly one implementation. Both paths pass the
same dict shape: {"current": ..., "select": ..., "logs": ...}.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.card_db.static_table import TABLE_ROWS

# ---------------------------------------------------------------------------
# Widths, read off cg.api. Every one of these was wrong in the v1 draft.
# ---------------------------------------------------------------------------
NUM_OPTION_TYPES = 17
NUM_ENERGY_TYPES = 12
NUM_AREA_SLOTS = 13          # AreaType is 1..12; slot 0 encodes "absent"

NUM_FIELD_SLOTS = 12         # self active + 5 bench, opp active + 5 bench
BENCH_MAX = 5
DYNAMIC_DIM = 24
CARD_OUT_DIM = 64
STRUCTURAL_DIM = 64
ACTION_IN_DIM = 192          # 64 structural + 64 source card + 64 target slot
RAW_STATE_DIM = 984          # 12*64 + 64 + 64 + 64 + 24
GLOBAL_SCALAR_DIM = 24

# Normalization divisors (configs/feature_config.yaml).
MAX_HP = 380.0
MAX_ENERGIES = 5.0
MAX_PRE_EVOLUTION = 2.0
MAX_ZONE_INDEX = 48.0
MAX_DECK = 60.0
MAX_HAND = 13.0
MAX_DISCARD = 45.0
MAX_PRIZE = 6.0
MAX_TURN = 32.0
MAX_TURN_ACTIONS = 72.0
MAX_NUMBER = 8.0
MAX_K = 80.0
MAX_DAMAGE_COUNTER = 10.0

# AreaType values we resolve against (cg.api.AreaType).
AREA_DECK, AREA_HAND, AREA_DISCARD = 1, 2, 3
AREA_ACTIVE, AREA_BENCH, AREA_PRIZE = 4, 5, 6

# OptionType values reachable from MAIN (cg.api.OptionType).
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE = 7, 8, 9
OPT_ABILITY, OPT_DISCARD, OPT_RETREAT = 10, 11, 12
OPT_ATTACK, OPT_END = 13, 14

# Structural block layout, summing to exactly 64.
_S_TYPE = 0             # 17
_S_AREA = 17            # 13
_S_INDEX = 30           # 1
_S_INPLAY_AREA = 31     # 13
_S_INPLAY_INDEX = 44    # 1
_S_HAS_ATTACK = 45      # 1
_S_DAMAGE = 46          # 1
_S_COST_SIZE = 47       # 1
_S_COST_HIST = 48       # 12
_S_ENERGY_INDEX = 60    # 1
_S_TOOL_INDEX = 61      # 1
_S_NUMBER = 62          # 1
_S_TARGETS_SELF = 63    # 1


# ---------------------------------------------------------------------------
# Per-card dynamics
# ---------------------------------------------------------------------------

def extract_card_dynamics(pokemon: dict | None,
                          player_state: dict,
                          is_active: bool) -> torch.Tensor:
    """Live state of one Pokémon in play -> (24,). Zeros when the slot is empty.

    Status flags sit on PlayerState, not on the Pokémon, so they describe the
    Active only; writing them into a bench slot would be a lie the model would
    happily learn.
    """
    out = torch.zeros(DYNAMIC_DIM, dtype=torch.float32)
    if not pokemon:
        return out

    max_hp = pokemon.get("maxHp") or 0
    out[0] = (pokemon.get("hp") or 0) / max_hp if max_hp else 0.0
    out[1] = max_hp / MAX_HP
    energies = pokemon.get("energies") or []
    out[2] = len(energies) / MAX_ENERGIES
    for energy in energies:
        idx = int(energy)
        if 0 <= idx < NUM_ENERGY_TYPES:
            out[3 + idx] += 1.0 / MAX_ENERGIES
    out[15] = len(pokemon.get("tools") or [])
    out[16] = len(pokemon.get("preEvolution") or []) / MAX_PRE_EVOLUTION
    out[17] = float(bool(pokemon.get("appearThisTurn")))
    out[18] = float(is_active)
    if is_active:
        for i, flag in enumerate(("poisoned", "burned", "asleep",
                                  "paralyzed", "confused")):
            out[19 + i] = float(bool(player_state.get(flag)))
    return out


# ---------------------------------------------------------------------------
# Board state
# ---------------------------------------------------------------------------

@dataclass
class RawBoard:
    """Deterministic, pre-encoder board features. No learned parameters, so
    this is what training and inference must agree on exactly."""
    slot_card_ids: torch.Tensor      # (12,)   long; 0 = empty slot
    slot_dynamics: torch.Tensor      # (12, 24)
    hand_card_ids: torch.Tensor      # (H,)    long; H may be 0
    self_discard_hist: torch.Tensor  # (1268,)
    opp_discard_hist: torch.Tensor   # (1268,)
    scalars: torch.Tensor            # (24,)


def _sides(current: dict) -> tuple[dict, dict]:
    """(self, opponent) PlayerState dicts, ordered by obs.current.yourIndex."""
    you = int(current.get("yourIndex") or 0)
    players = current.get("players") or [{}, {}]
    return players[you], players[1 - you]


def _active_of(player_state: dict) -> dict | None:
    """PlayerState.active is a list of size 0 or 1, and its element is None when
    the card is facedown."""
    active = player_state.get("active") or []
    return active[0] if active else None


def _discard_histogram(player_state: dict) -> torch.Tensor:
    hist = torch.zeros(TABLE_ROWS, dtype=torch.float32)
    for card in player_state.get("discard") or []:
        cid = int(card.get("id") or 0)
        if 0 < cid < TABLE_ROWS:
            hist[cid] += 1.0 / MAX_DISCARD
    return hist


def extract_global_scalars(obs: dict) -> torch.Tensor:
    """24 global scalars (spec §4.1), self-relative throughout.

    Zone sizes come from counts rather than list lengths wherever the engine
    hides contents: the opponent's hand is None (only handCount is visible) and
    every prize card is facedown, so len(prize) is the count and filtering on
    non-None would yield 0.
    """
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    me, opp = _sides(current)
    you = int(current.get("yourIndex") or 0)

    out = torch.zeros(GLOBAL_SCALAR_DIM, dtype=torch.float32)
    for i, ps in enumerate((me, opp)):
        base = i * 5
        out[base + 0] = (ps.get("deckCount") or 0) / MAX_DECK
        out[base + 1] = (ps.get("handCount") or 0) / MAX_HAND
        out[base + 2] = len(ps.get("discard") or []) / MAX_DISCARD
        out[base + 3] = len(ps.get("prize") or []) / MAX_PRIZE
        out[base + 4] = len(ps.get("bench") or []) / BENCH_MAX

    out[10] = (current.get("turn") or 0) / MAX_TURN
    out[11] = (current.get("turnActionCount") or 0) / MAX_TURN_ACTIONS
    out[12] = float(current.get("firstPlayer") == you)
    out[13] = float(bool(current.get("supporterPlayed")))
    out[14] = float(bool(current.get("stadiumPlayed")))
    out[15] = float(bool(current.get("energyAttached")))
    out[16] = float(bool(current.get("retreated")))
    out[17] = float(bool(current.get("stadium")))
    out[18] = len(current.get("looking") or []) / MAX_HAND
    out[19] = float(_active_of(me) is not None)
    out[20] = float(_active_of(opp) is not None)
    out[21] = (select.get("remainDamageCounter") or 0) / MAX_DAMAGE_COUNTER
    out[22] = (select.get("remainEnergyCost") or 0) / MAX_ENERGIES
    out[23] = len(select.get("option") or []) / MAX_K
    return out


def extract_raw_board(obs: dict) -> RawBoard:
    """Observation -> deterministic board features. No model, no parameters."""
    current = obs.get("current") or {}
    me, opp = _sides(current)

    slot_ids = torch.zeros(NUM_FIELD_SLOTS, dtype=torch.long)
    slot_dyn = torch.zeros(NUM_FIELD_SLOTS, DYNAMIC_DIM, dtype=torch.float32)
    for side, ps in enumerate((me, opp)):
        base = side * (1 + BENCH_MAX)
        active = _active_of(ps)
        if active:
            slot_ids[base] = int(active.get("id") or 0)
            slot_dyn[base] = extract_card_dynamics(active, ps, is_active=True)
        for i, mon in enumerate((ps.get("bench") or [])[:BENCH_MAX]):
            if not mon:
                continue
            slot_ids[base + 1 + i] = int(mon.get("id") or 0)
            slot_dyn[base + 1 + i] = extract_card_dynamics(mon, ps, is_active=False)

    hand = me.get("hand") or []          # None for the opponent, never for self
    hand_ids = torch.tensor([int(c.get("id") or 0) for c in hand], dtype=torch.long)

    return RawBoard(
        slot_card_ids=slot_ids,
        slot_dynamics=slot_dyn,
        hand_card_ids=hand_ids,
        self_discard_hist=_discard_histogram(me),
        opp_discard_hist=_discard_histogram(opp),
        scalars=extract_global_scalars(obs),
    )


@dataclass
class BoardBatch:
    """A batch of RawBoard, padded and stacked. B = 1 at inference."""
    slot_card_ids: torch.Tensor      # (B, 12)     long
    slot_dynamics: torch.Tensor      # (B, 12, 24)
    hand_card_ids: torch.Tensor      # (B, H)      long, 0-padded
    hand_mask: torch.Tensor          # (B, H)      bool
    self_discard_hist: torch.Tensor  # (B, 1268)
    opp_discard_hist: torch.Tensor   # (B, 1268)
    scalars: torch.Tensor            # (B, 24)

    def to(self, device) -> "BoardBatch":
        return BoardBatch(*(t.to(device) for t in (
            self.slot_card_ids, self.slot_dynamics, self.hand_card_ids,
            self.hand_mask, self.self_discard_hist, self.opp_discard_hist,
            self.scalars)))


def collate_boards(raws: list[RawBoard]) -> BoardBatch:
    """Stack RawBoards, right-padding hands to the longest in the batch."""
    hand_len = max((r.hand_card_ids.numel() for r in raws), default=0) or 1
    hand_ids = torch.zeros(len(raws), hand_len, dtype=torch.long)
    hand_mask = torch.zeros(len(raws), hand_len, dtype=torch.bool)
    for i, r in enumerate(raws):
        n = r.hand_card_ids.numel()
        if n:
            hand_ids[i, :n] = r.hand_card_ids
            hand_mask[i, :n] = True
    return BoardBatch(
        slot_card_ids=torch.stack([r.slot_card_ids for r in raws]),
        slot_dynamics=torch.stack([r.slot_dynamics for r in raws]),
        hand_card_ids=hand_ids,
        hand_mask=hand_mask,
        self_discard_hist=torch.stack([r.self_discard_hist for r in raws]),
        opp_discard_hist=torch.stack([r.opp_discard_hist for r in raws]),
        scalars=torch.stack([r.scalars for r in raws]),
    )


def assemble_board_state(batch: BoardBatch,
                         card_encoder,
                         self_discard_proj,
                         opp_discard_proj) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (board (B, 984), slot_vecs (B, 12, 64)).

    Batch-native so that ONE implementation serves both modes (invariant 1):
    training passes B = N, live inference passes B = 1. A separate single-sample
    assembly path would be a second implementation of exactly the thing that
    must not diverge.

    `slot_vecs` is returned rather than discarded because the action matrix's
    target block is a *gather* into it (spec §5.3). Recomputing per option would
    cost k card-encoder passes per turn against a 10 ms budget.

    The two discard projections take separate parameters on purpose: what my
    discard pile says about my deck is not what the opponent's says about
    theirs. The spec's Linear(1268, 64) -> ReLU lives in the caller's module so
    this stays a featurizer with the learned pieces injected.
    """
    b = batch.slot_card_ids.shape[0]
    slot_vecs = card_encoder(batch.slot_card_ids, batch.slot_dynamics)  # (B,12,64)

    # Mean-pool the hand over real cards only; padding must not drag the mean.
    hand_dyn = torch.zeros(*batch.hand_card_ids.shape, DYNAMIC_DIM,
                           device=batch.hand_card_ids.device)
    hand_vecs = card_encoder(batch.hand_card_ids, hand_dyn)             # (B,H,64)
    mask = batch.hand_mask.unsqueeze(-1).float()
    hand_pool = (hand_vecs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    board = torch.cat([
        slot_vecs.reshape(b, -1),                            # 768
        hand_pool,                                           #  64
        self_discard_proj(batch.self_discard_hist),          #  64
        opp_discard_proj(batch.opp_discard_hist),            #  64
        batch.scalars,                                       #  24
    ], dim=-1)
    if board.shape[-1] != RAW_STATE_DIM:
        raise ValueError(f"board state is {board.shape[-1]}, expected {RAW_STATE_DIM}")
    return board, slot_vecs


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _card_in_area(player_state: dict, area: int | None, index: int | None
                  ) -> tuple[int, dict | None, bool]:
    """(card_id, pokemon_or_None, is_active) for a positional (area, index).

    Returns the Pokémon object only for cards in play, since that is the only
    place dynamic features exist. Unknown or hidden zones resolve to id 0, which
    the static table maps to its zero row.
    """
    if area is None or index is None:
        return 0, None, False
    idx = int(index)

    if area == AREA_ACTIVE:
        mon = _active_of(player_state)
        return (int(mon.get("id") or 0) if mon else 0), mon, True
    if area == AREA_BENCH:
        bench = player_state.get("bench") or []
        mon = bench[idx] if 0 <= idx < len(bench) else None
        return (int(mon.get("id") or 0) if mon else 0), mon, False
    if area == AREA_HAND:
        hand = player_state.get("hand") or []
        card = hand[idx] if 0 <= idx < len(hand) else None
        return (int(card.get("id") or 0) if card else 0), None, False
    if area == AREA_DISCARD:
        discard = player_state.get("discard") or []
        card = discard[idx] if 0 <= idx < len(discard) else None
        return (int(card.get("id") or 0) if card else 0), None, False
    # DECK and PRIZE are facedown; STADIUM/ENERGY/TOOL/PRE_EVOLUTION are reached
    # only from non-MAIN contexts, which the fallback agent handles.
    return 0, None, False


def resolve_source_card(option: dict, current: dict) -> tuple[int, torch.Tensor]:
    """The card an option acts WITH -> (card_id, dynamics (24,)).

    MAIN options never carry `playerIndex`, and every MAIN option acts with one
    of our own cards, so this resolves against the self side.
    """
    me, _opp = _sides(current)
    otype = int(option.get("type", -1))

    if otype == OPT_PLAY:
        # PLAY carries no `area`; its index is into the hand by definition.
        cid, _mon, _act = _card_in_area(me, AREA_HAND, option.get("index"))
        return cid, torch.zeros(DYNAMIC_DIM)

    if otype in (OPT_ATTACK, OPT_RETREAT):
        # No (area, index) on these; the acting Pokémon is the Active.
        mon = _active_of(me)
        if not mon:
            return 0, torch.zeros(DYNAMIC_DIM)
        return int(mon.get("id") or 0), extract_card_dynamics(mon, me, is_active=True)

    if otype == OPT_END:
        return 0, torch.zeros(DYNAMIC_DIM)

    cid, mon, is_active = _card_in_area(me, option.get("area"), option.get("index"))
    dyn = (extract_card_dynamics(mon, me, is_active) if mon
           else torch.zeros(DYNAMIC_DIM))
    return cid, dyn


def target_slot_index(option: dict) -> int:
    """Field slot an option targets, or -1 for none.

    MAIN targets are always our own side, so this maps into slots 0..5.
    """
    area = option.get("inPlayArea")
    if area is None:
        return -1
    if area == AREA_ACTIVE:
        return 0
    if area == AREA_BENCH:
        idx = int(option.get("inPlayIndex") or 0)
        return 1 + idx if 0 <= idx < BENCH_MAX else -1
    return -1


def encode_option_structural(option: dict,
                             obs: dict,
                             attack_table: torch.Tensor) -> torch.Tensor:
    """One option's structural fields -> (64,)."""
    current = obs.get("current") or {}
    out = torch.zeros(STRUCTURAL_DIM, dtype=torch.float32)

    otype = int(option.get("type", -1))
    if 0 <= otype < NUM_OPTION_TYPES:
        out[_S_TYPE + otype] = 1.0

    area = option.get("area")
    if area is not None and 0 < int(area) < NUM_AREA_SLOTS:
        out[_S_AREA + int(area)] = 1.0
    if option.get("index") is not None:
        out[_S_INDEX] = int(option["index"]) / MAX_ZONE_INDEX

    in_play_area = option.get("inPlayArea")
    if in_play_area is not None and 0 < int(in_play_area) < NUM_AREA_SLOTS:
        out[_S_INPLAY_AREA + int(in_play_area)] = 1.0
    if option.get("inPlayIndex") is not None:
        out[_S_INPLAY_INDEX] = int(option["inPlayIndex"]) / BENCH_MAX

    # ATTACK options carry only an attackId, so damage and cost come from the
    # precomputed attack table -- two attacks on the same Pokémon are otherwise
    # indistinguishable.
    attack_id = option.get("attackId")
    if attack_id is not None and 0 < int(attack_id) < attack_table.shape[0]:
        row = attack_table[int(attack_id)]
        out[_S_HAS_ATTACK] = 1.0
        out[_S_DAMAGE] = row[0]
        out[_S_COST_SIZE] = row[1]
        out[_S_COST_HIST:_S_COST_HIST + NUM_ENERGY_TYPES] = row[2:]

    if option.get("energyIndex") is not None:
        out[_S_ENERGY_INDEX] = int(option["energyIndex"]) / MAX_ENERGIES
    if option.get("toolIndex") is not None:
        out[_S_TOOL_INDEX] = float(option["toolIndex"])
    if option.get("number") is not None:
        out[_S_NUMBER] = int(option["number"]) / MAX_NUMBER

    # MAIN options omit playerIndex and always act on our own side.
    player_index = option.get("playerIndex")
    out[_S_TARGETS_SELF] = float(player_index is None
                                 or int(player_index) == int(current.get("yourIndex") or 0))
    return out


@dataclass
class RawActions:
    """Deterministic, pre-encoder action features for the k options."""
    structural: torch.Tensor      # (k, 64)
    source_card_ids: torch.Tensor  # (k,)  long
    source_dynamics: torch.Tensor  # (k, 24)
    target_slots: torch.Tensor     # (k,)  long; -1 = no target


def extract_raw_actions(obs: dict, attack_table: torch.Tensor) -> RawActions:
    current = obs.get("current") or {}
    options = (obs.get("select") or {}).get("option") or []
    k = len(options)

    structural = torch.zeros(k, STRUCTURAL_DIM, dtype=torch.float32)
    source_ids = torch.zeros(k, dtype=torch.long)
    source_dyn = torch.zeros(k, DYNAMIC_DIM, dtype=torch.float32)
    targets = torch.full((k,), -1, dtype=torch.long)

    for i, option in enumerate(options):
        structural[i] = encode_option_structural(option, obs, attack_table)
        cid, dyn = resolve_source_card(option, current)
        source_ids[i] = cid
        source_dyn[i] = dyn
        targets[i] = target_slot_index(option)

    return RawActions(structural, source_ids, source_dyn, targets)


@dataclass
class ActionBatch:
    """A batch of RawActions, padded to a common k. B = 1 at inference."""
    structural: torch.Tensor       # (B, k, 64)
    source_card_ids: torch.Tensor  # (B, k)     long
    source_dynamics: torch.Tensor  # (B, k, 24)
    target_slots: torch.Tensor     # (B, k)     long, -1 = none
    mask: torch.Tensor             # (B, k)     bool, True = real candidate

    def to(self, device) -> "ActionBatch":
        return ActionBatch(*(t.to(device) for t in (
            self.structural, self.source_card_ids, self.source_dynamics,
            self.target_slots, self.mask)))


def collate_actions(raws: list[RawActions], k_max: int | None = None) -> ActionBatch:
    """Stack RawActions, right-padding the candidate dimension.

    Pads, never truncates: dropping a candidate can drop the expert's chosen
    move and leave the training target undefined (configs/feature_config.yaml).
    """
    k = max((r.structural.shape[0] for r in raws), default=0) or 1
    if k_max is not None and k > k_max:
        raise ValueError(f"k={k} exceeds k_max={k_max}; such decisions belong "
                         "to the fallback, filter them out before collating")
    n = len(raws)
    structural = torch.zeros(n, k, STRUCTURAL_DIM)
    source_ids = torch.zeros(n, k, dtype=torch.long)
    source_dyn = torch.zeros(n, k, DYNAMIC_DIM)
    targets = torch.full((n, k), -1, dtype=torch.long)
    mask = torch.zeros(n, k, dtype=torch.bool)
    for i, r in enumerate(raws):
        j = r.structural.shape[0]
        structural[i, :j] = r.structural
        source_ids[i, :j] = r.source_card_ids
        source_dyn[i, :j] = r.source_dynamics
        targets[i, :j] = r.target_slots
        mask[i, :j] = True
    return ActionBatch(structural, source_ids, source_dyn, targets, mask)


def assemble_action_matrix(batch: ActionBatch,
                           slot_vecs: torch.Tensor,
                           card_encoder) -> torch.Tensor:
    """-> (B, k, 192): [structural (64) | source card (64) | target slot (64)].

    The target block gathers from `slot_vecs`, which assemble_board_state
    already computed. A slot of -1 (no target) gathers a zero vector.
    """
    source_vecs = card_encoder(batch.source_card_ids, batch.source_dynamics)

    # Prepend a zero slot so target -1 indexes it, avoiding a per-option branch.
    b, _, d = slot_vecs.shape
    padded = torch.cat([torch.zeros(b, 1, d, device=slot_vecs.device),
                        slot_vecs], dim=1)                       # (B, 13, 64)
    idx = (batch.target_slots + 1).unsqueeze(-1).expand(-1, -1, d)
    target_vecs = padded.gather(1, idx)                          # (B, k, 64)

    matrix = torch.cat([batch.structural, source_vecs, target_vecs], dim=-1)
    if matrix.shape[-1] != ACTION_IN_DIM:
        raise ValueError(f"action row is {matrix.shape[-1]}, expected {ACTION_IN_DIM}")
    return matrix


# ---------------------------------------------------------------------------
# Spec §5.4: the encoding-collision invariant
# ---------------------------------------------------------------------------

def action_collision_report(obs: dict, attack_table: torch.Tensor) -> dict:
    """Do any two options in this decision encode identically?

    Checked on RAW features. The encoders are deterministic functions of them,
    so identical raw rows are guaranteed to collide after encoding, and distinct
    raw rows cannot be made to collide by a function. Any collision means the
    training target is undefined: two candidates look the same and only one is
    labelled correct.
    """
    raw = extract_raw_actions(obs, attack_table)
    k = raw.structural.shape[0]
    if k < 2:
        return {"k": k, "collisions": 0, "colliding_options": 0, "groups": []}

    keys = torch.cat([
        raw.structural,
        raw.source_card_ids.unsqueeze(1).float(),
        raw.source_dynamics,
        raw.target_slots.unsqueeze(1).float(),
    ], dim=1)

    groups: dict[bytes, list[int]] = {}
    for i in range(k):
        groups.setdefault(keys[i].numpy().tobytes(), []).append(i)
    dupes = [idxs for idxs in groups.values() if len(idxs) > 1]
    return {
        "k": k,
        "collisions": sum(len(g) - 1 for g in dupes),
        "colliding_options": sum(len(g) for g in dupes),
        "groups": dupes,
    }
