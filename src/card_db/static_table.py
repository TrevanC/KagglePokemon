"""Build the StaticCardFeatureTable: a (1268, 72) precomputed lookup tensor.

Source is the ENGINE, not the CSV (spec §2). cg.api.all_card_data() returns
typed CardData -- cardId, name, cardType, retreatCost, hp, weakness, resistance,
energyType, basic/stage1/stage2, ex/megaEx/tera/aceSpec, evolvesFrom, skills,
attacks -- and all_attack() returns Attack (attackId, name, text, damage,
energies) for all 1556 attacks. That sidesteps the CSV's "{G}" symbols, its
"50+"/"20x" damage suffixes, and its 2022-rows-for-1267-cards layout, and it
stays consistent with the engine actually judging the match.

Row index IS the card ID. Row 0 is zeroed so IDs 1..1267 index directly --
this removes the off-by-one convention that would otherwise have to be kept in
sync across this file, state_extractor.py, and standalone_agent.py.

Feature index map (spec §2.1):

     0..6    cardType one-hot     POKEMON, ITEM, TOOL, SUPPORTER, STADIUM,
                                  BASIC_ENERGY, SPECIAL_ENERGY
     7..9    stage one-hot        basic, stage1, stage2
    10..13   rule flags           ex, megaEx, tera, aceSpec
    14       base HP              hp / 380.0        (measured max, not 340)
    15       retreat cost         retreatCost / 4.0
    16..27   energyType one-hot   12 EnergyType values (not 10)
    28..39   weakness one-hot     12; all-zero when None
    40..51   resistance one-hot   12; all-zero when None
    52       attack count         len(attacks) / 2.0
    53..54   attack 0             damage / 350.0, len(energies) / 5.0
    55..56   attack 1             same; zeros when absent
    57..68   attack cost hist     per EnergyType across all attacks, / 5.0
    69       has ability          len(skills) > 0
    70       evolves from         evolvesFrom is not None
    71       reserved             0.0

Written once by scripts/build_static_features.py to
data/processed/static_feature_table.pt. Design rule 2: never call
all_card_data() or read the CSV inside the live turn loop.

Consumers: src/models/card_encoder.py, src/models/standalone_agent.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

# Table geometry. Row index is the card ID, so the table has one more row than
# there are cards; row 0 stays zeroed and doubles as the "no card here" vector,
# which is what lets empty field slots and absent option sources be encoded as
# card id 0 rather than by a separate branch.
NUM_CARDS = 1267
TABLE_ROWS = 1268
FEATURE_DIM = 72

# Enum widths, read off cg.api rather than assumed -- every one of these was
# wrong in the v1 spec draft (spec §0).
NUM_CARD_TYPES = 7
NUM_ENERGY_TYPES = 12

# Normalization divisors. All four are the measured maxima over the live card
# DB (see scripts/build_static_features.py --verify), so features land in
# [0, 1] rather than merely near it.
MAX_HP = 380.0
MAX_DAMAGE = 350.0
MAX_RETREAT_COST = 4.0
MAX_ENERGIES_ATTACK = 5.0
MAX_ATTACKS_PER_CARD = 2.0

# Field offsets, named so the map above and the code cannot drift apart.
_CARD_TYPE = 0          # 7
_STAGE = 7              # 3
_RULE_FLAGS = 10        # 4
_HP = 14
_RETREAT = 15
_ENERGY_TYPE = 16       # 12
_WEAKNESS = 28          # 12
_RESISTANCE = 40        # 12
_ATTACK_COUNT = 52
_ATTACK_0 = 53          # 2
_ATTACK_1 = 55          # 2
_COST_HIST = 57         # 12
_HAS_ABILITY = 69
_EVOLVES_FROM = 70
_RESERVED = 71

DEFAULT_PATH = "data/processed/static_feature_table.pt"


def build_static_table(rows: int = TABLE_ROWS,
                       feature_dim: int = FEATURE_DIM) -> torch.Tensor:
    """Build the (1268, 72) static table from the engine's card DB.

    Imports cg.api lazily so that merely importing this module -- which
    src/models/ does -- never requires the native engine to be loadable.
    """
    from cg.api import all_attack, all_card_data  # noqa: PLC0415 - native dep

    cards = all_card_data()
    attacks = {a.attackId: a for a in all_attack()}

    table = torch.zeros(rows, feature_dim, dtype=torch.float32)

    for card in cards:
        row = table[card.cardId]

        # The dataclass fields come back as plain ints, not IntEnum members
        # (to_dataclass does not coerce), so index with int() directly.
        row[_CARD_TYPE + int(card.cardType)] = 1.0

        for i, is_stage in enumerate((card.basic, card.stage1, card.stage2)):
            row[_STAGE + i] = float(is_stage)
        for i, flag in enumerate((card.ex, card.megaEx, card.tera, card.aceSpec)):
            row[_RULE_FLAGS + i] = float(flag)

        row[_HP] = card.hp / MAX_HP
        row[_RETREAT] = card.retreatCost / MAX_RETREAT_COST

        # energyType is always populated, but only *means* anything for Pokémon
        # and Energy cards; on an Item or Supporter it is a default COLORLESS.
        # Kept raw and unbranched per the spec -- the cardType one-hot above is
        # what disambiguates a Colorless Pokémon from a Trainer card.
        row[_ENERGY_TYPE + int(card.energyType)] = 1.0
        if card.weakness is not None:
            row[_WEAKNESS + int(card.weakness)] = 1.0
        if card.resistance is not None:
            row[_RESISTANCE + int(card.resistance)] = 1.0

        card_attacks = [attacks[aid] for aid in card.attacks if aid in attacks]
        row[_ATTACK_COUNT] = len(card_attacks) / MAX_ATTACKS_PER_CARD
        for i, slot in enumerate((_ATTACK_0, _ATTACK_1)):
            if i < len(card_attacks):
                atk = card_attacks[i]
                row[slot] = atk.damage / MAX_DAMAGE
                row[slot + 1] = len(atk.energies) / MAX_ENERGIES_ATTACK
        for atk in card_attacks:
            for energy in atk.energies:
                row[_COST_HIST + int(energy)] += 1.0 / MAX_ENERGIES_ATTACK

        row[_HAS_ABILITY] = float(len(card.skills) > 0)
        row[_EVOLVES_FROM] = float(card.evolvesFrom is not None)
        row[_RESERVED] = 0.0

    return table


# ---------------------------------------------------------------------------
# Attack table
# ---------------------------------------------------------------------------
# MAIN's ATTACK options carry *only* an attackId -- no cardId, no area, no
# damage (measured over 2307 real ATTACK options; see the field-availability
# table in state_extractor.py). Two attacks on the same Pokémon are therefore
# distinguishable only through what the attackId resolves to, so the action
# encoder needs damage and energy cost keyed by attack id. Design rule 2 forbids
# calling all_attack() in the turn loop, so it is precomputed here exactly like
# the card table.
#
#     0       damage / 350.0
#     1       len(energies) / 5.0
#     2..13   energy cost histogram per EnergyType, / 5.0
ATTACK_FEATURE_DIM = 14
_ATK_DAMAGE = 0
_ATK_COST_SIZE = 1
_ATK_COST_HIST = 2

DEFAULT_ATTACK_PATH = "data/processed/attack_feature_table.pt"


def build_attack_table(feature_dim: int = ATTACK_FEATURE_DIM) -> torch.Tensor:
    """Build the (max_attack_id + 1, 14) attack table. Row index IS the attack
    id, and row 0 is the zero row for "no attack", mirroring the card table."""
    from cg.api import all_attack  # noqa: PLC0415 - native dep

    attacks = all_attack()
    rows = max(a.attackId for a in attacks) + 1
    table = torch.zeros(rows, feature_dim, dtype=torch.float32)
    for atk in attacks:
        row = table[atk.attackId]
        row[_ATK_DAMAGE] = atk.damage / MAX_DAMAGE
        row[_ATK_COST_SIZE] = len(atk.energies) / MAX_ENERGIES_ATTACK
        for energy in atk.energies:
            row[_ATK_COST_HIST + int(energy)] += 1.0 / MAX_ENERGIES_ATTACK
    return table


def save_attack_table(table: torch.Tensor,
                      path: str | Path = DEFAULT_ATTACK_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, path)
    return path


def load_attack_table(path: str | Path = DEFAULT_ATTACK_PATH) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu")


def save_static_table(table: torch.Tensor, path: str | Path = DEFAULT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, path)
    return path


def load_static_table(path: str | Path = DEFAULT_PATH) -> torch.Tensor:
    """Load the precomputed table. No engine import, no CSV -- this is the only
    card-data path allowed inside the live turn loop."""
    table = torch.load(Path(path), map_location="cpu")
    if table.shape != (TABLE_ROWS, FEATURE_DIM):
        raise ValueError(
            f"{path}: expected {(TABLE_ROWS, FEATURE_DIM)}, got {tuple(table.shape)}. "
            "Rebuild with scripts/build_static_features.py."
        )
    return table


class StaticCardFeatureTable:
    """Gather static features for card IDs.

    Card id 0 means "no card" (empty slot, absent option source) and returns the
    zero row, so callers never need an is-present branch.
    """

    def __init__(self, table: torch.Tensor):
        self.table = table

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "StaticCardFeatureTable":
        return cls(load_static_table(path))

    @classmethod
    def build(cls) -> "StaticCardFeatureTable":
        return cls(build_static_table())

    def get_features(self, card_ids: torch.Tensor) -> torch.Tensor:
        """(...,) card ids -> (..., 72) features. Shape-preserving gather."""
        return self.table[card_ids.long()]

    def __len__(self) -> int:
        return self.table.shape[0]


# (name, start, width) for each feature block, in index order. Used by
# verify_static_table to check coverage per block rather than per column.
FEATURE_BLOCKS = (
    ("cardType", _CARD_TYPE, NUM_CARD_TYPES),
    ("stage", _STAGE, 3),
    ("rule_flags", _RULE_FLAGS, 4),
    ("hp", _HP, 1),
    ("retreat", _RETREAT, 1),
    ("energyType", _ENERGY_TYPE, NUM_ENERGY_TYPES),
    ("weakness", _WEAKNESS, NUM_ENERGY_TYPES),
    ("resistance", _RESISTANCE, NUM_ENERGY_TYPES),
    ("attack_count", _ATTACK_COUNT, 1),
    ("attack_0", _ATTACK_0, 2),
    ("attack_1", _ATTACK_1, 2),
    ("cost_hist", _COST_HIST, NUM_ENERGY_TYPES),
    ("has_ability", _HAS_ABILITY, 1),
    ("evolves_from", _EVOLVES_FROM, 1),
    ("reserved", _RESERVED, 1),
)

# The cost histogram sums energies across a card's attacks while dividing by the
# per-*attack* maximum (5), so its ceiling is 2 attacks x 5 energies / 5 = 2.0,
# not 1.0. Measured worst case is Mega Zygarde ex (id 1056), 8 Fighting energies
# across two attacks -> 1.6. Every other block is genuinely bounded by 1.0.
MAX_FEATURE_VALUE = 2.0


def verify_static_table(table: torch.Tensor) -> dict:
    """Structural checks on a built table.

    These are cheap and catch the failure that matters: a table that is the
    right *shape* but silently mostly-empty, e.g. because an engine update
    renamed a CardData field. A downstream model trains happily on zeros.

    Coverage is checked per *block*, not per column. Individual columns are
    legitimately all-zero because the enums are wider than the card pool uses:
    no card is weak to Colorless/Dragon/Rainbow/Team Rocket, resistance only
    ever takes Grass or Fighting, and no attack costs Dragon/Rainbow/Team
    Rocket energy. An entirely dead block, on the other hand, means that block
    is not being written.
    """
    populated = int((table.abs().sum(dim=1) > 0).sum())
    blocks = {
        name: int((table[:, start:start + width].abs().sum(dim=0) > 0).sum())
        for name, start, width in FEATURE_BLOCKS
    }
    # "reserved" is documented as permanently zero; everything else must write.
    dead_blocks = [n for n, used in blocks.items() if used == 0 and n != "reserved"]
    return {
        "shape": tuple(table.shape),
        "row0_zeroed": bool(table[0].abs().sum() == 0),
        "populated_rows": populated,
        "expected_populated": NUM_CARDS,
        "all_populated": populated == NUM_CARDS,
        "min": float(table.min()),
        "max": float(table.max()),
        "in_range": bool(table.min() >= 0.0 and table.max() <= MAX_FEATURE_VALUE),
        "block_columns_used": blocks,
        "dead_blocks": dead_blocks,
        "dead_columns": [i for i in range(table.shape[1])
                         if float(table[:, i].abs().sum()) == 0.0],
    }
