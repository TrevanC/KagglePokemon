"""Parse data/raw/EN_Card_Data.csv into structured per-card records.

The CSV has one row per card ID (1267 of them, IDs 1..1267) with these columns:

    Card ID, Card Name, Expansion, Collection No.,
    Stage (Pokémon)/Type (Energy and Trainer), Rule, Category, Previous stage,
    HP, Type, Weakness, Resistance (Type), Retreat,
    Move Name, Cost, Damage, Effect Explanation

Quirks to handle:
  - "n/a" and empty strings are both used for missing values.
  - Element types are bracketed symbols in the Type column, e.g. "{G}", "{R}".
  - Damage may carry suffixes (e.g. "50+", "20x") that need stripping.
  - A card's attack fields describe its *primary* attack only; the engine's own
    ATTACKS table (pokemon-tcg-ai-battle/arena/cards.py) has the full set.

TODO: implement
    parse_card_csv(csv_path) -> list[CardRecord]
        One record per card ID, with cleaned/typed fields.
    load_card_records(csv_path) -> dict[int, CardRecord]
        Same, keyed by card ID for lookup.

Downstream consumer: src/card_db/static_table.py.
"""
