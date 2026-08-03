"""Step 1: precompute the static card feature lookup table.

    python scripts/build_static_features.py

Input:   the engine's card DB — cg.api.all_card_data() (1267 cards) and
         all_attack() (1556 attacks), NOT the CSV (spec §2)
Output:  data/processed/static_feature_table.pt   tensor of shape (1268, 72)
Config:  configs/feature_config.yaml

Row index IS the card ID; row 0 is zeroed so IDs 1..1267 index directly.

Runs once, offline. Design rule 2: the agent must never build this table during
a live turn — it loads the tensor instead.

Uses src/card_db/static_table.py; see the feature index map there. The CSV
reader in src/card_db/parser.py is reference-only and not on this path.
"""

if __name__ == "__main__":
    raise NotImplementedError("scripts/build_static_features.py is a stub")
