"""Step 1: precompute the static card feature lookup table.

    python scripts/build_static_features.py

Input:   data/raw/EN_Card_Data.csv          (1267 unique card IDs)
Output:  data/processed/static_feature_table.pt   tensor of shape (1267, 32)
Config:  configs/feature_config.yaml

Runs once, offline. Design rule 2: the agent must never parse the CSV during a
live turn — it reads this tensor instead.

Uses src/card_db/parser.py to read the CSV and src/card_db/static_table.py to
build and save the tensor. See the feature index map in static_table.py.
"""

if __name__ == "__main__":
    raise NotImplementedError("scripts/build_static_features.py is a stub")
