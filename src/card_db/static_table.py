"""Build the StaticCardFeatureTable: a (1267, 32) precomputed lookup tensor.

Computed once offline so the agent never parses strings or CSV rows during a
live turn. Written to data/processed/static_feature_table.pt by
scripts/build_static_features.py, and loaded at agent init.

Feature index map (technical specification §2.1):

    0..2    Category one-hot        [is_pokemon, is_trainer, is_energy]
    3..5    Stage one-hot           [is_basic, is_stage1, is_stage2]
    6       Base HP                 hp / 340.0            -> [0, 1]
    7..16   Element type one-hot    10 Pokémon types (Grass, Fire, Water, ...)
    17      Primary attack damage   damage / 300.0        -> [0, 1]
    18      Retreat cost            retreat / 4.0         -> [0, 1]
    19..31  Reserved                energy costs, trainer effect flags (zero-padded)

Row index == card ID. IDs are 1-based in the CSV; decide on and document one
indexing convention here (allocate 1268 rows, or subtract 1) and make
src/env/state_extractor.py use the same one.

TODO: implement
    build_static_table(records, num_cards=1267, feature_dim=32) -> torch.Tensor
    save_static_table(table, path) / load_static_table(path) -> torch.Tensor
    class StaticCardFeatureTable:
        get_features(card_ids: Tensor) -> Tensor  # (..., 32) gather

Consumers: src/models/card_encoder.py, src/models/standalone_agent.py.
"""
