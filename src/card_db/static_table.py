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

TODO: implement
    build_static_table(rows=1268, feature_dim=72) -> torch.Tensor
    save_static_table(table, path) / load_static_table(path) -> torch.Tensor
    class StaticCardFeatureTable:
        get_features(card_ids: Tensor) -> Tensor  # (..., 72) gather

Consumers: src/models/card_encoder.py, src/models/standalone_agent.py.
"""
