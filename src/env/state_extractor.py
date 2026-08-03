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

  2. Candidate action matrix, (k, 192) for the k legal options, k <= 80
     (spec §5). Three concatenated blocks:

        structural   (64)  type one-hot(17) | area one-hot(13) | index/48 |
                           inPlayArea one-hot(13) | inPlayIndex/5 |
                           has_attack | damage/350 | cost size/5 |
                           cost histogram(12) | energyIndex/5 | toolIndex |
                           number/8 | targets_self
        source card  (64)  resolve (area, index) against obs.current to the
                           card the option acts WITH, then embed it with the
                           SAME HybridCardEncoder used for the board, dynamic
                           features zeroed for cards not in play
        target slot  (64)  gather the already-computed field-slot vector for
                           (inPlayArea, inPlayIndex) -- a lookup, not a
                           recomputation, so it is free at inference

     >>> The source-card and target-slot blocks are not optional polish. The
     >>> engine never populates `cardId` on MAIN options; it identifies targets
     >>> positionally. Encoding only (type, cardId, slot) -- the v1 design --
     >>> made 50.2% of options byte-identical to a sibling and 60.5% of MAIN
     >>> decisions unlearnable. See spec §0.

     INVARIANT (spec §5.4): any two options in one decision that differ in
     engine semantics must differ in their encoded vector. Assert a 0%
     collision rate over sampled real decisions before training, and keep it
     as a regression test -- broken features and an undertrained model look
     identical on a loss curve.

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
metrics keep improving -- keep exactly one implementation.

Use cg.api.to_observation_class(obs_dict) for the typed schema, and the enums
(OptionType=17, AreaType=12, EnergyType=12) rather than hardcoded widths.

TODO: implement
    extract_board_state(obs, static_table, card_encoder) -> (Tensor (984,), slot_vecs (12,64))
    extract_action_matrix(obs, slot_vecs, card_encoder) -> Tensor (k, 192)
    extract_card_dynamics(pokemon, player_state, is_active) -> Tensor (24,)
    encode_option_structural(option, obs) -> Tensor (64,)
"""
