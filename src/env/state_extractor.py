"""Convert a raw observation dict into model input tensors.

Two outputs per decision point (technical specification §2.3, §2.4):

  1. Board state vector, 912 dimensions:

        12 field slots x 64      = 768   self active + 5 bench, opp active + 5 bench;
                                         unoccupied slots are zero vectors
        hand mean-pool           =  64   mean over hand card vectors, zeros if empty
        discard projection       =  64   Linear(1267 -> 64) over a count histogram
                                         normalized by 60.0 (self & opponent)
        global scalars           =  16   [self_prizes/6, opp_prizes/6, self_deck/60,
                                          opp_deck/60, opp_hand/20, energy_attached_flag,
                                          supporter_played_flag, turn/100, ...zero-padded]
        ---------------------------------
        total                    = 912

  2. Candidate action matrix, (k, 16) for the k legal options (k in [1, 50]):

        0..9    option type one-hot   PLAY, ATTACH, EVOLVE, ABILITY, DISCARD,
                                      RETREAT, ATTACK, END, ...
        10      target card ID        card_id / 1267.0
        11      target slot index     slot_idx / 12.0
        12..15  padded parameter channels

Per-card dynamic features (8 dims), combined with the static table row and the
learned ID embedding in src/models/card_encoder.py:

        0       current_hp / max_hp   (0.0 if not on the field)
        1       min(1.0, attached_energy / 10.0)
        2..7    status flags          [poisoned, burned, asleep, paralyzed,
                                       confused, frozen]

The same code path must serve both modes (§2.5): live inference reads
obs["current"] and obs["select"]["option"] from the engine, while training
reconstructs the identical tensors from replay log entries. Any divergence
between the two silently poisons training — keep one implementation.

Use cg.api.to_observation_class(obs_dict) (pokemon-tcg-ai-battle/arena/cg/api.py)
for the typed observation schema and OptionType enum.

TODO: implement
    extract_board_state(obs, static_table) -> Tensor (912,)
    extract_action_matrix(obs) -> Tensor (k, 16)
    extract_card_dynamics(card_obs) -> Tensor (8,)
"""
