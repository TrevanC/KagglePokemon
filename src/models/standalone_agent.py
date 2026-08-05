"""CPU inference agent: the trained policy plus its non-MAIN fallback.

Vendored into submission/build/ by scripts/package_submission.py.

ON RULE 4 ("fully self-contained, no imports outside the deployment folder")
---------------------------------------------------------------------------
The original plan was for this file to inline copies of the extractor and
encoders. That plan is wrong, and quietly so: a hand-maintained copy of
src/env/state_extractor.py is EXACTLY the train/deploy divergence invariant 1
exists to prevent, and the failure is invisible -- training metrics keep
improving while the deployed agent featurizes differently.

Rule 4 is a statement about the deployment folder's import graph, not a
mandate to duplicate code by hand. Packaging therefore COPIES the needed src/
modules into the deployment folder, so the bundle is self-contained *and*
byte-identical to what was trained against. One source of truth, mechanically
vendored.

Other deployment rules this satisfies:
  - Rule 1: < 10 ms/turn on CPU. Model, static table and attack table load once
    at construction; the turn loop allocates only the per-decision tensors.
  - Rule 2: reads static_feature_table.pt, never EN_Card_Data.csv.
  - torch.no_grad() around every forward, model.eval(), one thread (the Kaggle
    box has 2 vCPUs and thread contention costs more than it buys at this size).

The returned list must have length in [minCount, maxCount] with no duplicate
indices, and must be valid even when obs["select"] is None -- the initial
selection, where the agent returns its 60-card deck instead of option indices.
An illegal selection forfeits the match, so every return path goes through
clamp_selection().
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.card_db.static_table import load_attack_table, load_static_table
from src.env.state_extractor import (
    collate_actions,
    collate_boards,
    extract_raw_actions,
    extract_raw_board,
)
from src.evaluation.heuristics import DEFAULT_FALLBACK, clamp_selection, get_baseline
from src.models.policy_net import K_MAX, build_policy


class StandalonePTCGAgent:
    """Kaggle-contract agent: `agent(obs_dict) -> list[int]`.

    MAIN decisions are scored by the network; the other 48 contexts delegate to
    a complete rule-based agent, which is legal by construction for all of them.
    """

    def __init__(self,
                 model_path: str | Path,
                 deck_path: str | Path,
                 static_table_path: str | Path,
                 attack_table_path: str | Path,
                 fallback: str = DEFAULT_FALLBACK,
                 main_context: int | None = None,
                 k_max: int = K_MAX,
                 num_threads: int = 1):
        torch.set_num_threads(num_threads)

        checkpoint = torch.load(Path(model_path), map_location="cpu",
                                weights_only=False)
        self.config = checkpoint["config"]

        static_table = load_static_table(static_table_path)
        self.attack_table = load_attack_table(attack_table_path)

        # Rebuilt from the checkpoint's own config, so architecture drift in the
        # yaml cannot silently produce a mismatched load.
        self.model = build_policy(self.config, static_table)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.deck = self._read_deck(deck_path)
        self.fallback = get_baseline(fallback)
        self.k_max = k_max
        if main_context is None:
            from src.env.cabt_wrapper import main_context as _mc  # noqa: PLC0415
            main_context = _mc()
        self.main_context = main_context

        # Observability: the arena and run_arena.py read these to confirm the
        # policy is actually being consulted rather than silently falling back.
        self.stats = {"main": 0, "fallback": 0, "policy_error": 0, "over_k_max": 0}

    @staticmethod
    def _read_deck(path: str | Path) -> list[int]:
        """One card ID per line; tolerates a trailing newline and blank lines."""
        text = Path(path).read_text()
        deck = [int(line.split(",")[0]) for line in text.splitlines() if line.strip()]
        if len(deck) != 60:
            raise ValueError(f"{path}: deck has {len(deck)} cards, expected 60")
        return deck

    @torch.no_grad()
    def _rank(self, obs: dict) -> list[int]:
        """Score the k MAIN candidates and return the best index.

        Uses the same extract_raw_* + collate path as training (invariant 1),
        with a batch of one.
        """
        board = collate_boards([extract_raw_board(obs)])
        actions = collate_actions([extract_raw_actions(obs, self.attack_table)])
        logits, _ = self.model(board, actions)
        return [int(logits[0].argmax())]

    def act(self, obs: dict) -> list[int]:
        select = obs.get("select")
        if select is None:
            # The initial call asks for the decklist, not option indices.
            return list(self.deck)

        if (select.get("context") == self.main_context
                and len(select["option"]) <= self.k_max):
            try:
                choice = self._rank(obs)
                self.stats["main"] += 1
                return clamp_selection(choice, select)
            except Exception:                  # noqa: BLE001 - never forfeit
                self.stats["policy_error"] += 1
        elif (select.get("context") == self.main_context
              and len(select["option"]) > self.k_max):
            self.stats["over_k_max"] += 1

        self.stats["fallback"] += 1
        return clamp_selection(self.fallback(obs), select)

    # The Kaggle contract is a plain callable.
    __call__ = act
