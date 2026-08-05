"""PyTorch Dataset for behavioral cloning / imitation learning.

Reads gzipped-JSONL records (src/dataset/records.py) and featurizes them
ON THE FLY through src/env/state_extractor.py.

WHY NO PRECOMPUTE STAGE
-----------------------
The original design had scripts/process_dataset.py bake records into
train_bc_v1.pt / val_bc_v1.pt tensors. Measured, that is the wrong trade:

    raw featurization      0.48 ms/decision  -> 34 s/epoch on 1 core,
                                                ~8 s with 4 DataLoader workers
    precomputed tensors    ~14.9 kB/record   -> 1.08 GB for 70k records,
                                                dominated by a discard
                                                histogram that is 7.4 non-zero
                                                of 1268 (99.4% zeros)

On-the-fly is cheaper in both time and space, and it makes invariant 1
structural rather than aspirational: training and live inference call the same
extract_raw_* functions, so they cannot silently desync. That is the failure
this project most needs to design out -- it is invisible in every metric.

The learned half of featurization (card encoder, discard projections) stays in
the model, so the Dataset emits only deterministic raw features.

Only SelectContext.MAIN decisions belong here (spec §1) -- the other 48
contexts are served by the fallback and are not learned. MAIN is always
minCount == maxCount == 1, so the target is a single index into the k
candidates.

Padding to a common k happens in collate, never truncation: dropping a
candidate can remove the expert's chosen move and leave the target undefined.
Decisions with k > K_MAX are filtered out at load time instead.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.dataset.records import read_records
from src.env.state_extractor import (
    collate_actions,
    collate_boards,
    extract_raw_actions,
    extract_raw_board,
)


class PTCGReplayDataset(Dataset):
    """Records -> (RawBoard, RawActions, target index).

    Records are held in memory as decoded dicts: 70k records is ~7 MB gzipped
    and a few hundred MB decoded, which is affordable and keeps epochs free of
    repeated gzip work. `team` filters to a single expert -- cloning a mixture
    of policies that play different decks averages incompatible strategies.
    """

    def __init__(self,
                 records_path: str | Path,
                 attack_table: torch.Tensor,
                 team: str | None = None,
                 k_max: int = 80,
                 limit: int | None = None):
        self.attack_table = attack_table
        self.k_max = k_max
        self.records: list[dict] = []
        self.skipped = {"team": 0, "k_max": 0, "multi_select": 0, "illegal": 0}

        for rec in read_records(records_path):
            if team is not None and rec.get("team") != team:
                self.skipped["team"] += 1
                continue
            select = rec["obs"]["select"]
            k = len(select["option"])
            if k > k_max:
                self.skipped["k_max"] += 1
                continue
            action = rec["action"]
            # MAIN is single-select by construction; anything else here would
            # mean the record set contains non-MAIN contexts.
            if len(action) != 1:
                self.skipped["multi_select"] += 1
                continue
            if not (0 <= action[0] < k):
                self.skipped["illegal"] += 1
                continue
            self.records.append(rec)
            if limit is not None and len(self.records) >= limit:
                break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        obs = rec["obs"]
        return (extract_raw_board(obs),
                extract_raw_actions(obs, self.attack_table),
                rec["action"][0])


def bc_collate(samples):
    """-> (BoardBatch, ActionBatch, targets (B,)).

    Padding the candidate dimension to the batch max (not a fixed K_MAX) keeps
    the scorer's work proportional to the actual decisions in the batch; the
    mask makes the two equivalent.
    """
    boards, actions, targets = zip(*samples)
    return (collate_boards(list(boards)),
            collate_actions(list(actions)),
            torch.tensor(targets, dtype=torch.long))
