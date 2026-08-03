"""The one data format everything else speaks.

A *record* is a single decision an expert made:

    {
      "obs":     <the raw observation dict the agent was given>,
      "action":  <list[int], the option indices the expert returned>,
      "player":  <0 or 1>,
      "reward":  <+1 if this player eventually won, -1 lost, 0 draw>,
      "source":  <"selfplay:lucario" | "kaggle:89083821" | ...>
    }

Both data sources — locally generated self-play games and downloaded Kaggle
episode replays — produce exactly this. That means the featurizer, the dataset
and the training loop never need to know where the data came from, and you can
mix the two freely.

Records are stored as gzipped JSON Lines (one JSON object per line). That's a
good default for this kind of data: streamable, appendable, greppable, and it
compresses ~10x because observations repeat a lot of structure.
"""

from __future__ import annotations

import gzip
import json
from typing import Iterable, Iterator


def write_records(path: str, records: Iterable[dict]) -> int:
    """Write records to a .jsonl.gz file. Returns how many were written."""
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


def read_records(path: str) -> Iterator[dict]:
    """Stream records back out of a .jsonl.gz file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_records(path: str) -> int:
    return sum(1 for _ in read_records(path))
