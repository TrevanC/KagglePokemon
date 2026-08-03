"""Harvest Kaggle episode replays for expert training data.

    python scripts/fetch_kaggle_replays.py --submissions 55186239 --limit 50
    python scripts/fetch_kaggle_replays.py --submissions-file subs.txt

Why this exists
---------------
Self-play can only produce lucario-quality decisions, and cloning the agent we
already delegate to is a null operation (spec §1). Real Kaggle episodes are the
only source of MAIN decisions stronger than the fallback.

How to get submission IDs
-------------------------
`kaggle competitions episodes` takes a SUBMISSION id, not a team id, and it
works for *other people's* submissions. Submission ids appear in leaderboard
URLs when you click a team on the competition leaderboard page:

    .../leaderboard?submissionId=55186239&episodeId=89625016
                                 ^^^^^^^^

Collect those for the top-ranked teams (see the TeamId/Score list from
`kaggle competitions leaderboard <comp> -d`) and pass them here. Each episode
contains BOTH players' decisions and `rewards` identifies the winner, so one
strong player's episode list also yields their opponents' play.

Politeness: this hits Kaggle's API once per episode. Default delay is 1.0s.
Do not lower it just to go faster.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset.kaggle_replays import winner_of  # noqa: E402

COMPETITION = "pokemon-tcg-ai-battle"
DEFAULT_OUT = Path("data/replays/kaggle")


def _kaggle(*args: str) -> str:
    """Run the kaggle CLI, returning stdout. Raises on failure."""
    proc = subprocess.run(
        ["kaggle", *args], capture_output=True, text=True,
        env={**os.environ, "PATH": f"{Path.home()}/anaconda3/bin:{os.environ.get('PATH','')}"},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{proc.stderr or proc.stdout}")
    return proc.stdout


def list_episodes(submission_id: str) -> list[str]:
    """Episode ids for one submission, newest first."""
    out = _kaggle("competitions", "episodes", str(submission_id), "-v")
    ids = []
    for line in out.splitlines():
        head = line.split(",", 1)[0].strip()
        if head.isdigit():
            ids.append(head)
    return ids


def fetch_replay(episode_id: str, out_dir: Path) -> Path | None:
    """Download one replay. Returns its path, or None if already present."""
    dest = out_dir / f"episode-{episode_id}-replay.json"
    if dest.exists() and dest.stat().st_size > 0:
        return None
    _kaggle("competitions", "replay", str(episode_id), "-p", str(out_dir), "-q")
    return dest if dest.exists() else None


def summarize(path: Path) -> dict:
    """Cheap sanity read: who played, who won, how many decisions."""
    d = json.loads(path.read_text())
    steps = d["steps"]
    rewards = d.get("rewards") or []
    # Reward entries may be None when an agent errors out (statuses=[...,'ERROR']),
    # so winner selection goes through the shared helper rather than max().
    winner = winner_of(d)
    n_dec = sum(
        1 for step in steps for slot in step
        if slot.get("status") == "ACTIVE" and (slot.get("observation") or {}).get("select")
    )
    return {
        "episode": d["info"]["EpisodeId"],
        "teams": d["info"]["TeamNames"],
        "rewards": rewards,
        "winner": winner,
        "decisions": n_dec,
        "bytes": path.stat().st_size,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submissions", nargs="*", default=[],
                   help="submission ids (from leaderboard URLs)")
    p.add_argument("--submissions-file", type=Path,
                   help="file with one submission id per line")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=None,
                   help="max NEW replays to download this run")
    p.add_argument("--delay", type=float, default=1.0,
                   help="seconds between API calls (default 1.0; do not lower)")
    p.add_argument("--list-only", action="store_true",
                   help="print episode ids without downloading")
    args = p.parse_args()

    subs = list(args.submissions)
    if args.submissions_file:
        subs += [ln.strip() for ln in args.submissions_file.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
    if not subs:
        p.error("give --submissions and/or --submissions-file")

    args.out.mkdir(parents=True, exist_ok=True)

    episodes: list[str] = []
    for s in subs:
        try:
            got = list_episodes(s)
        except RuntimeError as e:
            print(f"[warn] submission {s}: {e}", file=sys.stderr)
            continue
        print(f"submission {s}: {len(got)} episodes")
        episodes += got
        time.sleep(args.delay)

    episodes = list(dict.fromkeys(episodes))          # dedupe, keep order
    have = {p.name for p in args.out.glob("episode-*-replay.json")}
    todo = [e for e in episodes
            if f"episode-{e}-replay.json" not in have][: args.limit]
    print(f"\n{len(episodes)} unique episodes | {len(have)} already on disk "
          f"| downloading {len(todo)}")

    if args.list_only:
        print("\n".join(episodes))
        return 0

    got = 0
    for i, ep in enumerate(todo, 1):
        try:
            path = fetch_replay(ep, args.out)
        except RuntimeError as e:
            print(f"[warn] episode {ep}: {e}", file=sys.stderr)
            continue
        if path:
            got += 1
            s = summarize(path)
            print(f"  [{i}/{len(todo)}] {s['episode']}  {s['decisions']:>3} decisions  "
                  f"{s['bytes']/1e6:.1f}MB  {s['teams']} -> winner {s['winner']}")
        time.sleep(args.delay)

    total = len(list(args.out.glob("episode-*-replay.json")))
    print(f"\ndownloaded {got} new | {total} replays total in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
