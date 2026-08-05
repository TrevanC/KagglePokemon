"""Recover decklists (and per-deck win rates) from harvested Kaggle replays.

    python scripts/mine_decks.py
    python scripts/mine_decks.py --team "Yushin Ito" --write data/processed/deck_yushin_ito.csv

Input:   data/replays/kaggle/episode-*-replay.json
Output:  a ranked report; optionally one team's deck as a Kaggle deck.csv

Every replay carries both players' 60-card decks in steps[1][p]["action"] --
the answer to the initial `select is None` deck request, subject to the same
+1 action lag as everything else. So harvesting replays yields the top players'
exact decklists for free, alongside their decisions.

Deck choice is a separate lever from policy quality: the model clones MAIN
decisions, but which 60 cards it plays is fixed at submission time and is not
something behavioral cloning can learn.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                      / "pokemon-tcg-ai-battle" / "arena"))

from src.dataset.kaggle_replays import DECK_SIZE, decks_of, winner_of  # noqa: E402


def scan(replay_dir: Path) -> tuple[dict, dict]:
    """-> (per-team stats, per-deck stats). Deck identity is the sorted 60."""
    teams: dict[str, dict] = collections.defaultdict(
        lambda: {"games": 0, "wins": 0, "decks": collections.Counter()})
    decks: dict[tuple, dict] = collections.defaultdict(
        lambda: {"games": 0, "wins": 0, "teams": collections.Counter()})

    files = sorted(replay_dir.glob("episode-*-replay.json"))
    for i, path in enumerate(files, 1):
        if i % 200 == 0:
            print(f"  ...{i}/{len(files)}", file=sys.stderr)
        try:
            replay = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] {path.name}: {e}", file=sys.stderr)
            continue

        names = replay.get("info", {}).get("TeamNames") or []
        winner = winner_of(replay)
        for p, deck in decks_of(replay).items():
            team = names[p] if p < len(names) else f"player{p}"
            key = tuple(sorted(deck))
            won = int(winner == p)
            teams[team]["games"] += 1
            teams[team]["wins"] += won
            teams[team]["decks"][key] += 1
            decks[key]["games"] += 1
            decks[key]["wins"] += won
            decks[key]["teams"][team] += 1
    return teams, decks


def write_deck_csv(deck: list[int], path: Path) -> None:
    """Kaggle deck.csv is one card ID per line, 60 lines, no header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(c) for c in deck) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replays", type=Path, default=Path("data/replays/kaggle"))
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--team", help="show this team's deck in full")
    p.add_argument("--write", type=Path, help="write --team's deck to a deck.csv")
    args = p.parse_args()

    teams, decks = scan(args.replays)
    if not teams:
        print(f"no decks found in {args.replays}", file=sys.stderr)
        return 1

    print(f"\n{len(teams)} teams, {len(decks)} distinct decks\n")
    print(f"{'games':>6} {'wins':>6} {'win%':>6}  {'decks':>5}  team")
    for team, s in sorted(teams.items(), key=lambda kv: -kv[1]["games"])[:args.top]:
        print(f"{s['games']:6d} {s['wins']:6d} {s['wins']/s['games']:6.1%}  "
              f"{len(s['decks']):5d}  {team}")

    print(f"\nmost-played decks:")
    print(f"{'games':>6} {'wins':>6} {'win%':>6}  top team")
    for _key, s in sorted(decks.items(), key=lambda kv: -kv[1]["games"])[:args.top]:
        top = s["teams"].most_common(1)[0][0]
        print(f"{s['games']:6d} {s['wins']:6d} {s['wins']/s['games']:6.1%}  {top}")

    if args.team:
        if args.team not in teams:
            print(f"\nno team named {args.team!r}", file=sys.stderr)
            return 1
        s = teams[args.team]
        print(f"\n=== {args.team} ===")
        print(f"{s['games']} games, {s['wins']} wins ({s['wins']/s['games']:.1%}), "
              f"{len(s['decks'])} distinct deck(s)")
        key, count = s["decks"].most_common(1)[0]
        deck = list(key)
        print(f"primary deck used in {count}/{s['games']} games:")
        try:
            from cards import name
        except ImportError:
            def name(cid):  # noqa: ANN001
                return ""
        for cid, n in sorted(collections.Counter(deck).items()):
            print(f"  {n}x  {cid:5d}  {name(cid)}")
        if args.write:
            write_deck_csv(deck, args.write)
            print(f"\nwrote {args.write} ({len(deck)} cards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
