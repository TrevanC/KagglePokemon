"""Run many matches between two agents and report win rates.

Agents can be baseline names (`random`, `first`, `greedy`) or a path to a
Kaggle submission folder (containing `main.py` + `deck.csv`).

Examples:
    python tournament.py greedy random -n 200
    python tournament.py greedy random -n 200 --swap
    python tournament.py ../sample_submission/sample_submission greedy -n 100
    python tournament.py greedy random --deck0 my_deck.csv --deck1 other.csv
"""

from __future__ import annotations

import argparse
import collections
import math
import os

from agents import AGENTS
from runner import run_match, read_deck, load_submission

_DEFAULT_DECK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "sample_submission", "sample_submission", "deck.csv")


def resolve(spec: str):
    """Resolve an agent spec to (label, agent_callable, deck_or_None)."""
    if spec in AGENTS:
        return spec, AGENTS[spec], None
    if os.path.isdir(spec):
        agent, deck = load_submission(spec)
        return os.path.basename(os.path.normpath(spec)), agent, deck
    raise SystemExit(f"unknown agent '{spec}' — use one of {list(AGENTS)} or a "
                     f"submission folder path")


def wilson(wins: int, n: int) -> tuple:
    """95% Wilson confidence interval for a win proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = wins / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (center - half, center + half)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent0")
    ap.add_argument("agent1")
    ap.add_argument("-n", "--games", type=int, default=100)
    ap.add_argument("--deck0", help="deck.csv for agent0 (overrides submission deck)")
    ap.add_argument("--deck1", help="deck.csv for agent1 (overrides submission deck)")
    ap.add_argument("--swap", action="store_true",
                    help="alternate who starts as player 0 (controls for any "
                         "position bias); reports each agent's overall win rate")
    ap.add_argument("--verbose", action="store_true", help="print every selection")
    args = ap.parse_args()

    label0, agent0, sub_deck0 = resolve(args.agent0)
    label1, agent1, sub_deck1 = resolve(args.agent1)
    if label0 == label1:
        label0, label1 = label0 + "#0", label1 + "#1"

    deck0 = read_deck(args.deck0) if args.deck0 else \
        (sub_deck0 if sub_deck0 else read_deck(_DEFAULT_DECK))
    deck1 = read_deck(args.deck1) if args.deck1 else \
        (sub_deck1 if sub_deck1 else read_deck(_DEFAULT_DECK))

    wins = collections.Counter()          # keyed by label / "draw"
    reasons = collections.Counter()
    total_turns = 0
    forfeits = collections.Counter()

    for g in range(args.games):
        # With --swap, agent1 starts as player 0 on odd games.
        swapped = args.swap and (g % 2 == 1)
        if swapped:
            r = run_match(agent1, agent0, deck1, deck0, verbose=args.verbose)
            winner_label = {0: label1, 1: label0, 2: "draw"}[r.winner]
        else:
            r = run_match(agent0, agent1, deck0, deck1, verbose=args.verbose)
            winner_label = {0: label0, 1: label1, 2: "draw"}[r.winner]
        wins[winner_label] += 1
        reasons[r.reason_text] += 1
        total_turns += r.turns
        if r.error is not None:
            actual_player = (1 - r.error_player) if swapped else r.error_player
            forfeits[(label1 if swapped else label0) if actual_player == 0
                     else (label0 if swapped else label1)] += 1

    n = args.games
    print(f"\n{label0}  vs  {label1}   ({n} games"
          f"{', sides swapped each game' if args.swap else ''})\n")
    for label in (label0, label1, "draw"):
        w = wins[label]
        lo, hi = wilson(w, n)
        bar = "#" * round(40 * w / n)
        print(f"  {label:24s} {w:4d}  {w/n:6.1%}  [{lo:.0%}-{hi:.0%}]  {bar}")
    print(f"\n  avg game length: {total_turns / n:.1f} turns")
    print("  endings:")
    for reason, c in reasons.most_common():
        print(f"    {c:4d}  {reason}")
    if forfeits:
        print("  forfeits (illegal moves):")
        for label, c in forfeits.items():
            print(f"    {c:4d}  by {label}")


if __name__ == "__main__":
    main()
