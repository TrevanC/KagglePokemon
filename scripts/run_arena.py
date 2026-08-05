"""Step 5: evaluate a candidate model against baselines in the local arena.

    python scripts/run_arena.py --candidate models/checkpoints/bc_mlp_yushin_ito_v1.pt
    python scripts/run_arena.py --candidate ... --baseline lucario -n 200
    python scripts/run_arena.py --candidate ... --fallback lucario --promote

Input:   a checkpoint from models/checkpoints/, baselines from
         src/evaluation/heuristics.py
Output:  win rates with 95% Wilson confidence intervals; with --promote and a
         clear win, promotion of the checkpoint to models/registry/

Always swaps sides to cancel position bias, and promotion requires the CI's
LOWER bound to clear the threshold -- the engine's RNG is not seedable and
single games are noisy.

This, not validation loss, is the promotion gate for the model registry.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                      / "pokemon-tcg-ai-battle" / "arena"))

from src.card_db.static_table import DEFAULT_ATTACK_PATH, DEFAULT_PATH  # noqa: E402
from src.evaluation.arena import run_arena, should_promote  # noqa: E402
from src.evaluation.heuristics import (  # noqa: E402
    BASELINES, DEFAULT_FALLBACK, get_baseline,
)
from src.models.standalone_agent import StandalonePTCGAgent  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", type=Path, required=True, help="checkpoint .pt")
    p.add_argument("--deck", type=Path,
                   default=Path("data/processed/deck_yushin_ito.csv"))
    p.add_argument("--baseline", default="greedy",
                   help=f"opponent: one of {list(BASELINES)}")
    p.add_argument("--fallback", default=DEFAULT_FALLBACK,
                   help="agent handling the candidate's 48 non-MAIN contexts")
    p.add_argument("--baseline-deck", type=Path,
                   help="deck for the opponent (defaults to the candidate's)")
    p.add_argument("-n", "--games", type=int, default=100)
    p.add_argument("--no-swap", action="store_true")
    p.add_argument("--static-table", type=Path, default=Path(DEFAULT_PATH))
    p.add_argument("--attack-table", type=Path, default=Path(DEFAULT_ATTACK_PATH))
    p.add_argument("--promote", action="store_true",
                   help="copy to models/registry/best_policy.pt if it clears the gate")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="CI lower bound must exceed this to promote")
    p.add_argument("--registry", type=Path, default=Path("models/registry"))
    args = p.parse_args()

    agent = StandalonePTCGAgent(args.candidate, args.deck,
                                args.static_table, args.attack_table,
                                fallback=args.fallback)
    opponent = get_baseline(args.baseline)

    from src.env.cabt_wrapper import read_deck  # noqa: PLC0415
    baseline_deck = read_deck(args.baseline_deck) if args.baseline_deck else None

    label = f"{args.candidate.stem}(+{args.fallback})"
    print(f"{label} vs {args.baseline}, {args.games} games"
          f"{'' if args.no_swap else ', swapping sides'}...", flush=True)

    result = run_arena(agent, opponent, agent.deck, baseline_deck,
                       games=args.games, swap=not args.no_swap, verbose=True)

    print()
    print(result.summary(label, args.baseline))
    total = agent.stats["main"] + agent.stats["fallback"]
    print(f"\n  policy scored {agent.stats['main']} of {total} decisions "
          f"({agent.stats['main']/max(total,1):.1%} MAIN); "
          f"fallback {agent.stats['fallback']}; "
          f"policy errors {agent.stats['policy_error']}; "
          f"over k_max {agent.stats['over_k_max']}")

    promote = should_promote(result, args.threshold)
    print(f"\n  promotion gate (CI lower bound > {args.threshold:.0%}, "
          f"no forfeits): {'PASS' if promote else 'FAIL'}")

    if args.promote:
        if promote:
            args.registry.mkdir(parents=True, exist_ok=True)
            dest = args.registry / "best_policy.pt"
            shutil.copy2(args.candidate, dest)
            print(f"  promoted -> {dest}")
        else:
            print("  not promoted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
