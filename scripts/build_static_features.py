"""Step 1: precompute the static card feature lookup table.

    python scripts/build_static_features.py
    python scripts/build_static_features.py --verify   # also re-derive the maxima

Input:   the engine's card DB — cg.api.all_card_data() (1267 cards) and
         all_attack() (1556 attacks), NOT the CSV (spec §2)
Output:  data/processed/static_feature_table.pt   tensor of shape (1268, 72)
Config:  configs/feature_config.yaml

Row index IS the card ID; row 0 is zeroed so IDs 1..1267 index directly.

Runs once, offline. Design rule 2: the agent must never build this table during
a live turn — it loads the tensor instead.

Uses src/card_db/static_table.py; see the feature index map there. The CSV
reader in src/card_db/parser.py is reference-only and not on this path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                      / "pokemon-tcg-ai-battle" / "arena"))

from src.card_db.static_table import (  # noqa: E402
    DEFAULT_ATTACK_PATH,
    DEFAULT_PATH,
    FEATURE_BLOCKS,
    MAX_FEATURE_VALUE,
    build_attack_table,
    build_static_table,
    save_attack_table,
    save_static_table,
    verify_static_table,
)


def report_measured_maxima() -> None:
    """Re-derive the normalization divisors from the live card DB (spec §9).

    Run this after an engine update. Every divisor in static_table.py is a
    measured maximum; if one of these prints *above* its divisor, features
    silently leave [0, 1] and the divisor needs updating.
    """
    from cg.api import all_attack, all_card_data

    cards, attacks = all_card_data(), all_attack()
    print("measured against the live engine:")
    print(f"  cards                {len(cards)}   ids {min(c.cardId for c in cards)}"
          f"..{max(c.cardId for c in cards)}")
    print(f"  attacks              {len(attacks)}")
    print(f"  max hp               {max(c.hp for c in cards)}      (divisor 380.0)")
    print(f"  max retreat cost     {max(c.retreatCost for c in cards)}"
          f"        (divisor 4.0)")
    print(f"  max attack damage    {max(a.damage for a in attacks)}"
          f"      (divisor 350.0)")
    print(f"  max energies/attack  {max(len(a.energies) for a in attacks)}"
          f"        (divisor 5.0)")
    print(f"  max attacks/card     {max(len(c.attacks) for c in cards)}"
          f"        (divisor 2.0; slots exist for attack 0 and 1 only)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", type=Path, default=Path(DEFAULT_PATH))
    p.add_argument("--attack-out", type=Path, default=Path(DEFAULT_ATTACK_PATH))
    p.add_argument("--verify", action="store_true",
                   help="also re-derive the normalization maxima from the engine")
    args = p.parse_args()

    if args.verify:
        report_measured_maxima()
        print()

    table = build_static_table()
    checks = verify_static_table(table)

    print(f"static table        {checks['shape']}")
    print(f"  row 0 zeroed      {checks['row0_zeroed']}")
    print(f"  populated rows    {checks['populated_rows']} / "
          f"{checks['expected_populated']}")
    print(f"  value range       [{checks['min']:.4f}, {checks['max']:.4f}]")
    print("  block coverage    " + "  ".join(
        f"{name}:{used}/{width}"
        for (name, _start, width), used
        in zip(FEATURE_BLOCKS, checks["block_columns_used"].values())))
    # Columns inside a block are legitimately all-zero when the enum is wider
    # than the card pool uses (no Dragon weakness, etc.) -- informational only.
    print(f"  unused columns    {checks['dead_columns']}")

    problems = []
    if not checks["row0_zeroed"]:
        problems.append("row 0 is not zeroed -- 1-based card IDs would be off by one")
    if not checks["all_populated"]:
        problems.append(
            f"only {checks['populated_rows']} of {checks['expected_populated']} "
            "card rows are non-zero -- the engine schema may have changed")
    if not checks["in_range"]:
        problems.append(
            f"values outside [0, {MAX_FEATURE_VALUE}] "
            f"({checks['min']:.4f}..{checks['max']:.4f}) "
            "-- a normalization divisor is stale, see --verify")
    if checks["dead_blocks"]:
        problems.append(
            f"feature blocks {checks['dead_blocks']} are entirely zero "
            "-- they are not being written")
    if problems:
        for msg in problems:
            print(f"\n[FAIL] {msg}", file=sys.stderr)
        return 1

    path = save_static_table(table, args.out)
    print(f"\nwrote {path}  ({path.stat().st_size / 1e3:.0f} kB)")

    # MAIN's ATTACK options carry only an attackId, so the action encoder needs
    # damage/cost keyed by attack id and cannot call all_attack() at turn time.
    atk = build_attack_table()
    atk_path = save_attack_table(atk, args.attack_out)
    print(f"attack table        {tuple(atk.shape)}  "
          f"rows used {int((atk.abs().sum(dim=1) > 0).sum())}  "
          f"range [{float(atk.min()):.4f}, {float(atk.max()):.4f}]")
    print(f"wrote {atk_path}  ({atk_path.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
