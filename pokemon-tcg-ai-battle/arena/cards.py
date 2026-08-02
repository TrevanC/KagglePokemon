"""Card database helpers.

Wraps the engine's `all_card_data()` / `all_attack()` so you can look cards up
by ID when writing a smarter agent (observations reference cards by numeric ID).

    from cards import CARDS, ATTACKS, name
    print(name(1158))            # human-readable card name
    print(CARDS[1158].hp)        # full CardData dataclass
    print(ATTACKS[attack_id].damage)
"""

from __future__ import annotations

from cg.api import all_card_data, all_attack

# id -> CardData / Attack dataclass (see cg/api.py for all fields).
CARDS = {c.cardId: c for c in all_card_data()}
ATTACKS = {a.attackId: a for a in all_attack()}


def name(card_id: int) -> str:
    c = CARDS.get(card_id)
    return c.name if c else f"<unknown {card_id}>"


def describe_deck(deck: list) -> str:
    """Summarize a 60-card deck as a name -> count listing."""
    from collections import Counter
    counts = Counter(deck)
    lines = [f"{cnt}x  {name(cid)}  (id {cid})"
             for cid, cnt in sorted(counts.items(), key=lambda kv: -kv[1])]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from runner import read_deck
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "../sample_submission/sample_submission/deck.csv"
    print(f"card pool: {len(CARDS)} cards, {len(ATTACKS)} attacks\n")
    print(f"deck {path}:\n")
    print(describe_deck(read_deck(path)))
