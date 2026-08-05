"""Local ladder: multi-game bot-vs-bot tournaments.

Built on top of the vendored harness rather than a new match loop --
pokemon-tcg-ai-battle/arena/tournament.py already runs many games and reports
win rates with 95% Wilson confidence intervals, and arena/runner.py plays a
single match. Reach them through src/env/cabt_wrapper.py.

What this module adds on top:
  - Pitting a loaded candidate agent against the baselines in heuristics.py.
  - Round-robin ladders across several agents, for RL opponent pools.
  - The promotion decision: does the candidate beat the incumbent by enough
    margin to replace it?

Two things the harness README warns about, which the promotion logic depends
on:
  - The engine's RNG is not seedable, so single games are noisy. Use enough
    games that the confidence interval is narrower than the margin you care
    about, and always swap sides to cancel position bias.
  - Both agents run in one process. Fine for a model vs a baseline; avoid two
    agents that both use the engine's search_* planning API.

Reference baseline strength (from arena/README.md):
    lucario > iono > greedy > first > random
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.env.cabt_wrapper import play_match


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval -- correct near 0% and 100%, where the normal
    approximation produces impossible bounds."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass
class ArenaResult:
    games: int = 0
    candidate_wins: int = 0
    baseline_wins: int = 0
    draws: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.candidate_wins / self.games if self.games else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.candidate_wins, self.games)

    def summary(self, candidate: str = "candidate", baseline: str = "baseline") -> str:
        lo, hi = self.interval
        lines = [
            f"{candidate} vs {baseline}   ({self.games} games"
            f"{', sides swapped' if self.games else ''})",
            f"  {candidate:<26} {self.candidate_wins:4d}  {self.win_rate:6.1%}  "
            f"[{lo:.1%}-{hi:.1%}]",
            f"  {baseline:<26} {self.baseline_wins:4d}  "
            f"{self.baseline_wins / max(self.games, 1):6.1%}",
        ]
        if self.draws:
            lines.append(f"  {'draw':<26} {self.draws:4d}")
        if self.errors:
            lines.append(f"  FORFEITS/ERRORS ({len(self.errors)}): {self.errors[:3]}")
        return "\n".join(lines)


def should_promote(result: ArenaResult, threshold: float = 0.5) -> bool:
    """True only if the whole confidence interval clears `threshold`.

    Requiring the CI's LOWER BOUND -- not the point estimate -- to clear it is
    what stops a lucky short run from promoting a worse model. With the engine's
    unseedable RNG this is the difference between a real result and noise.

    Any forfeit blocks promotion outright: an illegal selection is a lost match
    on Kaggle, and one that happens locally will happen there.
    """
    return not result.errors and result.interval[0] > threshold


def run_arena(candidate, baseline, candidate_deck: list[int],
              baseline_deck: list[int] | None = None,
              games: int = 100, swap: bool = True, verbose: bool = False
              ) -> ArenaResult:
    """Play `games` matches, alternating seats when `swap`.

    A forfeit counts as a loss for whoever caused it -- which is the point: an
    illegal selection is not a bad move, it is a lost match, so it belongs in
    the win rate rather than being filtered out. Errors are collected verbatim
    too, because a candidate that forfeits even occasionally has a bug that the
    win rate alone understates.
    """
    baseline_deck = candidate_deck if baseline_deck is None else baseline_deck
    result = ArenaResult()

    for i in range(games):
        cand_first = (i % 2 == 0) if swap else True
        if cand_first:
            match = play_match(candidate, baseline, candidate_deck, baseline_deck)
            cand_seat = 0
        else:
            match = play_match(baseline, candidate, baseline_deck, candidate_deck)
            cand_seat = 1

        result.games += 1
        if match.error:
            result.errors.append(f"game {i}: {match.error}")
        if match.winner is None or match.winner < 0:
            result.draws += 1
        elif match.winner == cand_seat:
            result.candidate_wins += 1
        else:
            result.baseline_wins += 1

        if verbose and (i + 1) % 20 == 0:
            lo, hi = result.interval
            print(f"  {i+1}/{games}  {result.win_rate:.1%} [{lo:.1%}-{hi:.1%}]",
                  flush=True)

    return result


def round_robin(agents: dict, deck: list[int], games: int = 50) -> dict:
    """{(a, b): ArenaResult} for every ordered pair. For RL opponent pools.

    Skips pairs where both sides would use the engine's search_* planning API;
    they share one `agent_ptr` in-process and cannot legally play each other.
    """
    out = {}
    names = list(agents)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[(a, b)] = run_arena(agents[a], agents[b], deck, games=games)
    return out
