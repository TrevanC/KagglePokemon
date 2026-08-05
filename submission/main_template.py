"""Kaggle entrypoint. Copied into submission/build/main.py by
scripts/package_submission.py -- edit it HERE, not in the build folder.

Kaggle mounts the bundle at /kaggle_simulations/agent/, while the local arena
runs it from the folder itself, so every path is resolved against both.

DEGRADES INSTEAD OF FORFEITING. If torch is missing from the runtime, or the
checkpoint fails to load, or the policy raises mid-game, this falls back to the
bundled rule-based agent rather than dying. An exception here does not lose a
turn, it loses the match -- an illegal or missing selection is an instant
forfeit. Playing at baseline strength beats forfeiting 100% of games, so every
failure path leads to `greedy`.
"""

import os
import sys

# NEVER use __file__ here. Kaggle loads this module with a bare
# exec(code_object, env) whose env has no __file__, so referencing it raises
# NameError before a single line of agent code runs -- an instant ERROR on the
# whole submission. (The local arena's load_submission uses
# spec.loader.exec_module, which DOES define __file__, so this failure is
# invisible locally; package_submission.py reproduces Kaggle's loader
# specifically to catch it.) This is why the official sample resolves
# "deck.csv" by trying the relative path first and the mount point second.
_KAGGLE = "/kaggle_simulations/agent"
_ROOT = os.path.abspath(".") if os.path.exists("deck.csv") else _KAGGLE

# Force the bundle to the FRONT of sys.path -- remove-then-insert, never a
# "if not already present" guard. Kaggle already has the agent folder on
# sys.path but BEHIND its own package dirs, so a presence check skips the
# insert and leaves kaggle_environments' generic modules shadowing ours. That
# is what made `from agents import greedy_agent` resolve to
# kaggle_environments/envs/lux_ai_s3/agents.py.
for _p in (_KAGGLE, _ROOT):
    if os.path.isdir(_p):
        while _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)


def _path(name):
    p = os.path.join(_ROOT, name)
    return p if os.path.exists(p) else os.path.join(_KAGGLE, name)


# The rule-based fallback: pure Python + cg.api, no torch. This import must
# never fail, so it is not wrapped -- if cg.api is unavailable nothing works.
#
# NAMESPACED ON PURPOSE. This was `from agents import greedy_agent` and Kaggle
# resolved it to kaggle_environments/envs/lux_ai_s3/agents.py -- their package
# dirs sit ahead of the agent folder on sys.path, so any generic name is theirs,
# not ours. Same reason the vendored package is ptcg_src rather than src.
from cg.api import SelectContext  # noqa: E402
from ptcg_agents import greedy_agent  # noqa: E402

MAIN_CONTEXT = int(SelectContext.MAIN)
K_MAX = 80


def read_deck():
    with open(_path("deck.csv")) as f:
        rows = [r.strip() for r in f.read().split("\n") if r.strip()]
    return [int(r.split(",")[0]) for r in rows[:60]]


DECK = read_deck()

# ---------------------------------------------------------------------------
# Policy, loaded once. Any failure leaves _MODEL None and the agent plays
# greedy for the whole match.
# ---------------------------------------------------------------------------
_MODEL = None
_ATTACK_TABLE = None
_EXTRACT = None
_STATUS = "not loaded"

try:
    import torch

    torch.set_num_threads(1)

    from ptcg_src.card_db.static_table import load_attack_table, load_static_table
    from ptcg_src.env.state_extractor import (
        collate_actions,
        collate_boards,
        extract_raw_actions,
        extract_raw_board,
    )
    from ptcg_src.models.policy_net import build_policy

    _ckpt = torch.load(_path("best_model.pt"), map_location="cpu",
                       weights_only=False)
    _MODEL = build_policy(_ckpt["config"], load_static_table(_path("static_table.pt")))
    _MODEL.load_state_dict(_ckpt["model_state_dict"])
    _MODEL.eval()
    _ATTACK_TABLE = load_attack_table(_path("attack_table.pt"))
    _EXTRACT = (extract_raw_board, extract_raw_actions, collate_boards, collate_actions)
    _STATUS = "ok"
except Exception as exc:  # noqa: BLE001 - degrade, never forfeit
    _STATUS = f"policy unavailable ({type(exc).__name__}: {exc}); using greedy"
    print(_STATUS, file=sys.stderr)


def _clamp(indices, select):
    """Force any wishlist into a legal selection. An illegal one forfeits."""
    n = len(select["option"])
    lo, hi = select["minCount"], min(select["maxCount"], n)
    out = [i for i in dict.fromkeys(indices) if isinstance(i, int) and 0 <= i < n][:hi]
    if len(out) < lo:
        out += [i for i in range(n) if i not in out][: lo - len(out)]
    return out


def _rank(obs):
    raw_board, raw_actions, coll_b, coll_a = _EXTRACT
    with torch.no_grad():
        board = coll_b([raw_board(obs)])
        actions = coll_a([raw_actions(obs, _ATTACK_TABLE)])
        logits, _ = _MODEL(board, actions)
    return [int(logits[0].argmax())]


def agent(obs, config=None):
    select = obs["select"]
    if select is None:              # the initial call asks for the decklist
        return list(DECK)

    if (_MODEL is not None
            and select["context"] == MAIN_CONTEXT
            and len(select["option"]) <= K_MAX):
        try:
            return _clamp(_rank(obs), select)
        except Exception:           # noqa: BLE001 - degrade, never forfeit
            pass
    return _clamp(greedy_agent(obs), select)
