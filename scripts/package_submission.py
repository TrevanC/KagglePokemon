"""Step 6: package the agent into a Kaggle submission bundle.

    python scripts/package_submission.py
    python scripts/package_submission.py --model models/checkpoints/x.pt --no-archive

Input:   a checkpoint, the two feature tables, a deck.csv, and the src/ modules
         the agent needs at inference time
Output:  submission/build/ staging directory and submission/submission.tar.gz

Bundle layout:

    submission/build/
    ├── main.py               entrypoint (from submission/main_template.py)
    ├── agents.py             the rule-based fallback, copied from the arena
    ├── cg/                   the native engine bindings
    ├── best_model.pt         trained weights, optimizer state stripped
    ├── static_table.pt       precomputed (1268, 72) card features
    ├── attack_table.pt       precomputed (1557, 14) attack features
    ├── deck.csv              the 60 cards the agent plays
    └── src/                  VENDORED featurizer + model modules

ON RULE 4 ("no imports reaching outside the deployment folder")
--------------------------------------------------------------
The spec suggested hand-inlining the extractor into standalone_agent.py. That
is the one thing this project must not do: a hand-maintained copy of
state_extractor.py is exactly the train/deploy divergence invariant 1 exists to
prevent, and it fails silently -- training metrics keep improving while the
deployed agent featurizes differently.

Rule 4 constrains the deployment folder's IMPORT GRAPH, not the number of
copies of the source. So packaging vendors the real modules verbatim. The
bundle is fully self-contained AND byte-identical to what was trained against;
VENDORED_MODULES is checked to import nothing outside the bundle.

Validation before archiving -- a bad bundle costs a submission slot:
  - every vendored module imports only from the bundle, torch, or cg;
  - the bundle loads and plays a full match through the harness's own
    load_submission(), from a clean cwd, exactly as Kaggle does;
  - per-decision latency stays under the 10 ms budget.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pokemon-tcg-ai-battle" / "arena"))

ARENA = ROOT / "pokemon-tcg-ai-battle" / "arena"

# Exactly the modules inference touches. Everything here must be free of arena
# imports; src/evaluation/heuristics.py and src/env/cabt_wrapper.py are
# deliberately NOT vendored because they resolve the vendored arena path, which
# does not exist on Kaggle. main.py imports the bundled agents.py instead.
VENDORED_MODULES = [
    "src/__init__.py",
    "src/card_db/__init__.py",
    "src/card_db/static_table.py",
    "src/env/__init__.py",
    "src/env/state_extractor.py",
    "src/models/__init__.py",
    "src/models/card_encoder.py",
    "src/models/action_encoder.py",
    "src/models/state_mlp.py",
    "src/models/policy_net.py",
]

# EVERYTHING THE BUNDLE SHIPS IS NAMESPACED.
# Kaggle's runtime puts kaggle_environments' own package directories ahead of
# the agent folder on sys.path, and it ships generic module names --
# kaggle_environments/envs/lux_ai_s3/agents.py shadowed our `agents` and
# errored the submission before any of our code ran. `src` is just as exposed.
# Renaming both to a unique prefix removes the whole collision class rather
# than racing sys.path ordering, which we do not control.
PKG = "ptcg_src"
FALLBACK_MODULE = "ptcg_agents"

ALLOWED_TOP_LEVEL = {PKG, FALLBACK_MODULE, "torch", "cg", "os", "sys", "math",
                     "random", "json", "gzip", "pathlib", "typing",
                     "dataclasses", "collections", "__future__"}


def rewrite_imports(text: str) -> str:
    """Repoint vendored modules at the namespaced package."""
    return text.replace("from src.", f"from {PKG}.").replace(
        "import src.", f"import {PKG}.")


def check_imports(build: Path) -> list[str]:
    """Every import in the bundle must resolve inside it (or to torch/stdlib).

    Catches the classic Rule 4 failure -- a module that works locally because
    src/ happens to be on sys.path and dies on Kaggle.
    """
    problems = []
    for py in sorted(build.rglob("*.py")):
        if "cg/" in str(py.relative_to(build)):
            continue                      # vendor code, shipped as-is
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError as e:
            problems.append(f"{py.relative_to(build)}: syntax error {e}")
            continue
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                mod = node.names[0].name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod = node.module.split(".")[0]
            if mod and mod not in ALLOWED_TOP_LEVEL:
                problems.append(
                    f"{py.relative_to(build)}:{node.lineno} imports {mod!r}, "
                    "which is not in the bundle")
    return problems


def strip_checkpoint(src_path: Path, dest: Path) -> tuple[int, int]:
    """Copy the checkpoint without optimizer state. Returns (before, after)."""
    import torch

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    lean = {k: v for k, v in ckpt.items() if k != "optimizer_state_dict"}
    torch.save(lean, dest)
    return src_path.stat().st_size, dest.stat().st_size


def build_bundle(model: Path, deck: Path, static: Path, attack: Path,
                 build: Path) -> None:
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    (build / ".gitkeep").touch()

    shutil.copytree(ARENA / "cg", build / "cg")
    shutil.copy2(ARENA / "agents.py", build / f"{FALLBACK_MODULE}.py")
    shutil.copy2(ROOT / "submission" / "main_template.py", build / "main.py")
    shutil.copy2(deck, build / "deck.csv")
    shutil.copy2(static, build / "static_table.pt")
    shutil.copy2(attack, build / "attack_table.pt")

    for rel in VENDORED_MODULES:
        dest = build / rel.replace("src/", f"{PKG}/", 1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rewrite_imports((ROOT / rel).read_text()))

    before, after = strip_checkpoint(model, build / "best_model.pt")
    print(f"  checkpoint {before/1e6:.1f} MB -> {after/1e6:.1f} MB "
          "(optimizer state stripped)")
    prune_pycache(build)


def prune_pycache(build: Path) -> None:
    """Drop __pycache__ from the bundle.

    The vendored cg/ carries .pyc files built by other Python versions, and the
    validation run generates more. They are dead weight in the archive and
    stale bytecode is the last thing to ship to a runtime we cannot inspect.
    """
    for cache in sorted(build.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache, ignore_errors=True)


KAGGLE_LOADER_PROBE = r"""
import os, sys, json, tempfile

# --- reproduce Kaggle's hostile sys.path -------------------------------------
# kaggle_environments ships generic module names (envs/lux_ai_s3/agents.py) and
# its package dirs sit AHEAD of the agent folder. Plant poisoned `agents` and
# `src` modules in front of everything: if the bundle imports either, it dies
# here instead of on the ladder.
_poison = tempfile.mkdtemp()
open(os.path.join(_poison, "agents.py"), "w").write(
    "raise ImportError('POISON: bundle imported the generic module `agents`')")
os.makedirs(os.path.join(_poison, "src"), exist_ok=True)
open(os.path.join(_poison, "src", "__init__.py"), "w").write(
    "raise ImportError('POISON: bundle imported the generic package `src`')")

os.chdir({build!r})
# Reproduce Kaggle's ordering EXACTLY: the agent folder is already on
# sys.path, but BEHIND the runtime's own package dirs. A bundle that guards
# its insert with "if not already in sys.path" will therefore never reach the
# front, and the poisoned generic names win -- which is precisely how the real
# submission errored. Do not simplify this to a plain insert(0, poison): that
# makes the probe pass a bundle that Kaggle rejects.
# Append the path EXACTLY as main.py computes it (os.path.abspath(".") after
# the chdir). Appending a differently-spelled but equivalent path -- /tmp vs
# /private/tmp on macOS -- makes the "already in sys.path" condition false and
# silently defeats this whole check.
sys.path.append(os.path.abspath("."))
sys.path.insert(0, _poison)
# Reproduce kaggle_environments.agent.get_last_callable: a bare exec into a
# namespace with NO __file__. The arena's load_submission uses
# spec.loader.exec_module, which DOES set __file__, so a bundle that derives
# paths from it passes locally and errors instantly on Kaggle.
src = open("main.py").read()
env = {{"__name__": "__main__"}}
exec(compile(src, "main.py", "exec"), env)
fn = env.get("agent")
assert callable(fn), "main.py defines no callable `agent`"
deck = fn({{"select": None, "logs": [], "current": None}})
assert isinstance(deck, list) and len(deck) == 60, f"bad deck: {{deck!r}}"

# Everything the bundle loaded must come FROM the bundle, not from the runtime.
strays = [n for n, m in sys.modules.items()
          if getattr(m, "__file__", None) and _poison in str(m.__file__)]
assert not strays, f"resolved to poisoned modules: {{strays}}"
print(json.dumps({{"deck": len(deck), "status": env.get("_STATUS", "?")}}))
"""


def check_kaggle_loader(build: Path) -> bool:
    """Load main.py the way Kaggle does, in a fresh interpreter.

    Two failure modes this reproduces, both of which passed every other local
    check and errored the real submission:
      1. exec without __file__ (Kaggle's loader does not set it);
      2. generic module names shadowed by kaggle_environments' own packages,
         which sit earlier on sys.path than the agent folder.
    """
    proc = subprocess.run(
        [sys.executable, "-c", KAGGLE_LOADER_PROBE.format(build=str(build))],
        capture_output=True, text=True, cwd=str(build))
    if proc.returncode != 0:
        print("  FAIL: bundle does not load under Kaggle's exec-based loader")
        for line in proc.stderr.strip().splitlines()[-6:]:
            print(f"    {line}")
        return False
    print(f"  loads without __file__: {proc.stdout.strip()}")
    if '"status": "ok"' not in proc.stdout:
        print("  FAIL: policy did not load -- the bundle would play greedy only")
        return False
    return True


def validate(build: Path, games: int) -> bool:
    """Load the bundle exactly as Kaggle does and play real matches."""
    from runner import load_submission, run_match  # noqa: PLC0415

    from src.evaluation.heuristics import get_baseline  # noqa: PLC0415

    agent, deck = load_submission(str(build))
    if not deck or len(deck) != 60:
        print(f"  FAIL: bundle returned a {len(deck) if deck else 0}-card deck")
        return False
    print(f"  bundle loads; returned a {len(deck)}-card deck")

    greedy = get_baseline("greedy")
    wins = errors = steps = 0
    t0 = time.perf_counter()
    for i in range(games):
        r = (run_match(agent, greedy, deck, deck) if i % 2 == 0
             else run_match(greedy, agent, deck, deck))
        seat = 0 if i % 2 == 0 else 1
        wins += int(r.winner == seat)
        steps += r.steps
        if r.error:
            errors += 1
            print(f"  FAIL: game {i} forfeited: {r.error}")
    ms = (time.perf_counter() - t0) / max(steps, 1) * 1000
    print(f"  {games} games vs greedy: {wins} wins, {errors} forfeits")
    print(f"  {ms:.2f} ms per decision (budget 10 ms)")
    return errors == 0 and ms < 10.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path,
                   default=ROOT / "models/checkpoints/bc_mlp_yushin_ito_v1.pt")
    p.add_argument("--deck", type=Path,
                   default=ROOT / "data/processed/deck_yushin_ito.csv")
    p.add_argument("--static", type=Path,
                   default=ROOT / "data/processed/static_feature_table.pt")
    p.add_argument("--attack", type=Path,
                   default=ROOT / "data/processed/attack_feature_table.pt")
    p.add_argument("--build", type=Path, default=ROOT / "submission/build")
    p.add_argument("--out", type=Path, default=ROOT / "submission/submission.tar.gz")
    p.add_argument("--games", type=int, default=10, help="validation games")
    p.add_argument("--no-archive", action="store_true")
    args = p.parse_args()

    for path in (args.model, args.deck, args.static, args.attack):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    print("building bundle...")
    build_bundle(args.model, args.deck, args.static, args.attack, args.build)

    print("\nchecking imports (rule 4)...")
    problems = check_imports(args.build)
    if problems:
        for msg in problems:
            print(f"  FAIL: {msg}", file=sys.stderr)
        return 1
    print("  every import resolves inside the bundle")

    print("\nsimulating Kaggle's loader...")
    if not check_kaggle_loader(args.build):
        print("\nvalidation FAILED -- not archiving", file=sys.stderr)
        return 1

    print("\nvalidating against the engine...")
    if not validate(args.build, args.games):
        print("\nvalidation FAILED -- not archiving", file=sys.stderr)
        return 1

    if args.no_archive:
        print(f"\nbuilt {args.build} (not archived)")
        return 0

    # Validation imported the bundle, which regenerates bytecode.
    prune_pycache(args.build)
    with tarfile.open(args.out, "w:gz") as tar:
        for item in sorted(args.build.iterdir()):
            if item.name == ".gitkeep":
                continue
            tar.add(item, arcname=f"./{item.name}")
    size = args.out.stat().st_size
    print(f"\nwrote {args.out}  ({size/1e6:.1f} MB)")
    print(f"submit with:\n  kaggle competitions submit -c pokemon-tcg-ai-battle "
          f"-f {args.out} -m \"<message>\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
