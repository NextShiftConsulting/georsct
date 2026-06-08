"""
test_ladder_invariants.py -- enforce the representation-ladder definition.

The ladder's entire scientific claim is that each rung is the previous rung
PLUS additions: R0 subset R1 subset R2, with R1->R2 measuring "+ temporal"
and never "+ temporal - something". That additive containment is a property
the code must satisfy, not one a reviewer is trusted to notice. This test makes
a silent feature drop (the kind that confounded H3) impossible to merge.

Why AST instead of import: the training modules pull in boto3/sklearn/pandas at
import time. This test extracts the feature-list literals statically, so it runs
in any CI environment with nothing but the standard library. Commented-out
(QUARANTINED) features are invisible to the AST by construction -- exactly the
right behavior, since a commented feature is not in the model.

What it checks:
  1. Containment      : set(R0) subset set(R1) subset set(R2)   [the core invariant]
  2. No duplicates    : no feature appears twice within a level
  3. Non-empty adds   : each rung adds at least one feature over the previous
  4. Cross-file agree : shared sub-blocks (R0_FEATURES, R1_WMATRIX, ...) are
                        identical across the files that define them

Run as:
    pytest test_ladder_invariants.py -v
    python test_ladder_invariants.py          # standalone report, non-zero exit on failure
"""

import argparse
import ast
import sys
from pathlib import Path

# The three training scripts and the name of the FINAL feature list each uses
# to build its design matrix X.
LADDER = [
    ("R0", "train_r0_baseline.py", "R0_FEATURES"),
    ("R1", "train_r1_hydrology.py", "R1_FEATURES"),
    ("R2", "train_r2_temporal.py", "R2_FEATURES"),
]

# Sub-blocks that are copy-defined in more than one file and must not drift.
# (level label -> set of files expected to contain an identical copy)
SHARED_BLOCKS = ["R0_FEATURES", "R1_HYDRO", "R1_WMATRIX"]

# Default search dir: same folder as this test, override with --dir.
DEFAULT_DIR = Path(__file__).parent


# ──────────────────────────────────────────────────────────────────────────
# AST extraction -- resolve list literals and "+"-concatenation of names
# ──────────────────────────────────────────────────────────────────────────

def _eval_node(node, env):
    """Resolve an AST node to a list[str], or None if not statically resolvable."""
    if isinstance(node, ast.List):
        out = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            else:
                return None  # non-string element: not a plain feature list
        return out
    if isinstance(node, ast.Name):
        return list(env[node.id]) if node.id in env else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        if left is None or right is None:
            return None
        return left + right
    return None


def extract_feature_lists(path: Path) -> dict[str, list[str]]:
    """Return {name: [features]} for every top-level list-of-strings assignment,
    resolving names defined earlier in the same file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    env: dict[str, list[str]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            resolved = _eval_node(stmt.value, env)
            if resolved is not None:
                env[stmt.targets[0].id] = resolved
    return env


# ──────────────────────────────────────────────────────────────────────────
# Loading the ladder from disk
# ──────────────────────────────────────────────────────────────────────────

def load_ladder(base: Path):
    """Return (levels, per_file_env) where levels = {label: final_feature_list}."""
    levels = {}
    per_file = {}
    for label, fname, final_name in LADDER:
        path = base / fname
        if not path.exists():
            raise FileNotFoundError(f"{fname} not found in {base}")
        env = extract_feature_lists(path)
        per_file[fname] = env
        if final_name not in env:
            raise AssertionError(
                f"{fname}: could not statically resolve {final_name}. "
                f"It may use a construct (comprehension, function call) the "
                f"extractor does not evaluate. Resolvable lists: {sorted(env)}"
            )
        levels[label] = env[final_name]
    return levels, per_file


# ──────────────────────────────────────────────────────────────────────────
# Checks (each returns a list of failure strings; empty = pass)
# ──────────────────────────────────────────────────────────────────────────

def check_containment(levels) -> list[str]:
    fails = []
    order = ["R0", "R1", "R2"]
    for lo, hi in zip(order, order[1:]):
        missing = set(levels[lo]) - set(levels[hi])
        if missing:
            fails.append(
                f"{hi} is not a superset of {lo}. {hi} is MISSING "
                f"{len(missing)} feature(s) that {lo} uses: {sorted(missing)}"
            )
    return fails


def check_no_duplicates(levels) -> list[str]:
    fails = []
    for label, feats in levels.items():
        seen, dups = set(), set()
        for f in feats:
            (dups if f in seen else seen).add(f)
        if dups:
            fails.append(f"{label} lists duplicate feature(s): {sorted(dups)}")
    return fails


def check_nonempty_additions(levels) -> list[str]:
    fails = []
    for lo, hi in [("R0", "R1"), ("R1", "R2")]:
        added = set(levels[hi]) - set(levels[lo])
        if not added:
            fails.append(f"{hi} adds no features over {lo} (rung is a no-op).")
    return fails


def check_shared_blocks(per_file) -> list[str]:
    """Same-named sub-block must be identical across every file that defines it."""
    fails = []
    for block in SHARED_BLOCKS:
        definers = {fname: env[block] for fname, env in per_file.items() if block in env}
        if len(definers) < 2:
            continue
        ref_name, ref = next(iter(definers.items()))
        for fname, feats in definers.items():
            if feats != ref:
                only_ref = sorted(set(ref) - set(feats))
                only_this = sorted(set(feats) - set(ref))
                fails.append(
                    f"{block} differs between {ref_name} and {fname}: "
                    f"{ref_name}-only={only_ref}, {fname}-only={only_this}"
                )
    return fails


# ──────────────────────────────────────────────────────────────────────────
# pytest entry points
# ──────────────────────────────────────────────────────────────────────────

def _levels():
    levels, _ = load_ladder(DEFAULT_DIR)
    return levels


def _per_file():
    _, per_file = load_ladder(DEFAULT_DIR)
    return per_file


def test_containment():
    fails = check_containment(_levels())
    assert not fails, "\n".join(fails)


def test_no_duplicates():
    fails = check_no_duplicates(_levels())
    assert not fails, "\n".join(fails)


def test_nonempty_additions():
    fails = check_nonempty_additions(_levels())
    assert not fails, "\n".join(fails)


def test_shared_blocks_consistent():
    fails = check_shared_blocks(_per_file())
    assert not fails, "\n".join(fails)


# ──────────────────────────────────────────────────────────────────────────
# Standalone report
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Check representation-ladder feature invariants")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="Directory with the train_r*.py files")
    args = ap.parse_args()
    base = Path(args.dir)

    levels, per_file = load_ladder(base)

    print("Ladder feature counts:")
    for label in ["R0", "R1", "R2"]:
        print(f"  {label}: {len(levels[label])} features")
    print("Additions:")
    print(f"  R0 -> R1: +{len(set(levels['R1']) - set(levels['R0']))}  "
          f"({sorted(set(levels['R1']) - set(levels['R0']))})")
    print(f"  R1 -> R2: +{len(set(levels['R2']) - set(levels['R1']))}  "
          f"({sorted(set(levels['R2']) - set(levels['R1']))})")
    print()

    all_fails = []
    for name, fn, arg in [
        ("containment (R0 subset R1 subset R2)", check_containment, levels),
        ("no duplicate features", check_no_duplicates, levels),
        ("non-empty additions", check_nonempty_additions, levels),
        ("shared sub-blocks identical across files", check_shared_blocks, per_file),
    ]:
        fails = fn(arg)
        status = "PASS" if not fails else "FAIL"
        print(f"[{status}] {name}")
        for f in fails:
            print(f"        - {f}")
        all_fails += fails

    print()
    if all_fails:
        print(f"RESULT: FAIL ({len(all_fails)} violation(s))")
        sys.exit(1)
    print("RESULT: PASS -- the ladder satisfies its own definition.")


if __name__ == "__main__":
    main()
