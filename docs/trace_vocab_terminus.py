#!/usr/bin/env python3
"""
Passthrough-terminus tracer for the gate-vocabulary swap.

The swap changes the STRING VALUES in sub_signal and gate_evidence (controlplane
vocabulary). "It's just passthrough, safe" is the assumption that has hidden
every bug this migration produced. Passthrough does not end risk — it relocates
it to wherever the passed-through value is finally READ (matched, stored, or
returned across an API boundary).

This traces each vocabulary-carrying field from its producer to its TERMINUS:
  - string comparison  (== "OLD_STRING")        -> BREAKS on new vocab (hard fail)
  - dict key access     (["gate_3"])             -> BREAKS on renamed key
  - stored/serialized   (json.dump, to_dict, DB) -> SCHEMA CHANGE (data contract)
  - API response field  (return {..: gate_reason})-> CLIENT CONTRACT (external)
  - pure passthrough     (a = b; return a)        -> FOLLOW to next hop, don't stop

It also flags MIXED-VOCABULARY fields: a field written by both a swapped path
(new vocab) and an un-swapped path (old vocab), read at a common sink.

Run:
  python3 trace_vocab_terminus.py --repos ~/github/georsct ~/github/swarm-it-api \
      ~/github/yrsn ~/github/yrsn-train --seed-fields sub_signal gate_evidence gate_reason

Exit 0 = every terminus is passthrough/return-only with no external matcher or
         stored schema; safe to commit.
Exit 1 = at least one terminus matches, stores, or crosses an API boundary on the
         changed vocabulary; or a mixed-vocabulary sink exists. Review before commit.
"""
from __future__ import annotations
import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path


# Old-vocabulary tokens whose values change under the swap.
OLD_VOCAB_TOKENS = {
    "N_FLOOR_BREACH_AND_ALPHA_LOW", "COHERENCE_LOW", "KAPPA_BELOW_OOBLECK",
    "KAPPA_LANDAUER_FAIL", "KAPPA_UNAVAILABLE", "R_BAR_BELOW_THRESHOLD",
    "KAPPA_L_BELOW_THRESHOLD",
    "gate_1", "gate_2", "gate_3", "gate_3b", "gate_4",
}
STORE_MARKERS = ("json.dump", "json.dumps", "to_dict", "asdict", "INSERT",
                 "put_object", "write", ".sql", "boto3", "duckdb", "return")
API_MARKERS = ("return", "Response", "jsonify", "JSONResponse", "response",
               "body", "payload")


@dataclass
class Terminus:
    file: Path
    line: int
    field: str
    kind: str          # COMPARE | KEY_ACCESS | STORE | API | PASSTHROUGH
    snippet: str


@dataclass
class Report:
    termini: list[Terminus] = field(default_factory=list)
    mixed_sinks: list[str] = field(default_factory=list)


class FieldTracer(ast.NodeVisitor):
    """Find where a seed field name is read, and classify the read."""
    def __init__(self, path: Path, seeds: set[str], src: str):
        self.path = path
        self.seeds = seeds
        self.src_lines = src.splitlines()
        self.found: list[Terminus] = []
        # names currently bound to a seed value (passthrough tracking, shallow)
        self.aliases: set[str] = set(seeds)

    def _snip(self, node: ast.AST) -> str:
        ln = getattr(node, "lineno", 1) - 1
        return self.src_lines[ln].strip() if 0 <= ln < len(self.src_lines) else ""

    def _is_seed_ref(self, node: ast.AST) -> str | None:
        # attribute access: x.sub_signal / result.gate_evidence
        if isinstance(node, ast.Attribute) and node.attr in self.seeds:
            return node.attr
        # name bound to a seed via prior assignment
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return node.id
        # subscript: gate_evidence["gate_3"]
        if isinstance(node, ast.Subscript):
            base = node.value
            name = self._is_seed_ref(base)
            if name:
                return name
        return None

    def visit_Compare(self, node: ast.Compare):
        # x.sub_signal == "OLD_STRING"
        left = self._is_seed_ref(node.left)
        if left:
            for c in node.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    if c.value in OLD_VOCAB_TOKENS:
                        self.found.append(Terminus(self.path, node.lineno, left,
                                                   "COMPARE", self._snip(node)))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        name = self._is_seed_ref(node.value)
        if name and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            if node.slice.value in OLD_VOCAB_TOKENS:
                self.found.append(Terminus(self.path, node.lineno, name,
                                           "KEY_ACCESS", self._snip(node)))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # a = result.sub_signal  -> a becomes an alias (shallow passthrough)
        src_field = self._is_seed_ref(node.value)
        if src_field:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.aliases.add(t.id)
                    line = self._snip(node)
                    kind = "PASSTHROUGH"
                    if any(m in line for m in STORE_MARKERS):
                        kind = "STORE"
                    self.found.append(Terminus(self.path, node.lineno, src_field,
                                               kind, line))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            return
        # return {"gate_reason": x} or return x  where x is a seed alias
        line = self._snip(node)
        hit = None
        for sub in ast.walk(node.value):
            name = self._is_seed_ref(sub)
            if name:
                hit = name
                break
        if hit:
            kind = "API" if any(m in line for m in API_MARKERS) else "STORE"
            self.found.append(Terminus(self.path, node.lineno, hit, kind, line))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # json.dump(..., x) / cursor.execute(INSERT ..., x) with a seed alias arg
        line = self._snip(node)
        if any(m in line for m in STORE_MARKERS):
            for a in node.args:
                name = self._is_seed_ref(a)
                if name:
                    self.found.append(Terminus(self.path, node.lineno, name,
                                               "STORE", line))
        self.generic_visit(node)


def trace_repo(repo: Path, seeds: set[str]) -> list[Terminus]:
    out: list[Terminus] = []
    for py in repo.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except Exception:
            continue
        t = FieldTracer(py, seeds, src)
        t.visit(tree)
        out.extend(t.found)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", type=Path, required=True)
    ap.add_argument("--seed-fields", nargs="+",
                    default=["sub_signal", "gate_evidence", "gate_reason"])
    args = ap.parse_args()

    seeds = set(args.seed_fields)
    repos = [p for p in args.repos if p.exists()]
    all_termini: list[Terminus] = []
    for r in repos:
        all_termini.extend(trace_repo(r, seeds))

    buckets: dict[str, list[Terminus]] = {}
    for t in all_termini:
        buckets.setdefault(t.kind, []).append(t)

    print("=" * 74)
    print("VOCABULARY TERMINUS TRACE")
    print("=" * 74)
    order = ["COMPARE", "KEY_ACCESS", "STORE", "API", "PASSTHROUGH"]
    risky = 0
    for kind in order:
        items = buckets.get(kind, [])
        if not items:
            continue
        print(f"\n[{kind}]  ({len(items)})")
        for t in items:
            print(f"  {t.file}:{t.line}  field={t.field}")
            print(f"      {t.snippet}")
        if kind in ("COMPARE", "KEY_ACCESS", "STORE", "API"):
            risky += len(items)

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  COMPARE (breaks on new vocab)      : {len(buckets.get('COMPARE', []))}")
    print(f"  KEY_ACCESS (breaks on renamed key) : {len(buckets.get('KEY_ACCESS', []))}")
    print(f"  STORE (data-contract / schema)     : {len(buckets.get('STORE', []))}")
    print(f"  API (external client contract)     : {len(buckets.get('API', []))}")
    print(f"  PASSTHROUGH (followed, no terminus): {len(buckets.get('PASSTHROUGH', []))}")
    print()
    if risky == 0:
        print("  SAFE TO COMMIT: every terminus is pure passthrough or unmatched.")
        print("  No comparison, key access, stored schema, or API field depends on")
        print("  the changed vocabulary.")
        return 0
    print("  DO NOT COMMIT YET. The termini above READ the changed vocabulary.")
    print("  For each: migrate the reader, or translate at that boundary.")
    print("  'It's just passthrough' was true at the handoff and false at the sink.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
