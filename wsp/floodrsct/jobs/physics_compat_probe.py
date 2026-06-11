#!/usr/bin/env python3
"""
physics_compat_probe.py

Pre-registration probe: does realized per-geometry kappa_compat recover the
physics-compatibility ordering we declared in advance?

This is an INSTRUMENT-VALIDATION probe, not a kappa estimator. It does NOT
compute kappa. The canonical estimator lives behind compute_kappa_compat and
REQUIRED_KAPPA_COMPONENTS, sourced from EXPERIMENT_CONTRACT.yaml. This module
only consumes already-computed kappa_compat values and tests their ordering.

Discipline enforced here:
  * None-not-zero: a missing kappa_compat is NOT 0.0. It cannot be scored and
    is excluded from concordance; a geometry with None routes to ESCALATE
    semantics upstream, never SUPPRESS.
  * Degenerate-pin guard: if every present kappa_compat equals 0.0 (the current
    live bug) the probe RAISES. A degenerate router cannot order anything; a
    passing concordance under a 0.0 pin would be a false positive.
  * Minimum pairable-n floor + bootstrap CI on the concordance statistic.
    (Same defect class flagged in the R4 alpha review: no n-floor, no CI.)
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence
import math
import random

# Locked geometry vocabulary. Order here is not the prediction; ranks are below.
GEOMETRIES = (
    "smooth_metric",
    "spatial_dependence",
    "composite",
    "periodic",
    "hierarchical",
    "logical_relational",
)

# PRE-REGISTERED physics-compatibility ordering (higher rank == more compatible
# with a physics-constrained solver). Ties are permitted and handled by the
# tie-aware concordance below. Justification is in the paper; do not edit after
# data lock without flagging confirmatory->exploratory demotion.
PREREG_PHYSICS_RANK: dict[str, int] = {
    "smooth_metric":       6,   # feng2022, song2025: differentiable hydrology on continuous streamflow
    "spatial_dependence":  5,   # bindas2024, taghizadeh2025: river-network routing, mass-conserving GNN
    "composite":           4,   # feng2022: multiphysical outputs; ungauged transfer partially carries
    "periodic":            3,   # uncovered in cited corpus; storm/seasonal climatology plausible
    "hierarchical":        2,   # feng2022, song2025: watershed regionalization as parameter sharing only
    "logical_relational":  1,   # zhang2025: monotonicity is the only relational-ish constraint; weak
}

MIN_PAIRABLE_N = 4          # floor on geometries with a present (non-None) kappa
N_BOOTSTRAP = 2000
CI_ALPHA = 0.05
DEGENERATE_EPS = 1e-12      # |kappa| <= eps treated as the 0.0 pin


class DegenerateKappaError(RuntimeError):
    """All present kappa_compat are pinned at 0.0 -> router is degenerate."""


@dataclass(frozen=True)
class ProbeResult:
    n_pairable: int
    tau: float                      # tie-aware Kendall tau-b, predicted vs realized
    ci_low: float
    ci_high: float
    excluded_none: tuple[str, ...]  # geometries dropped for missing kappa
    verdict: str                    # CONCORDANT | DISCORDANT | INSUFFICIENT_N

    def row(self) -> str:
        """One verdict row, paper-ready."""
        return (
            f"physics_compat | n={self.n_pairable} | "
            f"tau_b={self.tau:+.3f} [{self.ci_low:+.3f},{self.ci_high:+.3f}] | "
            f"dropped_None={','.join(self.excluded_none) or '-'} | "
            f"{self.verdict}"
        )


def _kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Tie-aware Kendall tau-b. Returns nan if undefined (all-tie on a side)."""
    n = len(x)
    conc = disc = tx = ty = 0
    for i, j in combinations(range(n), 2):
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        if dx == 0 and dy == 0:
            tx += 1; ty += 1
        elif dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        else:
            if (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    if denom == 0:
        return float("nan")
    return (conc - disc) / denom


def _pairs(realized: Mapping[str, float | None]) -> tuple[
    list[str], list[float], list[float], list[str]
]:
    """Build aligned (predicted_rank, realized_kappa) over geometries with a
    present kappa. None is excluded, never coerced to 0.0."""
    geoms, pred, real, dropped = [], [], [], []
    for g in GEOMETRIES:
        k = realized.get(g, None)
        if k is None:
            dropped.append(g)            # None-not-zero: exclude, do not zero
            continue
        if g not in PREREG_PHYSICS_RANK:
            raise KeyError(f"unknown geometry {g!r}")
        geoms.append(g)
        pred.append(float(PREREG_PHYSICS_RANK[g]))
        real.append(float(k))
    return geoms, pred, real, dropped


def physics_compat_probe(
    realized_kappa_compat: Mapping[str, float | None],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    rng_seed: int = 0,
) -> ProbeResult:
    geoms, pred, real, dropped = _pairs(realized_kappa_compat)
    n = len(geoms)

    # Degenerate-pin guard BEFORE any scoring: the live kappa=0.0 bug.
    if n > 0 and all(abs(v) <= DEGENERATE_EPS for v in real):
        raise DegenerateKappaError(
            "All present kappa_compat == 0.0 (router degenerate). "
            "Probe cannot order geometries; fix the kappa pin first."
        )

    if n < MIN_PAIRABLE_N:
        return ProbeResult(
            n_pairable=n, tau=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"),
            excluded_none=tuple(dropped), verdict="INSUFFICIENT_N",
        )

    tau = _kendall_tau_b(pred, real)

    # Bootstrap CI over geometries (resample pairs with replacement).
    rng = random.Random(rng_seed)
    taus: list[float] = []
    idx = list(range(n))
    for _ in range(n_bootstrap):
        samp = [rng.choice(idx) for _ in range(n)]
        bp = [pred[i] for i in samp]
        br = [real[i] for i in samp]
        t = _kendall_tau_b(bp, br)
        if not math.isnan(t):
            taus.append(t)
    if taus:
        taus.sort()
        lo = taus[int((CI_ALPHA / 2) * len(taus))]
        hi = taus[int((1 - CI_ALPHA / 2) * len(taus)) - 1]
    else:
        lo = hi = float("nan")

    # Verdict: concordant only if CI lower bound clears 0 (ordering recovered).
    verdict = "CONCORDANT" if (not math.isnan(lo) and lo > 0) else "DISCORDANT"
    return ProbeResult(n, tau, lo, hi, tuple(dropped), verdict)


# --------------------------------------------------------------------------- #
# Self-tests: planted cases. Run: python physics_compat_probe.py
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # (a) The live bug: every present kappa pinned at 0.0 -> must RAISE.
    pinned = {g: 0.0 for g in GEOMETRIES}
    try:
        physics_compat_probe(pinned)
        raise AssertionError("degenerate pin did not raise")
    except DegenerateKappaError:
        pass

    # (b) Perfectly concordant: realized kappa monotone in the prereg rank.
    concordant = {g: PREREG_PHYSICS_RANK[g] / 6.0 for g in GEOMETRIES}
    r = physics_compat_probe(concordant)
    assert r.verdict == "CONCORDANT", r.row()
    assert r.tau > 0.9, r.row()
    assert r.ci_low > 0, r.row()

    # (c) Anti-concordant: realized reversed -> DISCORDANT, tau negative.
    reversed_k = {g: (7 - PREREG_PHYSICS_RANK[g]) / 6.0 for g in GEOMETRIES}
    r = physics_compat_probe(reversed_k)
    assert r.verdict == "DISCORDANT", r.row()
    assert r.tau < 0, r.row()

    # (d) None-not-zero: a missing geometry is dropped, not coerced to 0.0.
    with_none = {g: PREREG_PHYSICS_RANK[g] / 6.0 for g in GEOMETRIES}
    with_none["logical_relational"] = None
    r = physics_compat_probe(with_none)
    assert "logical_relational" in r.excluded_none, r.row()
    assert r.n_pairable == 5, r.row()
    # If None had been coerced to 0.0 it would have *strengthened* concordance
    # (low rank, low value). Excluding it must not fabricate that agreement.
    assert r.verdict in ("CONCORDANT", "DISCORDANT")

    # (e) Insufficient n after None drops -> INSUFFICIENT_N, no crash.
    sparse = {g: None for g in GEOMETRIES}
    sparse["smooth_metric"] = 0.9
    sparse["spatial_dependence"] = 0.7
    r = physics_compat_probe(sparse)
    assert r.verdict == "INSUFFICIENT_N", r.row()

    print("all self-tests passed")
    print("example verdict row:")
    print("  " + physics_compat_probe(concordant).row())


if __name__ == "__main__":
    _selftest()
