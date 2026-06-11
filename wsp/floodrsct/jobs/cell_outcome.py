"""
cell_outcome.py -- runnability gate + outcome record for variance-stack cells.

Every attempted (scenario, target) cell produces a CellOutcome -- no silent
drops. The readiness denominator is reconstructable from the JSON alone.

Fixes two structural bugs in run_variance_stack.py:
  1. Silent drop: process_cell() returned None, main() filtered with
     `if result:`. Cells that failed data-readiness vanished from the JSON.
  2. Floor below fit line: max_bins=min(255,n_train) produces max_bins=1
     when the smallest spatial fold starves. Fixed max_bins + derived fold
     floor from min_samples_leaf.

Design constraints:
  * None-not-zero: abstained cells carry None metrics, never 0.0.
  * Fixed max_bins: constant across all cells for comparability.
  * Derived fold floor: 2*min_samples_leaf, not a magic literal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

def split_floor(min_samples_leaf: int = 20) -> int:
    """Smallest training fold where HGB can make at least one split.

    A split yields two children, each >= min_samples_leaf rows. Below this
    threshold every tree is a single leaf (the training mean) and the
    ensemble is a constant predictor.
    """
    return 2 * int(min_samples_leaf)


# Fixed binning resolution for HistGBDT. Never min(255, n_train) -- that
# makes the estimator's resolution a function of cell size, creating a
# comparability confound across cells.
MAX_BINS = 32


# ---------------------------------------------------------------------------
# Status enum (single canonical source -- no duplicates elsewhere)
# ---------------------------------------------------------------------------

class CellStatus(str, Enum):
    RAN = "RAN"
    ABSTAIN_ZERO_TARGETS = "ABSTAIN_ZERO_TARGETS"
    ABSTAIN_CONSTANT_TARGET = "ABSTAIN_CONSTANT_TARGET"
    ABSTAIN_LOW_PREVALENCE = "ABSTAIN_LOW_PREVALENCE"
    ABSTAIN_INSUFFICIENT_N = "ABSTAIN_INSUFFICIENT_N"
    ABSTAIN_TOO_FEW_GROUPS = "ABSTAIN_TOO_FEW_GROUPS"


ABSTAIN_STATUSES = frozenset(s for s in CellStatus if s is not CellStatus.RAN)


# ---------------------------------------------------------------------------
# Outcome record
# ---------------------------------------------------------------------------

@dataclass
class CellOutcome:
    """One record per attempted cell. Serialize ALL of these.

    Abstained cells have metrics=None (never 0.0, never NaN).
    RAN cells have metrics populated with ladder results.
    """
    scenario: str
    target: str
    status: CellStatus
    reason: str = ""
    n_total: Optional[int] = None
    n_nonnull_target: Optional[int] = None
    n_groups: Optional[int] = None
    min_fold_train_n: Optional[int] = None
    prevalence: Optional[float] = None
    target_std: Optional[float] = None
    metrics: Optional[dict] = None
    stump_risk: bool = False

    @property
    def abstained(self) -> bool:
        return self.status in ABSTAIN_STATUSES

    def to_record(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# Post-hoc audit
# ---------------------------------------------------------------------------

def audit_fold_sizes(
    outcomes: list[CellOutcome],
    min_samples_leaf: int = 20,
) -> list[CellOutcome]:
    """Flag RAN cells whose smallest fold cannot support a single split.

    These cells produced a constant predictor; their ladder metrics are
    not measurements. They must be reclassified, not merely disclosed.
    """
    floor = split_floor(min_samples_leaf)
    flagged = []
    for o in outcomes:
        if o.status is CellStatus.RAN and o.min_fold_train_n is not None:
            if o.min_fold_train_n < floor:
                o.stump_risk = True
                flagged.append(o)
    return flagged


def readiness_summary(outcomes: list[CellOutcome]) -> dict:
    """Reconstruct the data-readiness denominator the paper claims.

    ran_valid is the number section 7.1 should be built on -- not ran,
    and not the raw attempted count.
    """
    total = len(outcomes)
    ran = [o for o in outcomes if o.status is CellStatus.RAN]
    stumps = [o for o in ran if o.stump_risk]
    by_reason: dict[str, int] = {}
    for o in outcomes:
        if o.abstained:
            by_reason[o.status.value] = by_reason.get(o.status.value, 0) + 1
    return {
        "attempted": total,
        "ran": len(ran),
        "ran_valid": len(ran) - len(stumps),
        "ran_stump_risk": len(stumps),
        "abstained": total - len(ran),
        "abstain_breakdown": by_reason,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    assert split_floor(20) == 40
    assert split_floor(5) == 10

    outs = [
        CellOutcome("Houston", "NFIP", CellStatus.RAN, n_total=396,
                    n_nonnull_target=396, min_fold_train_n=300,
                    metrics={"delta_rmse": -0.27}),
        CellOutcome("SWFL", "NFIP", CellStatus.ABSTAIN_INSUFFICIENT_N,
                    reason="min fold 28 < floor 40", n_total=606,
                    n_nonnull_target=140, min_fold_train_n=28),
        CellOutcome("Riverside", "311", CellStatus.ABSTAIN_ZERO_TARGETS,
                    reason="zero non-null target", n_total=172,
                    n_nonnull_target=0),
        CellOutcome("Riverside", "NFIP", CellStatus.RAN, n_total=172,
                    n_nonnull_target=170, min_fold_train_n=34,
                    metrics={"delta_rmse": 0.01}),
    ]

    flagged = audit_fold_sizes(outs)
    assert len(flagged) == 1 and flagged[0].scenario == "Riverside"
    assert flagged[0].stump_risk is True

    s = readiness_summary(outs)
    assert s == {
        "attempted": 4,
        "ran": 2,
        "ran_valid": 1,
        "ran_stump_risk": 1,
        "abstained": 2,
        "abstain_breakdown": {
            "ABSTAIN_INSUFFICIENT_N": 1,
            "ABSTAIN_ZERO_TARGETS": 1,
        },
    }, s

    assert all(o.metrics is None for o in outs if o.abstained)

    print("all self-tests passed")


if __name__ == "__main__":
    _selftest()
