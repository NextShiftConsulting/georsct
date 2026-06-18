"""
stress_geocert.py - GeoCert Evaluation-Failure Taxonomy Stress Testing

S035-MAST-Reconcile artifact.

Purpose
-------
Operationalize the GeoCert taxonomy as typed failure specs, synthetic failure
injection, and diagnostic scoring. This intentionally reuses the prior
stress_mast.py pattern (taxonomy -> signal signature -> diagnosis -> stress
suite) while replacing MAST execution-trace failure labels with GeoCert
evaluation-failure labels.

This module is standalone and has no YRSN runtime dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple
import math
import random

Range = Tuple[float, float]


class GeoCertCategory(str, Enum):
    """Three-stage GeoCert failure taxonomy."""

    GC1_LABEL_CONSTRUCTION = "GC1"
    GC2_DECOMPOSITION_REDUCTION = "GC2"
    GC3_DEPLOYMENT_TRANSLATION = "GC3"


class GeoCertFailureMode(str, Enum):
    """Nine fine-grained GeoCert evaluation-failure modes."""

    # GC1: label construction failures
    GCF_1_1_TERCILE_UNIFORMITY = "GCF-1.1"
    GCF_1_2_LABEL_SOLVER_COUPLING = "GCF-1.2"
    GCF_1_3_TARGET_DIFFICULTY_CONFLATION = "GCF-1.3"

    # GC2: decomposition reduction failures
    GCF_2_1_SCALAR_PROJECTION = "GCF-2.1"
    GCF_2_2_GATE_COMPRESSION = "GCF-2.2"
    GCF_2_3_RANGE_COMPRESSION = "GCF-2.3"

    # GC3: deployment translation failures
    GCF_3_1_TARGET_SOLVER_CONFLATION = "GCF-3.1"
    GCF_3_2_PROXY_CALIBRATION_DRIFT = "GCF-3.2"
    GCF_3_3_FINE_ROUTING_FAILURE = "GCF-3.3"


@dataclass(frozen=True)
class GeoCertFailureSpec:
    mode: GeoCertFailureMode
    category: GeoCertCategory
    name: str
    description: str
    expected_signal_pattern: Dict[str, Range]
    diagnostic_tests: List[str]
    example_scenario: str
    paper_use: str


@dataclass
class GeoCertDiagnosisCandidate:
    mode: str
    name: str
    category: str
    confidence: float


@dataclass
class GeoCertDiagnosis:
    top1_mode: str
    candidates: List[GeoCertDiagnosisCandidate]
    input_signals: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top1_mode": self.top1_mode,
            "candidates": [asdict(c) for c in self.candidates],
            "input_signals": self.input_signals,
        }


GEOCERT_FAILURE_SPECS: Dict[GeoCertFailureMode, GeoCertFailureSpec] = {
    GeoCertFailureMode.GCF_1_1_TERCILE_UNIFORMITY: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_1_1_TERCILE_UNIFORMITY,
        category=GeoCertCategory.GC1_LABEL_CONSTRUCTION,
        name="Tercile Uniformity",
        description="Residual-tercile labels collapse toward uniform or bland class assignments, making noisy controls look serious.",
        expected_signal_pattern={
            "label_entropy": (0.85, 1.00),
            "prediction_entropy": (0.80, 1.00),
            "delta_structure": (0.00, 0.25),
            "noisy_control_similarity": (0.70, 1.00),
            "aggregate_separation": (0.00, 0.35),
        },
        diagnostic_tests=[
            "label entropy by target/solver",
            "prediction entropy by solver",
            "distance from noisy control to serious solvers",
            "ablation-delta structure score",
        ],
        example_scenario="noisy_solver nearest neighbor is svr_rbf under aggregate certificate metrics.",
        paper_use="Explains why R/S/N should not be treated as ordinary supervised classes.",
    ),
    GeoCertFailureMode.GCF_1_2_LABEL_SOLVER_COUPLING: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_1_2_LABEL_SOLVER_COUPLING,
        category=GeoCertCategory.GC1_LABEL_CONSTRUCTION,
        name="Label-Solver Coupling",
        description="The label construction encodes the behavior of a particular solver rather than solver-independent representation evidence.",
        expected_signal_pattern={
            "solver_label_dependence": (0.65, 1.00),
            "cross_solver_label_agreement": (0.00, 0.45),
            "delta_structure": (0.35, 0.80),
            "target_variance": (0.30, 0.75),
            "aggregate_separation": (0.25, 0.65),
        },
        diagnostic_tests=[
            "label agreement across solvers",
            "solver-stratified label frequency",
            "delta-sign consistency across solver families",
        ],
        example_scenario="R/S/N assignments change primarily when the solver changes, not when representation evidence changes.",
        paper_use="Motivates label-free Y-U over supervised label heads.",
    ),
    GeoCertFailureMode.GCF_1_3_TARGET_DIFFICULTY_CONFLATION: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_1_3_TARGET_DIFFICULTY_CONFLATION,
        category=GeoCertCategory.GC1_LABEL_CONSTRUCTION,
        name="Target-Difficulty Conflation",
        description="Hard targets receive worse labels because the target is difficult, not because the representation is noisy.",
        expected_signal_pattern={
            "trf_correlation": (0.60, 1.00),
            "target_variance": (0.55, 1.00),
            "cross_solver_label_agreement": (0.50, 1.00),
            "solver_label_dependence": (0.00, 0.45),
            "delta_structure": (0.25, 0.65),
        },
        diagnostic_tests=[
            "label vs N-ceiling correlation",
            "target-family stratification",
            "within-target solver contrast",
        ],
        example_scenario="high-N-ceiling tasks are marked noisy even when all solvers agree the target is intrinsically hard.",
        paper_use="Separates task recoverability from representation quality.",
    ),
    GeoCertFailureMode.GCF_2_1_SCALAR_PROJECTION: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_2_1_SCALAR_PROJECTION,
        category=GeoCertCategory.GC2_DECOMPOSITION_REDUCTION,
        name="Scalar Projection",
        description="A single certificate projection such as alpha misranks solver quality by hiding other axes such as sigma.",
        expected_signal_pattern={
            "alpha_rank_gap": (0.60, 1.00),
            "sigma_outlier": (0.60, 1.00),
            "kappa_compression": (0.20, 0.65),
            "gate_entropy": (0.00, 0.35),
            "aggregate_separation": (0.35, 0.85),
        },
        diagnostic_tests=[
            "alpha rank vs actual performance rank",
            "alpha-sigma scatter",
            "control outlier analysis",
        ],
        example_scenario="mean_baseline has highest alpha but high sigma, so alpha-alone ranking is misleading.",
        paper_use="Direct support for 'compatibility is not scalar'.",
    ),
    GeoCertFailureMode.GCF_2_2_GATE_COMPRESSION: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_2_2_GATE_COMPRESSION,
        category=GeoCertCategory.GC2_DECOMPOSITION_REDUCTION,
        name="Gate Compression",
        description="A multi-axis certificate is reduced to one final gate decision, erasing meaningful metric separation.",
        expected_signal_pattern={
            "all_execute_rate": (0.85, 1.00),
            "gate_entropy": (0.00, 0.20),
            "metric_range": (0.35, 1.00),
            "kappa_compression": (0.30, 0.80),
            "false_execute_risk": (0.45, 1.00),
        },
        diagnostic_tests=[
            "gate decision entropy",
            "metric range conditional on same gate",
            "threshold sweep",
        ],
        example_scenario="all S018D solvers EXECUTE despite visible alpha/N/sigma differences.",
        paper_use="Shows typed decisions can become scalar-like when thresholds erase structure.",
    ),
    GeoCertFailureMode.GCF_2_3_RANGE_COMPRESSION: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_2_3_RANGE_COMPRESSION,
        category=GeoCertCategory.GC2_DECOMPOSITION_REDUCTION,
        name="Range Compression",
        description="Serious solvers occupy a narrow certificate band, limiting fine-grained discrimination.",
        expected_signal_pattern={
            "serious_solver_range": (0.00, 0.30),
            "native_tightness": (0.65, 1.00),
            "kappa_compression": (0.55, 1.00),
            "aggregate_separation": (0.15, 0.55),
            "fine_routing_accuracy": (0.00, 0.45),
        },
        diagnostic_tests=[
            "serious-only metric ranges",
            "native solver clustering",
            "within-family vs between-family distance",
        ],
        example_scenario="pca_v1, spatial_lag_v1, and gnn_v2 cluster tighter than the broad serious-solver panel.",
        paper_use="Explains why coarse separation can coexist with failed fine routing.",
    ),
    GeoCertFailureMode.GCF_3_1_TARGET_SOLVER_CONFLATION: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_3_1_TARGET_SOLVER_CONFLATION,
        category=GeoCertCategory.GC3_DEPLOYMENT_TRANSLATION,
        name="Target-Solver Conflation",
        description="The evaluation confuses hard-for-this-target with incompatible-for-this-solver.",
        expected_signal_pattern={
            "alpha_profile_corr": (0.00, 0.35),
            "target_variance": (0.50, 1.00),
            "solver_specificity": (0.50, 1.00),
            "trf_correlation": (0.35, 0.85),
            "fine_routing_accuracy": (0.00, 0.45),
        },
        diagnostic_tests=[
            "pairwise alpha-profile correlation",
            "target-family interaction test",
            "solver-specific delta profiles",
        ],
        example_scenario="mean pairwise alpha-profile correlation is low, so what is hard for one solver is not hard for another.",
        paper_use="Defines why target averaging is dangerous.",
    ),
    GeoCertFailureMode.GCF_3_2_PROXY_CALIBRATION_DRIFT: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_3_2_PROXY_CALIBRATION_DRIFT,
        category=GeoCertCategory.GC3_DEPLOYMENT_TRANSLATION,
        name="Proxy Calibration Drift",
        description="Proxy kappa or proxy certificate metrics decouple from true solver success or expected control behavior.",
        expected_signal_pattern={
            "kappa_compression": (0.55, 1.00),
            "proxy_success_corr": (0.00, 0.35),
            "noisy_control_similarity": (0.55, 1.00),
            "false_execute_risk": (0.45, 1.00),
            "gate_entropy": (0.00, 0.40),
        },
        diagnostic_tests=[
            "kappa vs solver success calibration",
            "proxy scale range audit",
            "control-nearest-neighbor audit",
        ],
        example_scenario="kappa is compressed and noisy_solver is not exposed as noisy by aggregate metrics.",
        paper_use="Turns proxy limitations into an explicit evaluation target.",
    ),
    GeoCertFailureMode.GCF_3_3_FINE_ROUTING_FAILURE: GeoCertFailureSpec(
        mode=GeoCertFailureMode.GCF_3_3_FINE_ROUTING_FAILURE,
        category=GeoCertCategory.GC3_DEPLOYMENT_TRANSLATION,
        name="Fine-Routing Failure",
        description="Certificate evidence supports coarse diagnosis but cannot select among serious solvers or route per sample.",
        expected_signal_pattern={
            "fine_routing_accuracy": (0.00, 0.40),
            "serious_solver_range": (0.00, 0.35),
            "solver_specificity": (0.25, 0.75),
            "native_tightness": (0.55, 1.00),
            "gate_entropy": (0.00, 0.40),
        },
        diagnostic_tests=[
            "oracle vs certificate routing gap",
            "serious-only clustering",
            "per-sample routing accuracy",
        ],
        example_scenario="certificate separates controls but not serious solvers well enough for routing.",
        paper_use="Prevents overclaiming that GeoCert solves routing.",
    ),
}


def _center(range_tuple: Range) -> float:
    return (range_tuple[0] + range_tuple[1]) / 2.0


def _score_range(value: float, range_tuple: Range) -> float:
    low, high = range_tuple
    if low <= value <= high:
        return 1.0
    width = max(high - low, 1e-6)
    if value < low:
        return max(0.0, 1.0 - (low - value) / width)
    return max(0.0, 1.0 - (value - high) / width)


def score_geocert_match(signals: Mapping[str, float], spec: GeoCertFailureSpec) -> float:
    """Score how well a signal record matches a failure spec."""
    scores: List[float] = []
    for key, rng in spec.expected_signal_pattern.items():
        if key in signals and signals[key] is not None:
            scores.append(_score_range(float(signals[key]), rng))
    return sum(scores) / len(scores) if scores else 0.0


def _all_signal_keys() -> List[str]:
    """Collect every signal key across all failure specs."""
    keys: set = set()
    for spec in GEOCERT_FAILURE_SPECS.values():
        keys.update(spec.expected_signal_pattern.keys())
    return sorted(keys)


def _anti_value(key: str, target_mode: GeoCertFailureMode, rng: random.Random) -> float:
    """Pick a value for *key* that falls outside as many competing specs as possible."""
    target_spec = GEOCERT_FAILURE_SPECS[target_mode]
    # Collect ranges from all OTHER modes that use this signal
    other_ranges = []
    for m, spec in GEOCERT_FAILURE_SPECS.items():
        if m == target_mode:
            continue
        if key in spec.expected_signal_pattern:
            other_ranges.append(spec.expected_signal_pattern[key])
    if not other_ranges:
        # No competing mode uses this signal -- neutral mid-range value
        return 0.5 + (rng.random() - 0.5) * 0.1
    # Find the value that is furthest outside the union of competing ranges
    union_lo = min(lo for lo, _ in other_ranges)
    union_hi = max(hi for _, hi in other_ranges)
    # Place the value below or above the union, whichever is farther from center
    if union_lo > 0.3:
        # Room below
        return max(0.0, union_lo - 0.15 + rng.random() * 0.05)
    elif union_hi < 0.7:
        # Room above
        return min(1.0, union_hi + 0.15 + rng.random() * 0.05)
    else:
        # Ranges span most of [0,1] -- pick the extreme that has more room
        return 0.0 + rng.random() * 0.05 if union_lo < 0.5 else 0.95 + rng.random() * 0.05


def inject_geocert_failure(
    mode: str | GeoCertFailureMode,
    intensity: float = 0.85,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate synthetic signals matching a GeoCert failure mode.

    Emits ALL 22 signals: the 5 matching signals are set to in-range values,
    and the remaining 17 are set to anti-pattern values that push competing
    modes' scores down.
    """
    rng = random.Random(seed)
    if not isinstance(mode, GeoCertFailureMode):
        mode = _resolve_mode(str(mode))
    spec = GEOCERT_FAILURE_SPECS[mode]
    signals: Dict[str, float] = {}

    # Set matching signals to in-range values
    for key, (low, high) in spec.expected_signal_pattern.items():
        c = _center((low, high))
        jitter = (rng.random() - 0.5) * 0.05 * (high - low)
        signals[key] = min(1.0, max(0.0, c + jitter * intensity))

    # Set non-matching signals to anti-pattern values
    for key in _all_signal_keys():
        if key not in signals:
            signals[key] = _anti_value(key, mode, rng)

    return {
        "injected_mode": mode.value,
        "name": spec.name,
        "category": spec.category.value,
        "signals": signals,
        "intensity": intensity,
    }


def diagnose_geocert_failure(signals: Mapping[str, float], top_k: int = 3) -> Dict[str, Any]:
    """Return typed GeoCert failure candidates with confidence scores."""
    scored: List[GeoCertDiagnosisCandidate] = []
    for mode, spec in GEOCERT_FAILURE_SPECS.items():
        confidence = score_geocert_match(signals, spec)
        scored.append(
            GeoCertDiagnosisCandidate(
                mode=mode.value,
                name=spec.name,
                category=spec.category.value,
                confidence=round(confidence, 4),
            )
        )
    scored.sort(key=lambda c: c.confidence, reverse=True)
    diagnosis = GeoCertDiagnosis(
        top1_mode=scored[0].mode,
        candidates=scored[:top_k],
        input_signals={k: float(v) for k, v in signals.items()},
    )
    return diagnosis.to_dict()


def run_geocert_stress_suite(intensity: float = 0.85, seed: int = 3500) -> Dict[str, Any]:
    """Run injection-detection validation over all nine GeoCert modes."""
    results: List[Dict[str, Any]] = []
    top1 = 0
    top3 = 0
    for i, mode in enumerate(GeoCertFailureMode):
        injected = inject_geocert_failure(mode, intensity=intensity, seed=seed + i)
        diagnosis = diagnose_geocert_failure(injected["signals"], top_k=3)
        candidates = [c["mode"] for c in diagnosis["candidates"]]
        result = {
            "injected_mode": mode.value,
            "injected_name": GEOCERT_FAILURE_SPECS[mode].name,
            "top1_mode": diagnosis["top1_mode"],
            "top3_modes": candidates,
            "top1_match": diagnosis["top1_mode"] == mode.value,
            "top3_match": mode.value in candidates,
            "signals": injected["signals"],
            "candidates": diagnosis["candidates"],
        }
        top1 += int(result["top1_match"])
        top3 += int(result["top3_match"])
        results.append(result)
    return {
        "total_modes": len(results),
        "top1_correct": top1,
        "top3_correct": top3,
        "top1_accuracy": round(top1 / len(results), 4),
        "top3_accuracy": round(top3 / len(results), 4),
        "results": results,
    }


def get_geocert_taxonomy() -> Dict[str, Any]:
    categories: Dict[str, Any] = {
        cat.value: {"name": cat.name, "modes": []}
        for cat in GeoCertCategory
    }
    for mode, spec in GEOCERT_FAILURE_SPECS.items():
        categories[spec.category.value]["modes"].append({
            "id": mode.value,
            "name": spec.name,
            "description": spec.description,
            "expected_signal_pattern": spec.expected_signal_pattern,
            "diagnostic_tests": spec.diagnostic_tests,
            "example_scenario": spec.example_scenario,
            "paper_use": spec.paper_use,
        })
    return {"version": "s035", "categories": categories}


def _resolve_mode(value: str) -> GeoCertFailureMode:
    value = value.strip()
    for mode in GeoCertFailureMode:
        if value == mode.value or value == mode.name:
            return mode
    value_upper = value.upper().replace("-", "_").replace(" ", "_")
    for mode in GeoCertFailureMode:
        if value_upper in mode.name:
            return mode
    raise ValueError(f"Unknown GeoCert failure mode: {value}")


__all__ = [
    "GeoCertCategory",
    "GeoCertFailureMode",
    "GeoCertFailureSpec",
    "GEOCERT_FAILURE_SPECS",
    "inject_geocert_failure",
    "diagnose_geocert_failure",
    "run_geocert_stress_suite",
    "get_geocert_taxonomy",
]
