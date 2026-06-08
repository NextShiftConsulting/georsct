"""Experiment certification: RSN simplex for model experiment cells.

Maps model evaluation metrics (R2, ROC-AUC, cross-validation folds)
to the RSCT certificate framework (R + S_sup + N = 1). Designed to be
importable by batch scripts today and a dedicated API tomorrow.

Pure functions (no external dependencies):
    derive_simplex  -- map model metrics to R, S_sup, N
    compute_tau     -- temporal stability from per-fold CV
    compute_sigma   -- cross-fold volatility

Service function (uses yrsn when available, falls back gracefully):
    certify_experiment_cell -- full certificate from model metrics
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# yrsn imports -- optional, with graceful fallback
try:
    from yrsn.core.quality.alpha import compute_alpha as _yrsn_alpha
except ImportError:
    _yrsn_alpha = None

try:
    from yrsn.core.quality.omega import compute_omega as _yrsn_omega
except ImportError:
    _yrsn_omega = None

try:
    from yrsn.core.certificates.core import YRSNCertificate as _YRSNCertificate
except ImportError:
    _YRSNCertificate = None

try:
    from yrsn.core.degradation.diagnosis import DegradationDiagnoser as _Diagnoser
except ImportError:
    _Diagnoser = None


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentCertificate:
    """RSCT certificate for a model experiment cell.

    Attributes:
        R: Relevance (honest signal captured by spatial-blocked evaluation).
        S_sup: Superfluous (leaked signal from random-spatial gap).
        N: Noise (simplex remainder).
        alpha: Signal purity R / (R + N).
        omega: Reliability 1 - S_sup.
        kappa: Geometric compatibility (from Phase 4 diagnostics).
        tau: Temporal stability 1 / (1 + CV).
        sigma: Cross-fold volatility std(fold_metrics).
        coherence: Cross-fold agreement (None if not computable).
        coherence_status: Why coherence was or wasn't computed.
        coherence_detail: Breakdown fields for audit.
        diagnosis_label: DegradationDiagnoser 3x3 grid label (or None).
        yrsn_available: Whether yrsn was used for alpha/omega computation.
    """

    R: float
    S_sup: float
    N: float
    alpha: float
    omega: float
    kappa: float
    tau: float
    sigma: float
    coherence: float | None = None
    coherence_status: str = "NOT_COMPUTED"
    coherence_detail: dict = field(default_factory=dict)
    diagnosis_label: str | None = None
    yrsn_available: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/parquet."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure functions (no yrsn dependency)
# ---------------------------------------------------------------------------

def derive_simplex(
    spatial_metric: float | None,
    random_metric: float | None,
    task_type: str,
) -> tuple[float, float, float]:
    """Map model metrics to the R + S_sup + N = 1 simplex.

    Args:
        spatial_metric: Primary metric on spatial-blocked split (honest).
        random_metric: Primary metric on random split (potentially leaked).
        task_type: "regression" (R2) or "classification" (ROC-AUC).

    Returns:
        (R, S_sup, N) all in [0, 1] summing to 1.

    R derivation:
        Regression R2: clip to [0, 1].
        Classification ROC-AUC: 2*(AUC - 0.5) maps [0.5, 1] -> [0, 1].

    S_sup derivation:
        Normalized gap between random and spatial performance.
        Represents leaked/spurious signal.

    N derivation:
        Simplex remainder: 1 - R - S_sup.
    """
    if spatial_metric is None:
        return (0.0, 0.0, 1.0)

    if task_type == "classification":
        R = float(np.clip(2.0 * (spatial_metric - 0.5), 0.0, 1.0))
    else:
        R = float(np.clip(spatial_metric, 0.0, 1.0))

    S_sup = 0.0
    if random_metric is not None and spatial_metric is not None:
        gap = random_metric - spatial_metric
        if gap > 0 and random_metric > 0:
            raw_s = gap / max(abs(random_metric), 0.01)
            S_sup = float(np.clip(raw_s, 0.0, 1.0 - R))

    N = float(np.clip(1.0 - R - S_sup, 0.0, 1.0))
    return (R, S_sup, N)


def compute_tau(fold_metrics: list[float]) -> float:
    """Temporal stability from per-fold coefficient of variation.

    tau = 1 / (1 + CV) where CV = std / |mean|.
    High tau = stable across folds; low tau = volatile.
    Returns value in (0, 1].
    """
    if len(fold_metrics) < 2:
        return 1.0
    mean = np.mean(fold_metrics)
    std = np.std(fold_metrics, ddof=1)
    if abs(mean) < 1e-10:
        return 0.5
    cv = std / abs(mean)
    return float(1.0 / (1.0 + cv))


def compute_sigma(fold_metrics: list[float]) -> float:
    """Cross-fold volatility (sample standard deviation)."""
    if len(fold_metrics) < 2:
        return 0.0
    return float(np.std(fold_metrics, ddof=1))


@dataclass(frozen=True)
class CoherenceResult:
    """Result of coherence computation with full audit trail."""

    coherence: float | None
    status: str
    n_folds_expected: int
    n_folds_valid: int
    fold_directional_agreement: float | None
    fold_magnitude_stability: float | None
    failure_reason: str | None


def compute_coherence(
    fold_metrics: list[float],
    n_folds_expected: int | None = None,
    task_type: str = "regression",
    alpha: float | None = None,
    min_prevalence: float = 0.01,
) -> CoherenceResult:
    """Cross-fold agreement: do the folds tell the same story?

    coherence = valid_fold_rate * (
        0.6 * directional_agreement + 0.4 * magnitude_stability
    )

    Args:
        fold_metrics: Per-fold primary metric values.
        n_folds_expected: Expected number of folds (defaults to len).
        task_type: "regression" or "classification".
        alpha: Signal purity from certificate. If 0, target is degenerate.
        min_prevalence: Minimum alpha to consider target non-degenerate.

    Returns:
        CoherenceResult with value, status, and breakdown fields.
    """
    n_expected = n_folds_expected or len(fold_metrics)

    # Degenerate target: alpha == 0 means no usable signal
    if alpha is not None and alpha < min_prevalence:
        return CoherenceResult(
            coherence=None,
            status="NOT_COMPUTED_TARGET_DEGENERATE",
            n_folds_expected=n_expected,
            n_folds_valid=len(fold_metrics),
            fold_directional_agreement=None,
            fold_magnitude_stability=None,
            failure_reason=f"alpha={alpha:.4f} < min_prevalence={min_prevalence}",
        )

    if len(fold_metrics) < 2:
        return CoherenceResult(
            coherence=None,
            status="NOT_COMPUTED_INSUFFICIENT_FOLDS",
            n_folds_expected=n_expected,
            n_folds_valid=len(fold_metrics),
            fold_directional_agreement=None,
            fold_magnitude_stability=None,
            failure_reason=f"need >= 2 folds, got {len(fold_metrics)}",
        )

    arr = np.asarray(fold_metrics, dtype=float)
    valid = arr[np.isfinite(arr)]
    n_valid = len(valid)

    if n_valid < 2:
        return CoherenceResult(
            coherence=None,
            status="NOT_COMPUTED_INSUFFICIENT_FOLDS",
            n_folds_expected=n_expected,
            n_folds_valid=n_valid,
            fold_directional_agreement=None,
            fold_magnitude_stability=None,
            failure_reason=f"need >= 2 finite folds, got {n_valid}",
        )

    valid_fold_rate = n_valid / max(n_expected, 1)

    # Directional agreement: share of folds agreeing with median direction.
    # For regression (R2): positive = model beats mean, negative = worse.
    # For classification (AUC): > 0.5 = above chance.
    if task_type == "classification":
        direction_threshold = 0.0  # AUC is already mapped to 2*(AUC-0.5)
    else:
        direction_threshold = 0.0

    med = float(np.median(valid))
    if med > direction_threshold:
        n_agree = int(np.sum(valid > direction_threshold))
    elif med < direction_threshold:
        n_agree = int(np.sum(valid <= direction_threshold))
    else:
        n_agree = n_valid  # all at threshold = unanimous
    directional_agreement = n_agree / n_valid

    # Magnitude stability: 1 - normalized dispersion via IQR.
    # IQR is robust to a single outlier fold (unlike range).
    q25, q75 = float(np.percentile(valid, 25)), float(np.percentile(valid, 75))
    iqr = q75 - q25
    metric_span = float(np.max(valid) - np.min(valid))
    if metric_span > 1e-12:
        magnitude_stability = float(np.clip(1.0 - iqr / metric_span, 0.0, 1.0))
    else:
        magnitude_stability = 1.0  # all folds identical = perfect stability

    coherence = valid_fold_rate * (
        0.6 * directional_agreement + 0.4 * magnitude_stability
    )
    coherence = float(np.clip(coherence, 0.0, 1.0))

    return CoherenceResult(
        coherence=round(coherence, 6),
        status="COMPUTED",
        n_folds_expected=n_expected,
        n_folds_valid=n_valid,
        fold_directional_agreement=round(directional_agreement, 6),
        fold_magnitude_stability=round(magnitude_stability, 6),
        failure_reason=None,
    )


# ---------------------------------------------------------------------------
# Service function (yrsn-enhanced with fallback)
# ---------------------------------------------------------------------------

def certify_experiment_cell(
    spatial_metric: float | None,
    random_metric: float | None,
    task_type: str,
    fold_metrics: list[float] | None = None,
    kappa_geom: float | None = None,
) -> ExperimentCertificate:
    """Produce an RSCT certificate for one experiment cell.

    Uses yrsn functions (compute_alpha, compute_omega, DegradationDiagnoser)
    when available; falls back to direct formulas otherwise.

    Args:
        spatial_metric: Primary metric on spatial-blocked split.
        random_metric: Primary metric on random split.
        task_type: "regression" or "classification".
        fold_metrics: Per-fold metric values for tau/sigma computation.
        kappa_geom: Geometric compatibility from Phase 0.5 (pre-training,
            model-free).  Must have zero dependency on RSN, fold metrics,
            predictions, or residuals.

    Returns:
        ExperimentCertificate with all RSCT signals.
    """
    fold_metrics = fold_metrics or []

    R, S_sup, N = derive_simplex(spatial_metric, random_metric, task_type)

    # Alpha: signal purity
    if _yrsn_alpha is not None:
        alpha = _yrsn_alpha(R, N)
    else:
        alpha = R / (R + N) if (R + N) > 0 else 0.0

    # Omega: reliability
    if _yrsn_omega is not None:
        omega = _yrsn_omega(S_sup)
    else:
        omega = 1.0 - S_sup

    kappa = kappa_geom if kappa_geom is not None else 0.0
    tau = compute_tau(fold_metrics)
    sigma = compute_sigma(fold_metrics)

    coh = compute_coherence(
        fold_metrics,
        n_folds_expected=len(fold_metrics),
        task_type=task_type,
        alpha=alpha,
    )

    # Degradation diagnosis (3x3 grid)
    diagnosis_label = None
    if _Diagnoser is not None:
        try:
            diagnoser = _Diagnoser()
            result = diagnoser.diagnose(alpha=alpha, kappa=kappa, sigma=sigma)
            diagnosis_label = (
                str(result.label) if hasattr(result, 'label') else str(result)
            )
        except Exception as e:
            log.warning("DegradationDiagnoser failed: %s", e)

    return ExperimentCertificate(
        R=round(R, 6),
        S_sup=round(S_sup, 6),
        N=round(N, 6),
        alpha=round(alpha, 6),
        omega=round(omega, 6),
        kappa=round(kappa, 6),
        tau=round(tau, 6),
        sigma=round(sigma, 6),
        coherence=coh.coherence,
        coherence_status=coh.status,
        coherence_detail={
            "n_folds_expected": coh.n_folds_expected,
            "n_folds_valid": coh.n_folds_valid,
            "fold_directional_agreement": coh.fold_directional_agreement,
            "fold_magnitude_stability": coh.fold_magnitude_stability,
            "failure_reason": coh.failure_reason,
        },
        diagnosis_label=diagnosis_label,
        yrsn_available=_yrsn_alpha is not None,
    )
