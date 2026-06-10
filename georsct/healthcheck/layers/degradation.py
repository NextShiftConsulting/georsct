"""Layer 2: Screening DegradationType classification.

This is a *screening* classifier for healthcheck triage, producing coarse
categories (UNSTABLE, NOISE_COLLAPSED, NOISE_DOMINANT, CONTENT_DEGRADATION,
GEOMETRY_MISMATCH, TOTAL_FAILURE, HEALTHY, MILD_CONTENT_ISSUE) to route
next-step recommendations in the HealthCard.

It is NOT the canonical yrsn ``DegradationDiagnoser`` (3x3 alpha-kappa grid),
which produces fine-grained root-cause labels (e.g. specific degradation
types from ``yrsn.core.degradation``). When a certificate already carries a
``diagnosis_label`` from the canonical classifier, this layer parses and
returns it rather than re-classifying.

Thresholds are sourced from ``ThresholdPreset`` (via yrsn-controlplane) to
prevent drift from production gate values.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import DegradationResult
from ..thresholds import ThresholdPreset


def _quality_level(value: float) -> str:
    if value >= 0.7:
        return "HIGH"
    if value >= 0.4:
        return "MEDIUM"
    return "LOW"


def _parse_diagnosis_label(label: str) -> DegradationResult | None:
    """Try to parse a stringified DiagnosisResult from s035 output."""
    if not label:
        return None

    # Try structured dict first
    if isinstance(label, dict):
        return DegradationResult(
            degradation_type=str(label.get("degradation_type", "UNKNOWN")),
            confidence=float(label.get("confidence", 0.0)),
            explanation=str(label.get("explanation", "")),
            alpha_level=str(label.get("alpha_level", "UNKNOWN")),
            kappa_level=str(label.get("kappa_level", "UNKNOWN")),
        )

    # Regex extraction from repr string
    deg_match = re.search(r"degradation_type=<[^.]+\.(\w+):", str(label))
    conf_match = re.search(r"confidence=([\d.]+)", str(label))
    expl_match = re.search(r"explanation='([^']*)'", str(label))
    alpha_match = re.search(r"alpha_level=<[^.]+\.(\w+):", str(label))
    kappa_match = re.search(r"kappa_level=<[^.]+\.(\w+):", str(label))

    if deg_match:
        return DegradationResult(
            degradation_type=deg_match.group(1),
            confidence=float(conf_match.group(1)) if conf_match else 0.0,
            explanation=expl_match.group(1) if expl_match else "",
            alpha_level=alpha_match.group(1).upper() if alpha_match else "UNKNOWN",
            kappa_level=kappa_match.group(1).upper() if kappa_match else "UNKNOWN",
        )

    return None


def classify_degradation(
    cert: dict[str, Any],
    gate_decision: str,
    preset: ThresholdPreset | None = None,
) -> DegradationResult:
    """Classify a certificate into a screening DegradationType.

    If the certificate carries a diagnosis_label from yrsn's canonical
    DegradationDiagnoser, parse and return it. Otherwise derive a
    screening classification from the alpha x kappa grid using
    threshold values from the preset.

    Note: This is a *screening* classifier for healthcheck triage, not
    the canonical yrsn DegradationDiagnoser (3x3 grid). The two produce
    different label sets by design: this classifier uses coarser
    categories (UNSTABLE, NOISE_COLLAPSED, etc.) to route next-step
    recommendations, while the canonical classifier provides fine-grained
    degradation typing for root-cause analysis.

    Args:
        cert: Certificate dict.
        gate_decision: EnforcementDecision from Layer 1.
        preset: ThresholdPreset for sourcing boundaries. If None,
            uses hardcoded fallbacks (0.3) for backward compatibility.

    Returns:
        DegradationResult.
    """
    # Try parsing existing label from canonical DegradationDiagnoser
    label = cert.get("diagnosis_label")
    if label:
        parsed = _parse_diagnosis_label(label)
        if parsed:
            return parsed

    alpha = cert.get("alpha", 0.0)
    kappa = cert.get("kappa_compat", 0.0)
    sigma = cert.get("sigma", 0.0)
    R = cert.get("R", 0.0)
    N = cert.get("N", 0.0)

    # Source thresholds from preset to prevent drift from gate values
    sigma_unstable = preset.coherence_warn if preset else 0.3
    alpha_collapsed = preset.alpha_min if preset else 0.3

    alpha_lvl = _quality_level(alpha)
    kappa_lvl = _quality_level(kappa)

    # Classify
    if sigma >= sigma_unstable:
        deg = "UNSTABLE"
        expl = f"sigma={sigma:.3f} >= {sigma_unstable}, metrics unreliable"
        conf = min(1.0, (sigma - sigma_unstable) / 0.2)
    elif N > R and alpha < alpha_collapsed:
        deg = "NOISE_COLLAPSED"
        expl = f"N={N:.3f} > R={R:.3f}, alpha={alpha:.3f} collapsed"
        conf = 0.9
    elif N > R:
        deg = "NOISE_DOMINANT"
        expl = f"N={N:.3f} > R={R:.3f}, noise exceeds signal"
        conf = min(1.0, (N - R) / 0.3)
    elif alpha_lvl == "LOW" and kappa_lvl != "LOW":
        deg = "CONTENT_DEGRADATION"
        expl = f"alpha={alpha:.3f} {alpha_lvl}, kappa={kappa:.3f} {kappa_lvl}"
        conf = 0.7
    elif kappa_lvl == "LOW" and alpha_lvl != "LOW":
        deg = "GEOMETRY_MISMATCH"
        expl = f"alpha={alpha:.3f} {alpha_lvl}, kappa={kappa:.3f} {kappa_lvl}"
        conf = 0.7
    elif alpha_lvl == "LOW" and kappa_lvl == "LOW":
        deg = "TOTAL_FAILURE"
        expl = f"alpha={alpha:.3f} {alpha_lvl}, kappa={kappa:.3f} {kappa_lvl}"
        conf = 0.9
    elif alpha_lvl == "HIGH" and kappa_lvl == "HIGH":
        deg = "HEALTHY"
        expl = f"alpha={alpha:.3f} {alpha_lvl}, kappa={kappa:.3f} {kappa_lvl}"
        conf = 0.1
    else:
        deg = "MILD_CONTENT_ISSUE"
        expl = f"alpha={alpha:.3f} {alpha_lvl}, kappa={kappa:.3f} {kappa_lvl}"
        conf = 0.3

    return DegradationResult(
        degradation_type=deg,
        confidence=round(conf, 4),
        explanation=expl,
        alpha_level=alpha_lvl,
        kappa_level=kappa_lvl,
    )
