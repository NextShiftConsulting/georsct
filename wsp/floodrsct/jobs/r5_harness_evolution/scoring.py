"""Scoring functions for R5 harness evolution.

Computes primary metric (zone_macro_f1) and diagnostic metrics
(activation, adherence, grounding, spatial coherence).
Spatial coherence is diagnostic-only, never a reward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary metric: zone classification macro-F1
# ---------------------------------------------------------------------------

ZONE_BUCKETS = ["none", "a_ae", "v_coastal", "unknown"]

_FEMA_ZONE_MAP = {
    "A": "a_ae", "AE": "a_ae", "AH": "a_ae", "AO": "a_ae",
    "V": "v_coastal", "VE": "v_coastal",
    "X": "none", "X500": "none",
    "D": "unknown", "": "unknown", None: "unknown",
}


def _bucket_fema_zone(zone: str | None) -> str:
    if zone is None:
        return "unknown"
    return _FEMA_ZONE_MAP.get(zone.strip().upper(), "unknown")


def zone_macro_f1(
    predictions: list[dict],
    references: list[dict],
) -> float:
    """Compute macro-F1 between VLM predicted zone bucket and FEMA reference.

    Args:
        predictions: list of {"zcta": str, "fema_zone_prediction": str}
        references: list of {"zcta": str, "fema_zone": str}

    Returns:
        Macro-averaged F1 score across zone buckets.
    """
    ref_map = {r["zcta"]: _bucket_fema_zone(r.get("fema_zone")) for r in references}
    pred_map = {p["zcta"]: _bucket_fema_zone(p.get("fema_zone_prediction")) for p in predictions}

    common = set(ref_map) & set(pred_map)
    if not common:
        log.warning("No overlapping ZCTAs between predictions and references")
        return 0.0

    y_true = [ref_map[z] for z in sorted(common)]
    y_pred = [pred_map[z] for z in sorted(common)]

    # Per-bucket precision/recall/F1
    f1_scores = []
    for bucket in ZONE_BUCKETS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == bucket and p == bucket)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != bucket and p == bucket)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == bucket and p != bucket)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


# ---------------------------------------------------------------------------
# RSN simplex from output-level judgments
# ---------------------------------------------------------------------------

def simplex_from_judgments(metrics: JudgmentMetrics) -> dict | None:
    """Derive RSN simplex from output-level judgment metrics.

    Mapping:
        R     = grounding_rate       (cited real evidence, no fabrication)
        N     = unprovided_claim_rate (invented unsupported facts)
        S_sup = 1 - R - N            (compliant but not fully grounded)

    The simplex R + S_sup + N = 1 holds by construction: grounding_rate
    requires no_unprovided_claims=True, so an output contributing to N
    cannot contribute to R.

    Returns None if no outputs were judged.
    """
    if metrics.n_outputs == 0:
        return None

    R = metrics.grounding_rate
    N = metrics.unprovided_claim_rate
    S_sup = max(0.0, 1.0 - R - N)

    return {"R": R, "S_sup": S_sup, "N": N}


# ---------------------------------------------------------------------------
# Activation / adherence / grounding metrics
# ---------------------------------------------------------------------------

@dataclass
class JudgmentMetrics:
    """Aggregated activation/adherence/grounding metrics from VLM outputs."""
    activation_rate: float = 0.0
    adherence_rate: float = 0.0
    grounding_rate: float = 0.0
    schema_validity_rate: float = 0.0
    unprovided_claim_rate: float = 0.0
    n_outputs: int = 0


def aggregate_judgments(judgments: list[dict]) -> JudgmentMetrics:
    """Aggregate per-ZCTA judgment dicts into summary metrics.

    Each judgment dict should have keys:
        map_loaded, evidence_table_loaded, rubric_loaded,
        used_visual_evidence, used_structured_evidence,
        followed_schema, no_unprovided_claims, adherence_score
    """
    if not judgments:
        return JudgmentMetrics()

    n = len(judgments)

    def rate(key: str) -> float:
        return sum(1 for j in judgments if j.get(key, False)) / n

    # Activation: all required artifacts loaded
    activation = sum(
        1 for j in judgments
        if j.get("map_loaded") and j.get("evidence_table_loaded")
        and j.get("rubric_loaded")
    ) / n

    # Adherence: followed schema + used evidence
    adherence = sum(
        1 for j in judgments
        if j.get("followed_schema") and j.get("used_structured_evidence")
    ) / n

    # Grounding: cited supplied evidence, no external claims
    grounding = sum(
        1 for j in judgments
        if j.get("used_visual_evidence") and j.get("used_structured_evidence")
        and j.get("no_unprovided_claims", True)
    ) / n

    return JudgmentMetrics(
        activation_rate=activation,
        adherence_rate=adherence,
        grounding_rate=grounding,
        schema_validity_rate=rate("followed_schema"),
        unprovided_claim_rate=1.0 - rate("no_unprovided_claims"),
        n_outputs=n,
    )


# ---------------------------------------------------------------------------
# Spatial diagnostics (diagnostic-only, NOT reward)
# ---------------------------------------------------------------------------

def spatial_residual_morans_i(
    residuals: pd.Series,
    weights_matrix: np.ndarray,
) -> float:
    """Moran's I on prediction residuals.

    Diagnostic only -- must NOT be used as optimization target.
    High Moran's I indicates spatially clustered errors.
    """
    n = len(residuals)
    if n < 3:
        return 0.0

    z = residuals - residuals.mean()
    s0 = weights_matrix.sum()

    if s0 == 0 or z.var() == 0:
        return 0.0

    numerator = float(n * (z.values @ weights_matrix @ z.values))
    denominator = float(s0 * (z @ z))

    return numerator / denominator if denominator != 0 else 0.0
