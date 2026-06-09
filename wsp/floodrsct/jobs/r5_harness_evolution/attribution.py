"""Failure attribution for R5 harness evolution.

Builds structured failure reports from VLM outputs and reference labels.
The evolver receives these reports (not raw labels) to propose patches.
Held-out data is redacted before the report reaches the evolver.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure types
# ---------------------------------------------------------------------------

FAILURE_TYPES = [
    "false_negative_high_reference",   # VLM says low, reference says flood zone
    "false_positive_low_reference",    # VLM says high, reference says no flood
    "schema_violation",                # output doesn't match expected schema
    "missing_evidence_citation",       # VLM didn't cite supplied evidence
    "unprovided_external_claim",       # VLM invented unsupported facts
    "low_confidence_correct",          # correct but low confidence
    "spatial_outlier",                 # disagrees with all neighbors
]


def _classify_failure(
    prediction: dict,
    reference: dict,
    judgment: dict,
) -> str | None:
    """Classify a single ZCTA outcome into a failure type, or None if ok."""
    pred_zone = (prediction.get("fema_zone_prediction") or "").upper()
    ref_zone = (reference.get("fema_zone") or "").upper()

    # Schema violation
    if not judgment.get("followed_schema", True):
        return "schema_violation"

    # Unprovided claims
    if not judgment.get("no_unprovided_claims", True):
        return "unprovided_external_claim"

    # Missing evidence citation
    if not judgment.get("used_structured_evidence", True):
        return "missing_evidence_citation"

    # Zone classification errors
    flood_zones = {"A", "AE", "AH", "AO", "V", "VE"}
    non_flood = {"X", "X500", "NONE", ""}

    if ref_zone in flood_zones and pred_zone in non_flood:
        return "false_negative_high_reference"
    if ref_zone in non_flood and pred_zone in flood_zones:
        return "false_positive_low_reference"

    return None


def build_failure_report(
    predictions: list[dict],
    references: list[dict],
    judgments: list[dict],
    step: int,
    harness_id: str,
    batch_id: str,
    primary_metric_value: float,
    primary_metric_baseline: float,
    heldout_ids: set[str] | None = None,
    max_examples_per_group: int = 5,
) -> dict:
    """Build a structured failure report for the evolver.

    Held-out ZCTAs are excluded from the report. The evolver should
    receive this report, not raw prediction/reference pairs.
    """
    heldout_ids = heldout_ids or set()
    ref_map = {r["zcta"]: r for r in references}
    judge_map = {j["zcta"]: j for j in judgments}

    # Classify failures
    failure_groups: dict[str, list[dict]] = {}
    for pred in predictions:
        zcta = pred["zcta"]
        if zcta in heldout_ids:
            continue
        ref = ref_map.get(zcta)
        if ref is None:
            continue
        judge = judge_map.get(zcta, {})

        ftype = _classify_failure(pred, ref, judge)
        if ftype is None:
            continue

        if ftype not in failure_groups:
            failure_groups[ftype] = []

        # Summarize without leaking exact labels
        summary = {
            "zcta": zcta,
            "scenario": pred.get("scenario", ""),
            "vlm_risk_level": pred.get("risk_level", ""),
            "vlm_zone_prediction": pred.get("fema_zone_prediction", ""),
            "adherence_flags": {
                "used_map": judge.get("map_loaded", False),
                "used_structured_evidence": judge.get(
                    "used_structured_evidence", False
                ),
                "invented_unprovided_claim": not judge.get(
                    "no_unprovided_claims", True
                ),
            },
        }
        # Include allowed feature summary (public data, not labels)
        if "allowed_features_summary" in pred:
            summary["allowed_features_summary"] = pred[
                "allowed_features_summary"
            ]

        failure_groups[ftype].append(summary)

    # Build report with capped examples
    groups_out = []
    for ftype, examples in failure_groups.items():
        groups_out.append({
            "failure_type": ftype,
            "count": len(examples),
            "examples": examples[:max_examples_per_group],
        })

    return {
        "step": step,
        "harness_id": harness_id,
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "primary_metric": {
            "name": "zone_macro_f1",
            "value": primary_metric_value,
            "baseline": primary_metric_baseline,
        },
        "failure_groups": groups_out,
        "do_not_use": {
            "heldout_zctas": "redacted",
            "heldout_labels": "redacted",
            "test_scenario_labels": "redacted",
        },
    }
