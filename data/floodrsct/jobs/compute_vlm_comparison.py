#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   generator: deepseek/deepseek-chat-v3.1 (via OpenRouter)
#   cleanup_by: Martin
#   cleanup_summary: Strip markdown fences, fix merge column (zcta not
#       zcta_id), add shebang/docstring/logging template, add all 5 scenarios,
#       add dry-run early exit, fix timestamp to datetime.now(timezone.utc),
#       remove unused boto3 import, add provenance header at top
#   see: ../exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml
# =============================================================================
"""compute_vlm_comparison.py -- Phase R4.5: VLM comparison.

Compares VLM risk scores against observed NFIP claims (H7) and
across VLMs (H8 pairwise agreement). Produces R4 money table
extension with per-fold Spearman correlations.

Usage:
    python compute_vlm_comparison.py --upload
    python compute_vlm_comparison.py --dry-run
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]
VLMS = ["gpt4o", "gemini", "jina", "nova", "qwen"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str) -> Optional[pd.DataFrame]:
    """Load parquet from S3, return None on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def _load_json(s3, key: str) -> Optional[Dict]:
    """Load JSON from S3, return None on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def _get_r02_headline(s3, scenario: str, level: str) -> Optional[float]:
    """Get R0-R2 headline metric for obs_nfip_event_claims."""
    key = f"{RESULTS_PREFIX}/{level}_{scenario}.json"
    data = _load_json(s3, key)
    if not data:
        return None
    for cell in data.get("cells", []):
        if (cell.get("target") == "obs_nfip_event_claims"
                and "spatial_metric" in cell):
            return cell["spatial_metric"]
    return None


def _compute_fold_rhos(
    merged: pd.DataFrame, score_col: str, obs_col: str,
) -> list[float]:
    """Compute Spearman rho per fold."""
    rhos = []
    for fold in sorted(merged["fold"].unique()):
        sub = merged[merged["fold"] == fold]
        if len(sub) < 3:
            continue
        rho, _ = spearmanr(sub[score_col], sub[obs_col])
        rhos.append(round(float(rho), 4))
    return rhos


# ---------------------------------------------------------------------------
# Per-scenario analysis
# ---------------------------------------------------------------------------

def analyze_scenario(s3, scenario: str) -> Dict[str, Any]:
    """Run VLM comparison for one scenario."""
    row: Dict[str, Any] = {
        "scenario": scenario,
        "n_zctas": 0,
        "h7_pass": False,
        "h8_pass": False,
    }

    # Load observed claims
    claims_key = f"processed/{scenario}/{scenario}_event_features.parquet"
    claims_df = _load_parquet(s3, claims_key)
    if claims_df is None:
        row["error"] = "claims data not available"
        return row
    claims_df["zcta_id"] = claims_df["zcta_id"].astype(str)

    # Load fold assignments
    folds_key = f"folds/{scenario}_folds.parquet"
    folds_df = _load_parquet(s3, folds_key)
    if folds_df is not None:
        folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
        claims_df = claims_df.merge(folds_df, on="zcta_id", how="inner")
    else:
        claims_df["fold"] = 0

    row["n_zctas"] = len(claims_df)

    # Load VLM results and compute correlations
    vlm_scores: Dict[str, pd.Series] = {}
    available_vlms: list[str] = []

    for vlm in VLMS:
        vlm_key = f"{RESULTS_PREFIX}/r4_{vlm}_{scenario}.parquet"
        vlm_df = _load_parquet(s3, vlm_key)
        if vlm_df is None:
            log.warning("No %s results for %s", vlm, scenario)
            continue

        vlm_df["zcta_id"] = vlm_df["zcta_id"].astype(str)
        merged = claims_df.merge(
            vlm_df[["zcta_id", "risk_score"]],
            on="zcta_id", how="inner",
        )
        merged = merged.dropna(subset=["risk_score", "obs_nfip_event_claims"])

        if len(merged) < 3:
            log.warning("Too few matched ZCTAs for %s/%s (n=%d)", vlm, scenario, len(merged))
            continue

        # Overall Spearman
        rho, p = spearmanr(merged["risk_score"], merged["obs_nfip_event_claims"])
        row[f"rho_{vlm}"] = round(float(rho), 4)
        row[f"p_{vlm}"] = round(float(p), 4)
        row[f"n_{vlm}"] = len(merged)

        # Per-fold
        fold_rhos = _compute_fold_rhos(merged, "risk_score", "obs_nfip_event_claims")
        row[f"rho_{vlm}_folds"] = fold_rhos

        vlm_scores[vlm] = merged.set_index("zcta_id")["risk_score"]
        available_vlms.append(vlm)

        # H7 check
        if rho > 0.3:
            row["h7_pass"] = True

    # Pairwise VLM agreement (H8)
    pairwise = {}
    if len(available_vlms) >= 2:
        for i, v1 in enumerate(available_vlms):
            for v2 in available_vlms[i + 1:]:
                common = vlm_scores[v1].index.intersection(vlm_scores[v2].index)
                if len(common) < 3:
                    continue
                rho, _ = spearmanr(
                    vlm_scores[v1].loc[common],
                    vlm_scores[v2].loc[common],
                )
                pairwise[f"{v1}_{v2}"] = round(float(rho), 4)

        row["pairwise_rhos"] = pairwise
        if pairwise:
            row["vlm_agreement"] = round(
                sum(pairwise.values()) / len(pairwise), 4
            )
            row["h8_pass"] = all(r > 0.7 for r in pairwise.values())

    # R0-R2 comparison
    for level in ["r0", "r1", "r2"]:
        metric = _get_r02_headline(s3, scenario, level)
        row[f"rho_{level}"] = round(metric, 4) if metric is not None else None

    # VLM vs R2 delta
    r2_rho = row.get("rho_r2")
    if r2_rho is not None and available_vlms:
        best_vlm_rho = max(
            row.get(f"rho_{v}", -1.0) for v in available_vlms
        )
        row["vlm_vs_r2_delta"] = round(best_vlm_rho - r2_rho, 4)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase R4.5: VLM comparison")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would compare VLMs across %s", SCENARIOS)
        log.info("Reads: r4_{vlm}_{scenario}.parquet + event features + folds")
        log.info("Writes: %s/r4_money_table.json", RESULTS_PREFIX)
        return 0

    s3 = get_s3_client()

    scenario_rows = []
    for scenario in SCENARIOS:
        log.info("--- %s ---", scenario)
        row = analyze_scenario(s3, scenario)
        scenario_rows.append(row)
        log.info("  h7=%s h8=%s vlms=%s",
                 row.get("h7_pass"), row.get("h8_pass"),
                 [v for v in VLMS if f"rho_{v}" in row])

    # Aggregate hypothesis tests
    h7_all_rhos = []
    h8_all_pairwise = []
    for row in scenario_rows:
        for vlm in VLMS:
            rho = row.get(f"rho_{vlm}")
            if rho is not None:
                h7_all_rhos.append((vlm, row["scenario"], rho))
        pw = row.get("pairwise_rhos", {})
        h8_all_pairwise.extend(pw.values())

    h7_pass = any(rho > 0.3 for _, _, rho in h7_all_rhos)
    h7_best = max(h7_all_rhos, key=lambda x: x[2]) if h7_all_rhos else (None, None, None)
    h8_pass = bool(h8_all_pairwise) and all(r > 0.7 for r in h8_all_pairwise)

    result = {
        "phase": "R4.5_vlm_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": SCENARIOS,
        "vlms": VLMS,
        "per_scenario": scenario_rows,
        "hypothesis_tests": {
            "h7": {
                "any_vlm_rho_gt_0.3": h7_pass,
                "best_vlm": h7_best[0],
                "best_scenario": h7_best[1],
                "best_rho": h7_best[2],
            },
            "h8": {
                "all_pairwise_gt_0.7": h8_pass,
                "min_pairwise": round(min(h8_all_pairwise), 4) if h8_all_pairwise else None,
                "n_pairs": len(h8_all_pairwise),
            },
        },
    }

    # Save local
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_file = out_dir / "r4_money_table.json"
    with open(local_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Written to %s", local_file)

    if args.upload:
        key = f"{RESULTS_PREFIX}/r4_money_table.json"
        upload_json_result(s3, BUCKET, key, result)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

    # Summary
    log.info("\n=== R4 VLM Comparison Summary ===")
    log.info("  H7 (any rho > 0.3): %s", h7_pass)
    if h7_best[0]:
        log.info("    best: %s/%s rho=%.4f", h7_best[0], h7_best[1], h7_best[2])
    log.info("  H8 (all pairwise > 0.7): %s", h8_pass)
    if h8_all_pairwise:
        log.info("    min pairwise: %.4f (n=%d)", min(h8_all_pairwise), len(h8_all_pairwise))

    return 0


if __name__ == "__main__":
    sys.exit(main())
