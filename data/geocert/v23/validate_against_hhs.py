#!/usr/bin/env python3
"""
validate_against_hhs.py -- Cross-validate GeoCert against official HHS/CDC PLACES on HF.

Downloads the official HHS-Official/places-zcta-data-gis-friendly-format-2023-release
from Hugging Face and runs three integrity checks:

  1. ZCTA Reconciliation: Are all GeoCert ZCTAs valid? How does our 31,789 relate
     to HHS's 32,409?
  2. Value Cross-Check: Do our 21 CDC PLACES target values match the official source?
     Compares values, checks rank correlation, flags any mismatches.
  3. Coverage Validation: Do our 260 "missing CDC PLACES" ZCTAs actually lack data
     in the official HHS release?

Output: validation_report.json + console summary

Usage:
    python validate_against_hhs.py
    python validate_against_hhs.py --geocert-path C:/tmp/geocert_release/geocert_table.parquet

Prerequisites:
    pip install datasets pandas scipy
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

HF_DATASET = "HHS-Official/places-zcta-data-gis-friendly-format-2023-release"

# Mapping: GeoCert target column -> HHS CrudePrev column
# These are the 21 CDC PLACES targets in GeoCert
TARGET_TO_HHS = {
    "target_annual_checkup":          "CHECKUP_CrudePrev",
    "target_arthritis":               "ARTHRITIS_CrudePrev",
    "target_asthma":                  "CASTHMA_CrudePrev",
    "target_binge_drinking":          "BINGE_CrudePrev",
    "target_bp_medicated":            "BPMED_CrudePrev",
    "target_cancer":                  "CANCER_CrudePrev",
    "target_cholesterol_screening":   "CHOLSCREEN_CrudePrev",
    "target_chronic_kidney_disease":  "KIDNEY_CrudePrev",
    "target_copd":                    "COPD_CrudePrev",
    "target_coronary_heart_disease":  "CHD_CrudePrev",
    "target_dental_visit":            "DENTAL_CrudePrev",
    "target_diabetes":                "DIABETES_CrudePrev",
    "target_high_blood_pressure":     "BPHIGH_CrudePrev",
    "target_high_cholesterol":        "HIGHCHOL_CrudePrev",
    "target_mental_health_not_good":  "MHLTH_CrudePrev",
    "target_obesity":                 "OBESITY_CrudePrev",
    "target_physical_health_not_good": "PHLTH_CrudePrev",
    "target_physical_inactivity":     "LPA_CrudePrev",
    "target_sleep_less_7hr":          "SLEEP_CrudePrev",
    "target_smoking":                 "CSMOKING_CrudePrev",
    "target_stroke":                  "STROKE_CrudePrev",
}


def load_hhs_dataset() -> pd.DataFrame:
    """Load official HHS PLACES 2023 from Hugging Face."""
    log.info("Loading HHS dataset from Hugging Face: %s", HF_DATASET)
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, split="train")
        hhs = ds.to_pandas()
    except ImportError:
        log.info("'datasets' not installed, trying direct parquet URL...")
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/data/dataset.csv"
        hhs = pd.read_csv(url)

    hhs["zcta_id"] = hhs["ZCTA5"].astype(str).str.zfill(5)
    log.info("  HHS dataset: %d rows, %d columns", len(hhs), len(hhs.columns))
    return hhs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geocert-path", type=str,
        default="C:/tmp/geocert_release/geocert_table.parquet",
        help="Path to GeoCert table parquet",
    )
    parser.add_argument(
        "--output", type=str,
        default="C:/tmp/geocert_release/validation_report.json",
        help="Output path for validation report",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # -- Load GeoCert --
    gc_path = Path(args.geocert_path)
    if not gc_path.exists():
        log.error("GeoCert not found at %s", gc_path)
        sys.exit(1)

    gc = pd.read_parquet(gc_path)
    gc["zcta_id"] = gc["zcta_id"].astype(str).str.zfill(5)
    log.info("GeoCert: %d rows, %d columns", len(gc), len(gc.columns))

    # -- Load HHS --
    hhs = load_hhs_dataset()

    # =====================================================================
    # CHECK 1: ZCTA Reconciliation
    # =====================================================================
    log.info("")
    log.info("=" * 60)
    log.info("CHECK 1: ZCTA RECONCILIATION")
    log.info("=" * 60)

    gc_zctas = set(gc["zcta_id"])
    hhs_zctas = set(hhs["zcta_id"])

    in_both = gc_zctas & hhs_zctas
    gc_only = gc_zctas - hhs_zctas
    hhs_only = hhs_zctas - gc_zctas

    log.info("  GeoCert ZCTAs:     %d", len(gc_zctas))
    log.info("  HHS PLACES ZCTAs:     %d", len(hhs_zctas))
    log.info("  In both:              %d", len(in_both))
    log.info("  GeoCert-only:         %d", len(gc_only))
    log.info("  HHS-only:             %d", len(hhs_only))

    if gc_only:
        log.warning("  GeoCert ZCTAs NOT in HHS (first 10): %s", sorted(gc_only)[:10])
    if hhs_only:
        log.info("  HHS ZCTAs NOT in GeoCert (first 10): %s", sorted(hhs_only)[:10])

    check1_pass = len(gc_only) == 0
    log.info("  RESULT: %s -- all GeoCert ZCTAs found in HHS",
             "PASS" if check1_pass else "INVESTIGATE")

    # =====================================================================
    # CHECK 2: Value Cross-Check
    # =====================================================================
    log.info("")
    log.info("=" * 60)
    log.info("CHECK 2: VALUE CROSS-CHECK (21 CDC PLACES targets)")
    log.info("=" * 60)

    # Check which HHS columns exist
    hhs_cols_available = set(hhs.columns)
    missing_hhs_cols = {
        gc_col: hhs_col for gc_col, hhs_col in TARGET_TO_HHS.items()
        if hhs_col not in hhs_cols_available
    }
    if missing_hhs_cols:
        log.warning("  HHS columns not found: %s", missing_hhs_cols)

    # Join on zcta_id for common ZCTAs
    merged = gc.merge(hhs, on="zcta_id", how="inner", suffixes=("_gc", "_hhs"))
    log.info("  Joined rows (common ZCTAs): %d", len(merged))

    value_checks = {}
    all_correlations = []
    all_max_diffs = []
    mismatched_cols = []

    for gc_col, hhs_col in TARGET_TO_HHS.items():
        if hhs_col not in hhs_cols_available:
            value_checks[gc_col] = {"status": "SKIP", "reason": f"HHS column {hhs_col} not found"}
            continue

        gc_vals = merged[gc_col].values
        hhs_vals = merged[hhs_col].values

        # Only compare where both are non-null
        valid_mask = ~(np.isnan(gc_vals) | np.isnan(hhs_vals))
        n_valid = int(valid_mask.sum())

        if n_valid < 10:
            value_checks[gc_col] = {"status": "SKIP", "reason": f"Only {n_valid} valid pairs"}
            continue

        gc_v = gc_vals[valid_mask]
        hhs_v = hhs_vals[valid_mask]

        # Absolute difference
        abs_diff = np.abs(gc_v - hhs_v)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))
        median_diff = float(np.median(abs_diff))

        # Exact match count
        n_exact = int(np.sum(abs_diff < 0.001))
        pct_exact = round(n_exact / n_valid * 100, 2)

        # Spearman rank correlation
        rho, p_val = stats.spearmanr(gc_v, hhs_v)

        # Pearson correlation
        r, r_pval = stats.pearsonr(gc_v, hhs_v)

        check = {
            "hhs_column": hhs_col,
            "n_valid_pairs": n_valid,
            "n_exact_match": n_exact,
            "pct_exact_match": pct_exact,
            "max_abs_diff": round(max_diff, 6),
            "mean_abs_diff": round(mean_diff, 6),
            "median_abs_diff": round(median_diff, 6),
            "spearman_rho": round(float(rho), 6),
            "spearman_p": float(p_val),
            "pearson_r": round(float(r), 6),
            "pearson_p": float(r_pval),
        }

        # Verdict
        if pct_exact >= 99.0 and rho > 0.999:
            check["status"] = "PASS"
        elif rho > 0.99:
            check["status"] = "PASS_CLOSE"
        elif rho > 0.95:
            check["status"] = "WARN"
            mismatched_cols.append(gc_col)
        else:
            check["status"] = "FAIL"
            mismatched_cols.append(gc_col)

        value_checks[gc_col] = check
        all_correlations.append(rho)
        all_max_diffs.append(max_diff)

        status_icon = check["status"]
        log.info("  %-40s %s  rho=%.6f  max_diff=%.4f  exact=%.1f%%",
                 gc_col, status_icon, rho, max_diff, pct_exact)

    if all_correlations:
        min_rho = min(all_correlations)
        mean_rho = np.mean(all_correlations)
        max_max_diff = max(all_max_diffs)
        log.info("")
        log.info("  Summary: min_rho=%.6f, mean_rho=%.6f, max_diff_any=%.4f",
                 min_rho, mean_rho, max_max_diff)

    check2_pass = len(mismatched_cols) == 0
    log.info("  RESULT: %s", "PASS" if check2_pass else f"INVESTIGATE ({len(mismatched_cols)} columns)")

    # =====================================================================
    # CHECK 3: Coverage Validation
    # =====================================================================
    log.info("")
    log.info("=" * 60)
    log.info("CHECK 3: COVERAGE VALIDATION (260 missing CDC PLACES ZCTAs)")
    log.info("=" * 60)

    # Our ZCTAs flagged as missing CDC PLACES
    gc_missing_cdc = set(gc.loc[~gc["has_cdc_places"], "zcta_id"])
    gc_has_cdc = set(gc.loc[gc["has_cdc_places"], "zcta_id"])
    log.info("  GeoCert ZCTAs with has_cdc_places=False: %d", len(gc_missing_cdc))

    # Check these against HHS: are they absent or do they have null values?
    missing_in_hhs = gc_missing_cdc - hhs_zctas  # Not in HHS at all
    missing_but_in_hhs = gc_missing_cdc & hhs_zctas  # In HHS but we flagged missing

    log.info("  Of those, absent from HHS entirely:      %d", len(missing_in_hhs))
    log.info("  Of those, present in HHS:                %d", len(missing_but_in_hhs))

    # For those present in HHS, check if HHS also has nulls
    false_negatives = []  # We say missing, HHS has data
    confirmed_missing = []  # We say missing, HHS also missing

    if missing_but_in_hhs:
        hhs_subset = hhs[hhs["zcta_id"].isin(missing_but_in_hhs)]
        # Pick a representative HHS column to check
        test_col = "DIABETES_CrudePrev"
        if test_col in hhs.columns:
            for _, row in hhs_subset.iterrows():
                zcta = row["zcta_id"]
                val = row[test_col]
                if pd.notna(val):
                    false_negatives.append(zcta)
                else:
                    confirmed_missing.append(zcta)

            log.info("  HHS has data for %d of our 'missing' ZCTAs (false negatives?)",
                     len(false_negatives))
            log.info("  HHS also null for %d of our 'missing' ZCTAs (confirmed)",
                     len(confirmed_missing))
            if false_negatives:
                log.warning("  FALSE NEGATIVES (first 10): %s", sorted(false_negatives)[:10])

    # Reverse check: ZCTAs where we say has_cdc=True but HHS has nulls
    reverse_problems = []
    if "DIABETES_CrudePrev" in hhs.columns:
        hhs_with_data = hhs[hhs["zcta_id"].isin(gc_has_cdc)]
        for _, row in hhs_with_data.iterrows():
            if pd.isna(row["DIABETES_CrudePrev"]):
                reverse_problems.append(row["zcta_id"])
        if reverse_problems:
            log.warning("  REVERSE: %d ZCTAs where we say has_cdc=True but HHS has null diabetes",
                        len(reverse_problems))

    check3_pass = len(false_negatives) == 0 and len(reverse_problems) == 0
    log.info("  RESULT: %s", "PASS" if check3_pass else "INVESTIGATE")

    # =====================================================================
    # OVERALL VERDICT
    # =====================================================================
    log.info("")
    log.info("=" * 60)
    log.info("OVERALL VALIDATION VERDICT")
    log.info("=" * 60)
    all_pass = check1_pass and check2_pass and check3_pass
    log.info("  Check 1 (ZCTA reconciliation):  %s", "PASS" if check1_pass else "INVESTIGATE")
    log.info("  Check 2 (Value cross-check):    %s", "PASS" if check2_pass else "INVESTIGATE")
    log.info("  Check 3 (Coverage validation):   %s", "PASS" if check3_pass else "INVESTIGATE")
    log.info("  OVERALL: %s", "PASS" if all_pass else "REVIEW NEEDED")

    # =====================================================================
    # Write report
    # =====================================================================
    report = {
        "validation": "geocert_vs_hhs_places_2023",
        "timestamp": timestamp,
        "geocert_source": str(gc_path),
        "hhs_source": HF_DATASET,
        "check_1_zcta_reconciliation": {
            "status": "PASS" if check1_pass else "INVESTIGATE",
            "geocert_count": len(gc_zctas),
            "hhs_count": len(hhs_zctas),
            "in_both": len(in_both),
            "geocert_only": sorted(gc_only),
            "hhs_only_count": len(hhs_only),
            "hhs_only_sample": sorted(hhs_only)[:20],
        },
        "check_2_value_crosscheck": {
            "status": "PASS" if check2_pass else "INVESTIGATE",
            "n_targets_checked": len([v for v in value_checks.values() if v.get("status") != "SKIP"]),
            "n_pass": len([v for v in value_checks.values() if v.get("status", "").startswith("PASS")]),
            "n_warn": len([v for v in value_checks.values() if v.get("status") == "WARN"]),
            "n_fail": len([v for v in value_checks.values() if v.get("status") == "FAIL"]),
            "min_spearman_rho": round(min(all_correlations), 6) if all_correlations else None,
            "mean_spearman_rho": round(float(np.mean(all_correlations)), 6) if all_correlations else None,
            "max_abs_diff_any_column": round(max(all_max_diffs), 6) if all_max_diffs else None,
            "per_column": value_checks,
        },
        "check_3_coverage_validation": {
            "status": "PASS" if check3_pass else "INVESTIGATE",
            "geocert_missing_cdc_count": len(gc_missing_cdc),
            "absent_from_hhs_entirely": len(missing_in_hhs),
            "in_hhs_but_null": len(confirmed_missing),
            "false_negatives": sorted(false_negatives),
            "reverse_problems": sorted(reverse_problems)[:20],
        },
        "overall": "PASS" if all_pass else "REVIEW NEEDED",
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("")
    log.info("Validation report saved: %s", out_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
