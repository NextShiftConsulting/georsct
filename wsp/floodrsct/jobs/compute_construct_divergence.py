#!/usr/bin/env python3
"""compute_construct_divergence.py -- DOE-C1: FAST vs NFIP construct divergence.

Computes spatial correlation and quadrant mass distribution between FAST
engineering damage estimates and NFIP administrative claims at the ZCTA level,
per (scenario, event).

DOE-C1 assertion: FAST and NFIP encode different spatial constructs.
Pass condition: correlation significantly different from +1.0.
Demotion: rho > 0.5 (positive and strong).

See: PREREG_floodcaster_story_doe.md (locked 2026-06-11)

Usage:
    python compute_construct_divergence.py --upload      # run + upload to S3
    python compute_construct_divergence.py --dry-run     # plan only
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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

RESULTS_PREFIX = "results/s035/story"

# Event-to-return-period mapping (from fast_validation.json)
EVENT_RP_MAP = {
    "houston": {
        "harvey2017": "1000yr",
        "imelda2019": "200yr",
        "beryl2024": "25yr",
    },
    "nyc": {
        "ida2021": "100yr",
        "henri2021": "1yr",
    },
    "southwest_florida": {
        "ian2022": "cat4",
        "helene2024": "cat4",
        "milton2024": "cat3",
    },
}

# Scenarios with FAST data available
FAST_SCENARIOS = ["houston", "southwest_florida", "nyc"]

# FAST signal columns
FAST_COLS = [
    "fast_total_loss_usd",
    "fast_mean_loss_per_sqft",
    "fast_pct_damaged",
]

# NFIP signal columns
NFIP_COLS = [
    "nfip_event_claim_count",
    "nfip_event_total_loss",
]


def _load_parquet(s3, key: str) -> pd.DataFrame | None:
    """Load a parquet from S3, return None on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def _load_fast_zcta(s3, scenario: str, rp: str) -> pd.DataFrame | None:
    """Load FAST ZCTA aggregates for a scenario + return period."""
    key = f"processed/{scenario}/{scenario}_fast_zcta_{rp}.parquet"
    df = _load_parquet(s3, key)
    if df is None:
        return None
    # FAST parquets use 'zcta' as index name; normalize to column
    if df.index.name == "zcta":
        df = df.reset_index().rename(columns={"zcta": "zcta_id"})
    elif "zcta" in df.columns:
        df = df.rename(columns={"zcta": "zcta_id"})
    df["zcta_id"] = df["zcta_id"].astype(str)
    return df


def _load_event_features(s3, scenario: str) -> pd.DataFrame | None:
    """Load event features parquet (contains NFIP columns)."""
    # Use the same key mapping as _coverage_common
    key_map = {
        "houston": "processed/houston/houston_event_features.parquet",
        "nyc": "processed/nyc/nyc_event_features.parquet",
        "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
    }
    key = key_map.get(scenario)
    if key is None:
        return None
    df = _load_parquet(s3, key)
    if df is not None:
        df["zcta_id"] = df["zcta_id"].astype(str)
    return df


def compute_correlation(fast_vals, nfip_vals):
    """Compute Spearman and Kendall correlations with p-values."""
    mask = np.isfinite(fast_vals) & np.isfinite(nfip_vals)
    n = int(mask.sum())
    if n < 10:
        return {"n": n, "error": "insufficient overlap (n < 10)"}

    f = fast_vals[mask]
    nf = nfip_vals[mask]

    sp_rho, sp_p = stats.spearmanr(f, nf)
    kt_tau, kt_p = stats.kendalltau(f, nf)

    return {
        "n": n,
        "spearman_rho": float(sp_rho),
        "spearman_p": float(sp_p),
        "kendall_tau": float(kt_tau),
        "kendall_p": float(kt_p),
    }


def compute_quadrant_mass(fast_vals, nfip_vals):
    """Compute 2x2 quadrant mass distribution (high/low FAST x high/low NFIP).

    Thresholds: median of each variable (within the overlapping set).
    """
    mask = np.isfinite(fast_vals) & np.isfinite(nfip_vals)
    n = int(mask.sum())
    if n < 10:
        return {"n": n, "error": "insufficient overlap"}

    f = fast_vals[mask]
    nf = nfip_vals[mask]

    f_med = float(np.median(f))
    nf_med = float(np.median(nf))

    f_hi = f >= f_med
    nf_hi = nf >= nf_med

    hh = int((f_hi & nf_hi).sum())
    hl = int((f_hi & ~nf_hi).sum())
    lh = int((~f_hi & nf_hi).sum())
    ll = int((~f_hi & ~nf_hi).sum())

    return {
        "n": n,
        "fast_median": f_med,
        "nfip_median": nf_med,
        "high_fast_high_nfip": hh,
        "high_fast_low_nfip": hl,
        "low_fast_high_nfip": lh,
        "low_fast_low_nfip": ll,
        "pct_agreement": float(hh + ll) / n,
        "pct_physical_dominant": float(hl) / n,
        "pct_claims_dominant": float(lh) / n,
    }


def process_scenario(s3, scenario: str) -> dict:
    """Process all events for a scenario."""
    events = EVENT_RP_MAP.get(scenario, {})
    if not events:
        return {"scenario": scenario, "error": "no event-RP mapping"}

    event_features = _load_event_features(s3, scenario)
    if event_features is None:
        return {"scenario": scenario, "error": "could not load event features"}

    event_results = []

    for event, rp in events.items():
        log.info("  %s / %s -> RP %s", scenario, event, rp)

        # Load FAST for this RP
        fast_df = _load_fast_zcta(s3, scenario, rp)
        if fast_df is None:
            event_results.append({
                "event": event,
                "return_period": rp,
                "error": f"FAST parquet missing for {rp}",
            })
            continue

        # Filter NFIP to this event
        nfip_df = event_features[event_features["event"] == event].copy()
        if nfip_df.empty:
            event_results.append({
                "event": event,
                "return_period": rp,
                "error": f"no NFIP rows for event {event}",
            })
            continue

        # Merge on zcta_id (inner join = overlapping ZCTAs)
        merged = pd.merge(
            nfip_df[["zcta_id"] + [c for c in NFIP_COLS if c in nfip_df.columns]],
            fast_df[["zcta_id"] + [c for c in FAST_COLS if c in fast_df.columns]],
            on="zcta_id",
            how="inner",
        )

        n_nfip = len(nfip_df)
        n_fast = len(fast_df)
        n_overlap = len(merged)

        log.info("    ZCTAs: NFIP=%d, FAST=%d, overlap=%d", n_nfip, n_fast, n_overlap)

        if n_overlap < 10:
            event_results.append({
                "event": event,
                "return_period": rp,
                "n_nfip_zctas": n_nfip,
                "n_fast_zctas": n_fast,
                "n_overlap": n_overlap,
                "error": "insufficient overlap (n < 10)",
            })
            continue

        # Primary: FAST total loss vs NFIP claim count
        primary_corr = compute_correlation(
            merged["fast_total_loss_usd"].values,
            merged["nfip_event_claim_count"].values,
        )

        # Secondary: all pairwise FAST x NFIP correlations
        pairwise = {}
        for fc in FAST_COLS:
            if fc not in merged.columns:
                continue
            for nc in NFIP_COLS:
                if nc not in merged.columns:
                    continue
                pairwise[f"{fc}_vs_{nc}"] = compute_correlation(
                    merged[fc].values, merged[nc].values,
                )

        # Quadrant mass (primary pair)
        quadrant = compute_quadrant_mass(
            merged["fast_total_loss_usd"].values,
            merged["nfip_event_claim_count"].values,
        )

        event_results.append({
            "event": event,
            "return_period": rp,
            "n_nfip_zctas": n_nfip,
            "n_fast_zctas": n_fast,
            "n_overlap": n_overlap,
            "primary_correlation": primary_corr,
            "pairwise_correlations": pairwise,
            "quadrant_mass": quadrant,
        })

    # Scenario-level summary
    rhos = [
        r["primary_correlation"]["spearman_rho"]
        for r in event_results
        if "primary_correlation" in r and "spearman_rho" in r["primary_correlation"]
    ]

    summary = {
        "n_events_computed": len(rhos),
        "n_events_total": len(events),
    }
    if rhos:
        summary["mean_spearman_rho"] = float(np.mean(rhos))
        summary["min_spearman_rho"] = float(np.min(rhos))
        summary["max_spearman_rho"] = float(np.max(rhos))
        summary["all_negative"] = all(r < 0 for r in rhos)
        summary["any_above_0_5"] = any(r > 0.5 for r in rhos)
        # DOE-C1 pass condition: rho significantly different from +1.0
        # Demotion: rho > 0.5
        if summary["any_above_0_5"]:
            summary["doe_verdict"] = "DEMOTION"
            summary["doe_reason"] = "rho > 0.5 for at least one event"
        else:
            summary["doe_verdict"] = "PASS"
            summary["doe_reason"] = "rho not strongly positive; constructs diverge"

    return {
        "scenario": scenario,
        "events": event_results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DOE-C1: FAST vs NFIP construct divergence"
    )
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Plan only")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would compute construct divergence for %s", FAST_SCENARIOS)
        return 0

    s3 = get_s3_client()

    all_results = []
    for scenario in FAST_SCENARIOS:
        log.info("Processing scenario: %s", scenario)
        result = process_scenario(s3, scenario)
        all_results.append(result)

        # Upload per-scenario result
        if args.upload:
            key = f"{RESULTS_PREFIX}/construct_divergence_{scenario}.json"
            payload = {
                "doe_id": "DOE-C1",
                "phase": "construct_divergence",
                "version": "1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "assertion": "FAST and NFIP encode different spatial constructs",
                **result,
            }
            upload_json_result(s3, BUCKET, key, payload)
            log.info("Uploaded to s3://%s/%s", BUCKET, key)

    # Cross-scenario summary
    verdicts = [
        r["summary"].get("doe_verdict")
        for r in all_results
        if "summary" in r and "doe_verdict" in r["summary"]
    ]
    all_rhos = []
    for r in all_results:
        for ev in r.get("events", []):
            if "primary_correlation" in ev:
                rho = ev["primary_correlation"].get("spearman_rho")
                if rho is not None:
                    all_rhos.append(rho)

    cross_scenario = {
        "doe_id": "DOE-C1",
        "phase": "construct_divergence",
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios_computed": FAST_SCENARIOS,
        "scenarios_without_fast": ["riverside_coachella", "new_orleans"],
        "per_scenario": all_results,
        "cross_scenario_summary": {
            "n_events_total": len(all_rhos),
            "mean_rho": float(np.mean(all_rhos)) if all_rhos else None,
            "min_rho": float(np.min(all_rhos)) if all_rhos else None,
            "max_rho": float(np.max(all_rhos)) if all_rhos else None,
            "per_scenario_verdicts": verdicts,
            "overall_verdict": "PASS" if all(v == "PASS" for v in verdicts) else "MIXED",
        },
    }

    # Write local copy
    out_path = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_path.mkdir(parents=True, exist_ok=True)
    local_file = out_path / "construct_divergence.json"
    with open(local_file, "w") as f:
        json.dump(cross_scenario, f, indent=2)
    log.info("Written local copy to %s", local_file)

    # Upload cross-scenario summary
    if args.upload:
        key = f"{RESULTS_PREFIX}/construct_divergence_summary.json"
        upload_json_result(s3, BUCKET, key, cross_scenario)
        log.info("Uploaded summary to s3://%s/%s", BUCKET, key)

    # Print summary table
    print()
    print("=" * 72)
    print("DOE-C1: Construct Divergence Summary")
    print("=" * 72)
    for r in all_results:
        s = r.get("summary", {})
        print(f"  {r['scenario']:25s}  "
              f"mean_rho={s.get('mean_spearman_rho', 'N/A'):>7s}"
              if isinstance(s.get('mean_spearman_rho'), str)
              else f"  {r['scenario']:25s}  "
                   f"mean_rho={s.get('mean_spearman_rho', float('nan')):+.3f}  "
                   f"verdict={s.get('doe_verdict', 'N/A')}")
    print()
    if all_rhos:
        print(f"  Overall: {len(all_rhos)} events, "
              f"rho range [{min(all_rhos):+.3f}, {max(all_rhos):+.3f}], "
              f"mean {np.mean(all_rhos):+.3f}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
