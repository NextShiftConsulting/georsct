#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   generator: deepseek-r1 (DeepSeek, via OpenRouter)
#   cleanup_by: Martin
#   cleanup_summary: Fix flood zone column names (flood_pct_zone_a not
#       flood_zone_pct_AE), fix upload_json_result signature, move S3
#       client to main, add shebang/docstring/logging, fix local path,
#       obs_has_311 is binary not count, add dry-run gate
#   see: ../exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml
# =============================================================================
"""build_zcta_evidence.py -- Phase R4.2: structured text evidence per ZCTA.

Assembles a structured text file for each ZCTA from event features,
county crosswalk, and SVI data. These text files become inputs to
VLM assessment in Phase R4.3.

Usage:
    python build_zcta_evidence.py --scenario houston --upload
    python build_zcta_evidence.py --scenario houston --dry-run
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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
SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str) -> pd.DataFrame | None:
    """Load parquet from S3, return None on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def _fmt(val, fmt: str) -> str:
    """Format a value with fallback to N/A."""
    if pd.isna(val):
        return "N/A"
    if fmt == "pct":
        return f"{val:.1f}%"
    if fmt == "currency":
        return f"${val:,.0f}"
    if fmt == "int":
        return f"{int(val):,}"
    return str(val)


def _svi_label(score: float) -> str:
    """Interpret SVI overall score."""
    if pd.isna(score):
        return "N/A"
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Moderate"
    if score < 0.75:
        return "High"
    return "Very High"


def build_text(row: pd.Series) -> str:
    """Assemble structured text evidence for one ZCTA."""
    county = row.get("county_name", "N/A")
    state = row.get("state_name", "N/A")

    # Flood zones (actual column names from assembled parquets)
    zone_a = _fmt(row.get("flood_pct_zone_a"), "pct")
    zone_x = _fmt(row.get("flood_pct_zone_x"), "pct")
    zone_x500 = _fmt(row.get("flood_pct_zone_x500"), "pct")

    # Demographics
    pop = _fmt(row.get("population"), "int")
    income = _fmt(row.get("median_income"), "currency")
    svi = row.get("svi_overall", float("nan"))
    svi_str = f"{svi:.2f}" if not pd.isna(svi) else "N/A"

    # Historical events
    nfip = _fmt(row.get("obs_nfip_event_claims"), "int")
    has_311 = row.get("obs_has_311")
    reports_311 = "Yes" if has_311 == 1 else ("No" if has_311 == 0 else "N/A")

    return (
        f"ZCTA {row['zcta_id']} in {county}, {state}.\n"
        f"\n"
        f"FEMA Flood Zones:\n"
        f"- {zone_a} in Zone A (1% annual chance floodplain)\n"
        f"- {zone_x} in Zone X (minimal flood hazard)\n"
        f"- {zone_x500} in Zone X500 (0.2% annual chance floodplain)\n"
        f"\n"
        f"Demographics (ACS):\n"
        f"- Population: {pop}, Median income: {income}\n"
        f"- SVI overall: {svi_str} ({_svi_label(svi)})\n"
        f"\n"
        f"Historical Events:\n"
        f"- NFIP claims: {nfip} events\n"
        f"- 311 flood reports: {reports_311}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase R4.2: ZCTA text evidence")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, skip execution")
    args = parser.parse_args()

    scenario = args.scenario

    if args.dry_run:
        log.info("DRY RUN: would build text evidence for %s", scenario)
        log.info("Reads: event features + crosswalk + SVI")
        log.info("Writes: %s/evidence/%s/{zcta_id}.txt", RESULTS_PREFIX, scenario)
        return 0

    s3 = get_s3_client()

    # Load data
    event_df = _load_parquet(s3, f"processed/{scenario}/{scenario}_event_features.parquet")
    crosswalk_df = _load_parquet(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    svi_df = _load_parquet(s3, "raw/geocertdb2026/svi_zcta.parquet")

    if event_df is None:
        log.error("Event features not found for %s", scenario)
        return 1

    # Ensure zcta_id is string for merging
    event_df["zcta_id"] = event_df["zcta_id"].astype(str)
    if crosswalk_df is not None:
        crosswalk_df["zcta_id"] = crosswalk_df["zcta_id"].astype(str)
    if svi_df is not None:
        svi_df["zcta_id"] = svi_df["zcta_id"].astype(str)

    # Merge
    merged = event_df
    if crosswalk_df is not None:
        merged = merged.merge(crosswalk_df, on="zcta_id", how="left")
    if svi_df is not None:
        merged = merged.merge(svi_df, on="zcta_id", how="left", suffixes=("", "_svi"))

    log.info("Building evidence for %d ZCTAs in %s", len(merged), scenario)

    # Output directory
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "evidence" / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    zcta_ids = []
    for _, row in merged.iterrows():
        zcta_id = str(row["zcta_id"])
        text = build_text(row)

        # Write local
        local_path = out_dir / f"{zcta_id}.txt"
        with open(local_path, "w") as f:
            f.write(text)

        # Upload to S3
        if args.upload:
            s3_key = f"{RESULTS_PREFIX}/evidence/{scenario}/{zcta_id}.txt"
            s3.put_object(
                Bucket=BUCKET, Key=s3_key,
                Body=text.encode(), ContentType="text/plain",
            )

        zcta_ids.append(zcta_id)

    log.info("Wrote %d evidence files to %s", len(zcta_ids), out_dir)

    # Manifest
    manifest = {
        "phase": "R4.2_zcta_evidence",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_zctas": len(zcta_ids),
        "zcta_ids": zcta_ids,
    }

    local_manifest = out_dir / f"{scenario}_manifest.json"
    with open(local_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest: %s", local_manifest)

    if args.upload:
        key = f"{RESULTS_PREFIX}/evidence/{scenario}_manifest.json"
        upload_json_result(s3, BUCKET, key, manifest)
        log.info("Uploaded manifest to s3://%s/%s", BUCKET, key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
