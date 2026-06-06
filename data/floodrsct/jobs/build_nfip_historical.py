#!/usr/bin/env python3
"""
build_nfip_historical.py -- Build temporally-gated NFIP historical features.

For each (scenario, event), aggregates all NFIP claims with dateOfLoss
strictly BEFORE the event's incidentBeginDate.  This enforces the IBNR
(Incurred But Not Reported) temporal boundary: historical loss development
informs the prior, but same-event claims are excluded.

Inputs:
  s3://swarm-floodrsct-data/raw/openfema/nfip_claims_dr{number}.parquet
  Scenario configs (events + start_date per event)

Outputs:
  s3://swarm-floodrsct-data/processed/{scenario}/{scenario}_nfip_historical.parquet
    Columns: zcta_id, event, nfip_historical_frequency, nfip_historical_severity

Usage:
    python build_nfip_historical.py --scenario houston --upload
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

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, load_processed_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "processed"

# Map scenario -> list of {event_name, dr_number, start_date}.
# start_date is the temporal gate: only claims with dateOfLoss < start_date
# are included in the historical features for that event.
SCENARIO_EVENTS = {
    "houston": [
        {"event": "harvey2017",  "dr": 4332, "start_date": "2017-08-17"},
        {"event": "imelda2019",  "dr": 4466, "start_date": "2019-09-17"},
        {"event": "beryl2024",   "dr": 4781, "start_date": "2024-07-08"},
    ],
    "new_orleans": [
        {"event": "katrina2005", "dr": 1603, "start_date": "2005-08-29"},
        {"event": "isaac2012",   "dr": 4080, "start_date": "2012-08-28"},
        {"event": "barry2019",   "dr": 4458, "start_date": "2019-07-11"},
        {"event": "ida2021",     "dr": 4611, "start_date": "2021-08-26"},
    ],
    "nyc": [
        {"event": "sandy2012",      "dr": 4085, "start_date": "2012-10-29"},
        {"event": "henri2021",      "dr": 4615, "start_date": "2021-08-22"},
        {"event": "ida2021",        "dr": 4615, "start_date": "2021-09-01"},
        {"event": "nyc_flood_2023", "dr": 4755, "start_date": "2023-09-29"},
    ],
    "riverside_coachella": [
        {"event": "hilary2023",    "dr": 4699, "start_date": "2023-08-20"},
        {"event": "ar_flood_2023", "dr": 4699, "start_date": "2023-01-09"},
    ],
    "southwest_florida": [
        {"event": "ian2022",     "dr": 4673, "start_date": "2022-09-23"},
        {"event": "helene2024",  "dr": 4828, "start_date": "2024-09-24"},
        {"event": "milton2024",  "dr": 4834, "start_date": "2024-10-07"},
    ],
}


def _discover_all_dr_parquets(s3) -> list[int]:
    """Scan S3 for all available NFIP claims parquets (any DR)."""
    import re
    dr_numbers = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="raw/openfema/nfip_claims_dr"):
        for obj in page.get("Contents", []):
            m = re.search(r"nfip_claims_dr(\d+)\.parquet$", obj["Key"])
            if m:
                dr_numbers.append(int(m.group(1)))
    return sorted(dr_numbers)


def load_all_claims(s3) -> pd.DataFrame:
    """Load and concatenate all per-DR NFIP claim parquets from S3.

    Scans S3 for all available DR parquets rather than using a hardcoded
    list, so that pre-event historical claims are always available even
    for single-event scenarios.
    """
    all_drs = _discover_all_dr_parquets(s3)
    log.info("Discovered %d DR parquets on S3: %s", len(all_drs), all_drs)
    frames = []
    for dr in all_drs:
        key = f"raw/openfema/nfip_claims_dr{dr}.parquet"
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
            df["dr_number"] = dr
            frames.append(df)
            log.info("DR-%d: %d claims loaded", dr, len(df))
        except s3.exceptions.NoSuchKey:
            log.warning("DR-%d: no claims parquet at %s", dr, key)
    if not frames:
        raise RuntimeError("No NFIP claims parquets found on S3")
    combined = pd.concat(frames, ignore_index=True)
    # Normalize zcta_id and dateOfLoss
    combined["zcta_id"] = combined["zcta_id"].astype(str).str.zfill(5)
    combined["dateOfLoss"] = pd.to_datetime(combined["dateOfLoss"], errors="coerce")
    log.info("Total claims corpus: %d rows across %d DRs", len(combined), len(frames))
    return combined


def compute_historical_features(
    all_claims: pd.DataFrame,
    zcta_ids: list[str],
    cutoff_date: str,
) -> pd.DataFrame:
    """Aggregate claims with dateOfLoss < cutoff_date per ZCTA.

    Returns DataFrame with columns:
        zcta_id, nfip_historical_frequency, nfip_historical_severity
    """
    cutoff = pd.Timestamp(cutoff_date)
    # Ensure cutoff matches dateOfLoss timezone (may be tz-aware UTC from parquet)
    if hasattr(all_claims["dateOfLoss"].dtype, "tz") and all_claims["dateOfLoss"].dtype.tz is not None:
        cutoff = cutoff.tz_localize(all_claims["dateOfLoss"].dtype.tz)
    historical = all_claims[all_claims["dateOfLoss"] < cutoff].copy()
    log.info(
        "Claims before %s: %d / %d total",
        cutoff_date, len(historical), len(all_claims),
    )

    # Determine the loss column (amountPaidOnBuildingClaim is the standard field)
    loss_col = None
    for candidate in ("amountPaidOnBuildingClaim", "totalBuildingInsuranceCoverage"):
        if candidate in historical.columns:
            loss_col = candidate
            break

    if historical.empty or loss_col is None:
        # Return zeros for all ZCTAs
        return pd.DataFrame({
            "zcta_id": zcta_ids,
            "nfip_historical_frequency": 0,
            "nfip_historical_severity": 0.0,
        })

    historical[loss_col] = pd.to_numeric(historical[loss_col], errors="coerce").fillna(0.0)

    agg = (
        historical
        .groupby("zcta_id")
        .agg(
            nfip_historical_frequency=(loss_col, "count"),
            _total_loss=(loss_col, "sum"),
        )
        .reset_index()
    )
    agg["nfip_historical_severity"] = np.where(
        agg["nfip_historical_frequency"] > 0,
        agg["_total_loss"] / agg["nfip_historical_frequency"],
        0.0,
    )
    agg = agg.drop(columns=["_total_loss"])

    # Ensure all ZCTAs are present (fill missing with 0)
    result = (
        pd.DataFrame({"zcta_id": zcta_ids})
        .merge(agg, on="zcta_id", how="left")
        .fillna({"nfip_historical_frequency": 0, "nfip_historical_severity": 0.0})
    )
    result["nfip_historical_frequency"] = result["nfip_historical_frequency"].astype(int)
    return result


def build_scenario(s3, scenario: str, upload: bool) -> None:
    """Build NFIP historical features for all events in a scenario."""
    events = SCENARIO_EVENTS[scenario]
    log.info("Scenario %s: %d events", scenario, len(events))

    # Load the assembled parquet to get the ZCTA list per event
    assembled = load_processed_parquet(s3, scenario)
    all_claims = load_all_claims(s3)

    frames = []
    for ev in events:
        event_name = ev["event"]
        cutoff_date = ev["start_date"]

        # Get ZCTAs for this event
        if "event" in assembled.columns:
            event_zctas = assembled.loc[
                assembled["event"] == event_name, "zcta_id"
            ].unique().tolist()
        else:
            event_zctas = assembled["zcta_id"].unique().tolist()

        if not event_zctas:
            log.warning("No ZCTAs for event %s in %s", event_name, scenario)
            continue

        log.info(
            "Event %s: %d ZCTAs, cutoff %s",
            event_name, len(event_zctas), cutoff_date,
        )
        feat = compute_historical_features(all_claims, event_zctas, cutoff_date)
        feat["event"] = event_name
        frames.append(feat)

        # Log summary stats
        freq = feat["nfip_historical_frequency"]
        sev = feat["nfip_historical_severity"]
        log.info(
            "  frequency: mean=%.1f, median=%.0f, max=%d, zeros=%d/%d",
            freq.mean(), freq.median(), freq.max(),
            (freq == 0).sum(), len(freq),
        )
        log.info(
            "  severity: mean=$%.0f, median=$%.0f, max=$%.0f",
            sev.mean(), sev.median(), sev.max(),
        )

    if not frames:
        raise RuntimeError(f"No events produced features for {scenario}")

    result = pd.concat(frames, ignore_index=True)
    result = result[["zcta_id", "event", "nfip_historical_frequency", "nfip_historical_severity"]]
    log.info("Final output: %d rows x %d cols", len(result), len(result.columns))

    if upload:
        key = f"{RESULTS_PREFIX}/{scenario}/{scenario}_nfip_historical.parquet"
        buf = io.BytesIO()
        result.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        log.info("Uploaded to s3://%s/%s", BUCKET, key)
    else:
        log.info("[NO UPLOAD] Would write to %s/%s_nfip_historical.parquet", scenario, scenario)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NFIP historical features")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()
    build_scenario(s3, args.scenario, args.upload)
    log.info("build_nfip_historical complete for %s", args.scenario)


if __name__ == "__main__":
    main()
