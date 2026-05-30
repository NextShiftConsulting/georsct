"""
_coverage_common.py -- Shared utilities for Stage 5 stratified coverage audits.

Each audit script (audit_a1 through audit_a6) imports from here to load
processed parquets and write evidence JSON. The scripts themselves are
independent -- each can run standalone.
"""

import io
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import boto3
import pandas as pd
from swarm_auth import get_aws_credentials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

BUCKET = "swarm-floodrsct-data"

OUTPUT_KEYS = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}

SCENARIOS = list(OUTPUT_KEYS.keys())


def get_s3_client():
    """Create S3 client via swarm_auth."""
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    return boto3.client("s3", region_name="us-east-1", **_aws)


def load_processed_parquet(s3, scenario: str) -> pd.DataFrame:
    """Load a scenario's processed parquet from S3."""
    key = OUTPUT_KEYS[scenario]
    log = logging.getLogger("coverage")
    log.info("Loading s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    df = pd.read_parquet(buf)
    log.info("Loaded %d rows x %d columns", len(df), len(df.columns))
    return df


def load_crosswalk(s3) -> pd.DataFrame:
    """Load ZCTA-county crosswalk from geocertdb2026."""
    key = "raw/geocertdb2026/zcta_county_crosswalk.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    return pd.read_parquet(buf)


def load_adjacency(s3) -> pd.DataFrame:
    """Load ZCTA adjacency (Queen's contiguity) edge list."""
    # Try known locations
    for key in [
        "raw/geocertdb2026/zcta_adjacency.parquet",
        "raw/geocert/zcta_adjacency.parquet",
    ]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            buf = io.BytesIO(resp["Body"].read())
            return pd.read_parquet(buf)
        except s3.exceptions.ClientError:
            continue
    raise FileNotFoundError("zcta_adjacency.parquet not found on S3")


@dataclass
class AuditResult:
    """Result of a single coverage audit check.

    Fields align with the data-lock manifest schema:
      audit_id:        GeoRSCT mode (mode_A1, mode_B1, ...) or support probe (P1-P6)
      mode:            failure mode name (leakage, maup, crs, ...)
      scenario:        scenario key (houston, nyc, ...)
      event:           event key or None for scenario-level audits
      status:          PASS | WARN | FAIL | NOT_READY | SKIP
      primary_metric:  name of the key metric tested (e.g. leakage_rate, variance_ratio)
      metric_value:    numeric value of primary_metric, or None
      threshold:       decision threshold for primary_metric, or None
      recommendation:  actionable next step
      detail:          full evidence dict
      min_support:     minimum sample size for check
      timestamp:       ISO-8601 timestamp
      evidence_path:   S3 key where evidence JSON was written (set by write_evidence)
    """
    audit_id: str
    mode: str
    scenario: str
    status: str  # PASS | WARN | FAIL | NOT_READY | SKIP
    detail: dict
    min_support: int
    timestamp: str
    event: str = ""
    primary_metric: str = ""
    metric_value: float | None = None
    threshold: float | None = None
    recommendation: str = ""
    evidence_path: str = ""


def write_evidence(results: list[AuditResult], audit_name: str, scenario: str,
                   s3=None, upload: bool = False) -> None:
    """Print and optionally upload audit evidence."""
    key = f"evidence/qa/coverage_{audit_name}_{scenario}.json"

    # Set evidence_path on each result
    for r in results:
        r.evidence_path = f"s3://{BUCKET}/{key}"

    payload = {
        "audit": audit_name,
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "pass": sum(1 for r in results if r.status == "PASS"),
            "warn": sum(1 for r in results if r.status == "WARN"),
            "fail": sum(1 for r in results if r.status == "FAIL"),
            "not_ready": sum(1 for r in results if r.status == "NOT_READY"),
            "skip": sum(1 for r in results if r.status == "SKIP"),
        },
    }

    print(json.dumps(payload, indent=2))

    if upload and s3:
        s3.put_object(
            Bucket=BUCKET, Key=key,
            Body=json.dumps(payload, indent=2).encode(),
            ContentType="application/json",
        )
        logging.getLogger("coverage").info(
            "Uploaded s3://%s/%s", BUCKET, key
        )
