"""build_flood_georsct_table.py -- R-080 T1: flood georsct feature table.

Adapts the existing v24 assembled table (`zcta_features_labels.parquet`, ~106 cols,
31,789 ZCTAs) into the FLOOD certification table the exporter + certifier expect.
This is a SELECT-and-RENAME adapter, NOT a from-scratch feature build: the source
already carries per-ZCTA TWI, NOAA flood events, NFIP claims, and SVI. It maps those
to the certifier's exact modal-feature contract and attaches the flood target.

Certifier contract (swarm-it-api/engine/geo_certifier.py::_MODAL_FEATURE_PREFIXES):
    twi         -> flood_twi, flood_spi, flood_gfi, flood_hand
    noaa_events -> flood_noaa, flood_storm
    nfip_claims -> flood_nfip
    (+ svi_* passthrough)
Target (exporter reads `target_{task}`): target_obs_nfip_event_claims
    -- leakage already handled upstream (event-year excluded from historical nfip_* per
       build_nfip_claims.py); do NOT add nfip_claim_count as a feature here.

PHASE-1 SCOPE: maps the modal features that have a direct v24 source
(flood_twi, flood_noaa, flood_storm, flood_nfip + svi_). The hydrology-modal
flood_spi / flood_gfi / flood_hand have NO direct v24 source and are a documented
FOLLOW-ON (StreamCat SPI + a national HAND-per-ZCTA build); a subset flood table is a
valid certifier input. This adapter does NOT calibrate -- calibration soundness
(ACCEPTANCE_SPEC.md Layer A) is validated at export time and must not be rushed.

Usage:
    python build_flood_georsct_table.py --dry-run                 # local, no upload
    python build_flood_georsct_table.py --upload                  # build + upload to S3
    python build_flood_georsct_table.py --local-dir /tmp/geo      # use local input
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
REGION = "us-east-1"
INPUT_KEY = "rsct_curriculum/series_018/processed/zcta_features_labels.parquet"
OUTPUT_KEY = "rsct_curriculum/flood/processed/zcta_flood_features_labels.parquet"
PROVENANCE_KEY = "rsct_curriculum/flood/processed/zcta_flood_features_labels_provenance.json"

# Certifier-contract feature name -> ordered candidate source columns in the v24 table.
# The first present candidate wins (robust to exact naming); a missing modal feature is
# WARNED and skipped (subset table is valid). SPI/GFI/HAND have no v24 source (follow-on).
FEATURE_MAP: dict[str, list[str]] = {
    "flood_twi":   ["twi_mean", "twi_twi", "terrain_flood_index"],
    "flood_noaa":  ["flood_event_count"],
    "flood_storm": ["flood_property_damage_k", "flood_deaths"],
    "flood_nfip":  ["nfip_claim_count", "nfip_total_loss"],
}
# Certifier-contract modal features with NO v24 source yet -- explicitly recorded as a
# gap in provenance (never silently dropped). Follow-on: StreamCat SPI + national HAND.
FOLLOWON_FEATURES: list[str] = ["flood_spi", "flood_gfi", "flood_hand"]
TARGET_CANDIDATES = ["target_obs_nfip_event_claims", "obs_nfip_event_claims"]
TARGET_OUT = "target_obs_nfip_event_claims"
ID_CANDIDATES = ["zcta_id", "ZCTA", "zcta"]


def _s3():
    return boto3.client("s3", **get_aws_credentials())


def _load_input(local_dir: Path | None) -> pd.DataFrame:
    if local_dir is not None:
        path = local_dir / Path(INPUT_KEY).name
        log.info("Reading local input: %s", path)
        return pd.read_parquet(path)
    import tempfile
    tmp = Path(tempfile.gettempdir()) / Path(INPUT_KEY).name
    log.info("Downloading s3://%s/%s", BUCKET, INPUT_KEY)
    _s3().download_file(BUCKET, INPUT_KEY, str(tmp))
    return pd.read_parquet(tmp)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select + rename the v24 table to the flood certifier contract."""
    id_col = _first_present(df, ID_CANDIDATES)
    if id_col is None:
        raise SystemExit(f"FATAL: no zcta id column in {list(df.columns)[:8]}...")

    out = pd.DataFrame({"zcta_id": df[id_col].values})
    mapping: dict[str, str] = {}
    missing: list[str] = []

    for contract_name, candidates in FEATURE_MAP.items():
        src = _first_present(df, candidates)
        if src is None:
            missing.append(contract_name)
            log.warning("  MISSING modal feature %s (candidates %s) -- skipped",
                        contract_name, candidates)
            continue
        out[contract_name] = df[src].values
        mapping[contract_name] = src
        log.info("  %s <- %s", contract_name, src)

    svi_cols = sorted(c for c in df.columns if c.startswith("svi_"))
    for c in svi_cols:
        out[c] = df[c].values
    log.info("  svi_ passthrough: %d columns", len(svi_cols))

    tgt = _first_present(df, TARGET_CANDIDATES)
    if tgt is None:
        raise SystemExit(f"FATAL: target not found (looked for {TARGET_CANDIDATES})")
    out[TARGET_OUT] = df[tgt].values
    log.info("  %s <- %s", TARGET_OUT, tgt)

    prov = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_key": INPUT_KEY,
        "source_rows": int(len(df)),
        "out_rows": int(len(out)),
        "feature_mapping": mapping,
        "svi_columns": svi_cols,
        "target": {"out": TARGET_OUT, "source": tgt},
        "mapped_but_absent": missing,
        "followon_not_yet_sourced": FOLLOWON_FEATURES,
        "note": "SELECT+RENAME adapter; no calibration. flood_spi/gfi/hand = follow-on "
                "(StreamCat SPI + national HAND-per-ZCTA). Validate calibration soundness "
                "at export (ACCEPTANCE_SPEC Layer A) -- do not rush.",
    }
    return out, prov


def _summarize(out: pd.DataFrame, prov: dict) -> None:
    flood_cols = sorted(c for c in out.columns if c.startswith("flood_"))
    svi_cols = sorted(c for c in out.columns if c.startswith("svi_"))
    log.info("\n=== FLOOD GEORSCT TABLE ===")
    log.info("  Rows:            %d", len(out))
    log.info("  flood_ features: %s", flood_cols)
    log.info("  svi_ features:   %d", len(svi_cols))
    log.info("  target:          %s", TARGET_OUT)
    log.info("  follow-on gaps:  %s", prov["followon_not_yet_sourced"])
    # non-null target coverage (a determination needs a label)
    nn = int(out[TARGET_OUT].notna().sum())
    log.info("  target non-null: %d/%d (%.1f%%)", nn, len(out), 100.0 * nn / max(len(out), 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R-080 T1: build the flood georsct table")
    ap.add_argument("--dry-run", action="store_true", help="build locally, do NOT upload")
    ap.add_argument("--upload", action="store_true", help="build and upload to S3")
    ap.add_argument("--local-dir", type=str, default=None, help="read input from this dir")
    ap.add_argument("--output", type=str, default="/tmp/zcta_flood_features_labels.parquet")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.upload):
        ap.error("pass --dry-run or --upload")

    local_dir = Path(args.local_dir) if args.local_dir else None
    df = _load_input(local_dir)
    log.info("Loaded input: %d rows, %d columns", len(df), len(df.columns))

    out, prov = build(df)
    _summarize(out, prov)

    out_path = Path(args.output)
    out.to_parquet(out_path, index=False)
    blob = out_path.read_bytes()
    prov["artifact_sha256"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    log.info("  wrote %s (%s)", out_path, prov["artifact_sha256"])

    if args.upload:
        s3 = _s3()
        s3.upload_file(str(out_path), BUCKET, OUTPUT_KEY)
        s3.put_object(Bucket=BUCKET, Key=PROVENANCE_KEY,
                      Body=json.dumps(prov, indent=2).encode(), ContentType="application/json")
        log.info("  uploaded -> s3://%s/%s (+provenance)", BUCKET, OUTPUT_KEY)
    else:
        log.info("  DRY-RUN: not uploaded. Provenance:\n%s", json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
