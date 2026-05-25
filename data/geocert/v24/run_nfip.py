#!/usr/bin/env python3
"""
run_nfip.py -- SageMaker run script for FEMA NFIP claims enrichment.

Designed for ml.m5.xlarge (4 vCPU, 16 GB RAM). Expected runtime: 25-40 min.

NFIP dataset: ~2.4M claims nationally, ~600 MB uncompressed.
Strategy:
  1. Stream OpenFEMA bulk CSV directly to disk in pages of 10K records
  2. Accumulate per-ZIP code aggregates in a rolling dict (no full DataFrame in memory)
  3. Normalize ZIP -> ZCTA (direct 1:1 for residential ZIPs)
  4. Zero-fill all 31K ZCTAs from crosswalk
  5. Upload parquet + provenance directly to S3

Memory optimization:
  - Streaming page accumulation: peak ~200MB (one page + running dict)
  - No pandas concat of full 2.4M records
  - Checkpoint every 50 pages to S3

Crash recovery:
  - Checkpoint saves running aggregation dict as JSON
  - On restart, resumes from last page offset

S3 output:
  s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/nfip_claims_zcta.parquet
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import boto3
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
for _h in logging.root.handlers:
    _h.flush = lambda _orig=_h.flush: (_orig(), sys.stdout.flush())
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_BUCKET = "swarm-yrsn-datasets"
S3_OUTPUT_KEY = "rsct_curriculum/series_018/processed/nfip_claims_zcta.parquet"
S3_PROVENANCE_KEY = "rsct_curriculum/series_018/processed/nfip_claims_provenance.json"
S3_CHECKPOINT_KEY = "rsct_curriculum/series_018/processed/nfip_checkpoint.json"
S3_LONG_KEY = "rsct_curriculum/series_018/processed/nfip_claims_long.parquet"
S3_WIDE_KEY = "rsct_curriculum/series_018/processed/nfip_claims_wide.parquet"
DATA_PREFIX = "rsct_curriculum/series_018/processed"

# Temporal epochs — must match NOAA epochs for cross-source joins
EPOCHS = {
    "e1": (1996, 2004),   # pre-Katrina baseline
    "e2": (2005, 2011),   # post-Katrina / pre-Sandy
    "e3": (2012, 2024),   # Sandy onward
}

OPENFEMA_URL = "https://www.fema.gov/api/open/v2/FimaNfipClaims"  # case-sensitive
PAGE_SIZE = 10_000
MAX_RETRIES = 4
RETRY_BASE_DELAY = 10  # seconds

SELECT_COLS = [
    "reportedZipCode",
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "yearOfLoss",
    "primaryResidenceIndicator",  # renamed from primaryResidence in 2024 API update
]

# ---------------------------------------------------------------------------
# S3 helpers (mirrors run_flood_zones.py pattern)
# ---------------------------------------------------------------------------
def _s3():
    return boto3.client("s3")


def _s3_upload(local_path: str, key: str, quiet: bool = False):
    try:
        _s3().upload_file(local_path, S3_BUCKET, key)
        if not quiet:
            log.info("  -> s3://%s/%s", S3_BUCKET, key)
    except Exception as e:
        log.warning("  S3 upload failed for %s: %s", key, e)


def _s3_download(key: str, local_path: str) -> bool:
    try:
        _s3().download_file(S3_BUCKET, key, local_path)
        log.info("  <- s3://%s/%s", S3_BUCKET, key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# OpenFEMA streaming fetch
# ---------------------------------------------------------------------------
def get_total_count() -> int:
    """Pre-flight count from OpenFEMA metadata."""
    try:
        resp = requests.get(
            "https://www.fema.gov/api/open/v2/FimaNfipClaims",
            params={"$top": 1, "$inlinecount": "allpages"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("metadata", {}).get("count", 0)
    except Exception as exc:
        log.warning("Could not get count: %s", exc)
        return 0


def fetch_page(offset: int) -> pd.DataFrame:
    """Fetch one page. Retries with exponential backoff."""
    params = {
        "$format": "csv",
        "$top": PAGE_SIZE,
        "$skip": offset,
        "$select": ",".join(SELECT_COLS),
        "$orderby": "reportedZipCode asc",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(OPENFEMA_URL, params=params, timeout=180)
            resp.raise_for_status()
            # API returns CSV text; first line may be headers
            if not resp.text.strip() or resp.text.strip().startswith("<"):
                return pd.DataFrame()
            df = pd.read_csv(StringIO(resp.text), dtype=str, low_memory=False)
            return df
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                log.error("Page offset=%d failed after %d attempts: %s",
                          offset, MAX_RETRIES, exc)
                return pd.DataFrame()
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning("  Retry %d/%d offset=%d in %.0fs: %s",
                        attempt + 1, MAX_RETRIES, offset, delay, exc)
            time.sleep(delay)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Rolling accumulation (no full DataFrame in memory)
# ---------------------------------------------------------------------------
def accumulate_page(
    page: pd.DataFrame,
    agg: dict,                  # {zip5: {count, building, contents}}
    temporal_agg: dict | None,  # {zip5: {year: {count, building, contents}}} or None
) -> int:
    """Parse one page and roll into running aggregation dicts. Returns paid count."""
    # Normalize ZIP to 5 digits
    page["zip5"] = (
        page["reportedZipCode"]
        .astype(str)
        .str.strip()
        .str.replace(r"\D", "", regex=True)
        .str[:5]
        .str.zfill(5)
    )
    page = page[page["zip5"].str.match(r"^\d{5}$") & (page["zip5"] != "00000")]

    for col in ("amountPaidOnBuildingClaim", "amountPaidOnContentsClaim"):
        page[col] = pd.to_numeric(page.get(col, 0), errors="coerce").fillna(0.0)

    page["total_loss"] = (
        page["amountPaidOnBuildingClaim"] + page["amountPaidOnContentsClaim"]
    )
    page_paid = page[page["total_loss"] > 0].copy()
    paid = len(page_paid)

    if temporal_agg is not None:
        page_paid["year"] = pd.to_numeric(
            page_paid.get("yearOfLoss", None), errors="coerce"
        ).fillna(0).astype(int)

    for _, row in page_paid.iterrows():
        z = row["zip5"]
        b = float(row["amountPaidOnBuildingClaim"])
        c = float(row["amountPaidOnContentsClaim"])

        # Standard aggregation (unchanged)
        if z not in agg:
            agg[z] = {"count": 0, "building": 0.0, "contents": 0.0}
        agg[z]["count"] += 1
        agg[z]["building"] += b
        agg[z]["contents"] += c

        # Temporal aggregation
        if temporal_agg is not None:
            yr = int(row["year"])
            if yr < 1970 or yr > 2030:
                yr = 0  # sentinel for unknown/bad year
            if z not in temporal_agg:
                temporal_agg[z] = {}
            if yr not in temporal_agg[z]:
                temporal_agg[z][yr] = {"count": 0, "building": 0.0, "contents": 0.0}
            temporal_agg[z][yr]["count"] += 1
            temporal_agg[z][yr]["building"] += b
            temporal_agg[z][yr]["contents"] += c

    return paid


# ---------------------------------------------------------------------------
# Temporal pipeline
# ---------------------------------------------------------------------------
def make_long_zcta_nfip(
    temporal_agg: dict, xwalk: pd.DataFrame
) -> pd.DataFrame:
    """Convert temporal_agg to long format (zcta_id x year), zero-filled.

    temporal_agg: {zip5: {year: {count, building, contents}}}
    xwalk: DataFrame with zcta_id column (canonical ZCTA list)
    """
    import numpy as np

    log.info("Building NFIP long format...")

    # Flatten dict to rows
    rows = []
    for zip5, year_data in temporal_agg.items():
        for yr, vals in year_data.items():
            if yr == 0:
                continue  # skip unknown year sentinel
            rows.append({
                "zcta_id": zip5,
                "year": yr,
                "nfip_claim_count": vals["count"],
                "nfip_building_loss": round(vals["building"], 2),
                "nfip_contents_loss": round(vals["contents"], 2),
                "nfip_total_loss": round(vals["building"] + vals["contents"], 2),
            })

    if not rows:
        log.warning("No temporal rows — returning empty long DataFrame")
        return pd.DataFrame(columns=["zcta_id", "year", "nfip_claim_count",
                                     "nfip_building_loss", "nfip_contents_loss",
                                     "nfip_total_loss"])

    flat = pd.DataFrame(rows)
    flat["zcta_id"] = flat["zcta_id"].astype(str).str.zfill(5)

    # Year range from data (NFIP goes back to ~1978)
    year_min = int(flat["year"].min())
    year_max = int(flat["year"].max())
    log.info("  Year range in data: %d-%d", year_min, year_max)

    # Complete zcta x year index for canonical ZCTAs only
    all_zctas = xwalk["zcta_id"].unique()
    all_years = np.arange(year_min, year_max + 1)
    idx = pd.MultiIndex.from_product([all_zctas, all_years], names=["zcta_id", "year"])
    template = pd.DataFrame(index=idx).reset_index()

    long = template.merge(flat, on=["zcta_id", "year"], how="left")
    long["nfip_claim_count"] = long["nfip_claim_count"].fillna(0).astype(int)
    for col in ("nfip_building_loss", "nfip_contents_loss", "nfip_total_loss"):
        long[col] = long[col].fillna(0.0)

    long = long.sort_values(["zcta_id", "year"]).reset_index(drop=True)
    log.info("  Long format: %d rows (%d ZCTAs x %d years)",
             len(long), len(all_zctas), len(all_years))
    return long


def make_wide_epochs_nfip(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot NFIP long format to epoch-aggregated wide format."""
    log.info("Building NFIP wide epoch format...")
    metrics = ["nfip_claim_count", "nfip_building_loss",
               "nfip_contents_loss", "nfip_total_loss"]

    result = long[["zcta_id"]].drop_duplicates().copy()

    for epoch, (y0, y1) in EPOCHS.items():
        mask = long["year"].between(y0, y1)
        epoch_agg = (
            long[mask]
            .groupby("zcta_id")[metrics]
            .sum()
            .reset_index()
            .rename(columns={m: f"{m}_{epoch}" for m in metrics})
        )
        result = result.merge(epoch_agg, on="zcta_id", how="left")

    # Rolling totals (all years, not just epoch range)
    totals = long.groupby("zcta_id")[metrics].sum().reset_index()
    totals = totals.rename(columns={m: f"{m}_total" for m in metrics})
    result = result.merge(totals, on="zcta_id", how="left")

    # Peak single-year loss
    peak = (
        long.groupby("zcta_id")["nfip_total_loss"]
        .max()
        .reset_index(name="nfip_total_loss_peak_yr")
    )
    result = result.merge(peak, on="zcta_id", how="left")

    # Zero-fill
    for col in result.columns:
        if col != "zcta_id":
            result[col] = result[col].fillna(0)

    int_cols = [c for c in result.columns if "claim_count" in c]
    for col in int_cols:
        result[col] = result[col].astype(int)

    log.info("  Wide format: %d rows, %d columns", len(result), len(result.columns))
    return result


# ---------------------------------------------------------------------------
# Load crosswalk
# ---------------------------------------------------------------------------
def load_crosswalk(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir) / "zcta_county_crosswalk.parquet"
    if not path.exists():
        # Try download from S3
        tmp = "/tmp/zcta_county_crosswalk.parquet"
        if _s3_download(f"{DATA_PREFIX}/zcta_county_crosswalk.parquet", tmp):
            path = Path(tmp)
        else:
            log.error("zcta_county_crosswalk.parquet not found")
            sys.exit(1)
    xwalk = pd.read_parquet(path)[["zcta_id"]].drop_duplicates()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    log.info("Crosswalk: %d ZCTAs", len(xwalk))
    return xwalk


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build NFIP claims per ZCTA (SageMaker)")
    parser.add_argument("--data-dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit pages for testing (None = all)")
    parser.add_argument("--temporal", action="store_true",
                        help="Also produce long (zcta x year) and wide (epoch) Parquet outputs")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "nfip_checkpoint.json"

    # --- Pre-flight count ---
    total_records = get_total_count()
    if total_records:
        total_pages = (total_records + PAGE_SIZE - 1) // PAGE_SIZE
        log.info("OpenFEMA total claims: %d (~%d pages)", total_records, total_pages)
    else:
        total_pages = None
        log.info("OpenFEMA total unknown — streaming until empty page")

    # --- Crash recovery ---
    agg: dict = {}
    temporal_agg: dict | None = {} if args.temporal else None
    start_offset = 0
    pages_done = 0
    total_paid = 0

    if not checkpoint_path.exists():
        _s3_download(S3_CHECKPOINT_KEY, str(checkpoint_path))

    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
            agg = ckpt.get("agg", {})
            start_offset = ckpt.get("next_offset", 0)
            pages_done = ckpt.get("pages_done", 0)
            total_paid = ckpt.get("total_paid", 0)
            if args.temporal and "temporal_agg" in ckpt:
                # Restore nested dict; JSON keys are strings so cast year keys to int
                raw = ckpt["temporal_agg"]
                temporal_agg = {
                    z: {int(yr): vals for yr, vals in yr_data.items()}
                    for z, yr_data in raw.items()
                }
            log.info("Resuming from checkpoint: offset=%d, pages=%d, paid=%d",
                     start_offset, pages_done, total_paid)
        except Exception as exc:
            log.warning("Checkpoint corrupt, starting fresh: %s", exc)
            agg, start_offset, pages_done, total_paid = {}, 0, 0, 0
            temporal_agg = {} if args.temporal else None

    # --- Stream and accumulate ---
    log.info("=== STREAMING OPENFEMA NFIP CLAIMS ===")
    offset = start_offset

    while True:
        if args.max_pages and pages_done >= args.max_pages:
            log.info("Reached --max-pages=%d limit", args.max_pages)
            break

        page_num = pages_done + 1
        if total_pages:
            log.info("  Page %d/%d (offset=%d)", page_num, total_pages, offset)
        else:
            log.info("  Page %d (offset=%d)", page_num, offset)

        page = fetch_page(offset)

        if page.empty:
            log.info("  Empty page at offset=%d — done.", offset)
            break

        paid_this_page = accumulate_page(page, agg, temporal_agg)
        total_paid += paid_this_page
        pages_done += 1
        offset += len(page)

        log.info("  Page %d: %d records, %d paid, running total paid=%d, ZIPs=%d",
                 page_num, len(page), paid_this_page, total_paid, len(agg))

        # Checkpoint every 50 pages
        if pages_done % 50 == 0:
            ckpt_data = {
                "agg": agg,
                "next_offset": offset,
                "pages_done": pages_done,
                "total_paid": total_paid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if temporal_agg is not None:
                ckpt_data["temporal_agg"] = temporal_agg
            checkpoint_path.write_text(json.dumps(ckpt_data))
            _s3_upload(str(checkpoint_path), S3_CHECKPOINT_KEY, quiet=True)
            log.info("  Checkpoint saved (page %d)", pages_done)

        if len(page) < PAGE_SIZE:
            log.info("  Partial page (%d) — done.", len(page))
            break

        time.sleep(0.3)  # gentle throttle

    log.info("Stream complete: %d pages, %d paid claims, %d unique ZIPs",
             pages_done, total_paid, len(agg))

    # --- Build ZCTA result ---
    log.info("=== BUILDING ZCTA RESULT ===")
    xwalk = load_crosswalk(args.data_dir)

    rows = []
    for zip5, vals in agg.items():
        rows.append({
            "zcta_id": zip5,
            "nfip_claim_count": vals["count"],
            "nfip_total_building_loss": round(vals["building"], 2),
            "nfip_total_contents_loss": round(vals["contents"], 2),
            "nfip_total_loss": round(vals["building"] + vals["contents"], 2),
        })
    claims = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["zcta_id", "nfip_claim_count",
                 "nfip_total_building_loss", "nfip_total_contents_loss",
                 "nfip_total_loss"]
    )
    claims["nfip_mean_loss_per_claim"] = (
        claims["nfip_total_loss"] / claims["nfip_claim_count"].replace(0, float("nan"))
    ).round(2)
    claims["nfip_has_claims"] = True

    # Zero-fill all ZCTAs
    result = xwalk.merge(claims, on="zcta_id", how="left")
    result["nfip_claim_count"] = result["nfip_claim_count"].fillna(0).astype(int)
    result["nfip_total_building_loss"] = result["nfip_total_building_loss"].fillna(0.0)
    result["nfip_total_contents_loss"] = result["nfip_total_contents_loss"].fillna(0.0)
    result["nfip_total_loss"] = result["nfip_total_loss"].fillna(0.0)
    result["nfip_mean_loss_per_claim"] = result["nfip_mean_loss_per_claim"].fillna(0.0)
    result["nfip_has_claims"] = result["nfip_has_claims"].fillna(False)

    # --- Validation ---
    log.info("=== VALIDATION ===")
    n_with = result["nfip_has_claims"].sum()
    log.info("ZCTAs total:           %d", len(result))
    log.info("ZCTAs with claims:     %d (%.1f%%)", n_with, 100 * n_with / len(result))
    log.info("Total paid claims:     %d", int(result["nfip_claim_count"].sum()))
    log.info("Total losses:          $%.1fB",
             result["nfip_total_loss"].sum() / 1e9)
    log.info("Mean loss per claim:   $%.0f",
             result[result["nfip_claim_count"] > 0]["nfip_mean_loss_per_claim"].mean())

    # Sanity checks
    if n_with < 5000:
        log.warning("VALIDATION WARN: Only %d ZCTAs with claims — expected >10K nationally", n_with)
    if result["nfip_total_loss"].sum() < 1e10:
        log.warning("VALIDATION WARN: Total losses < $10B — expected ~$50-60B historically")

    # Harris County TX spot check (48201 — Harvey ground zero)
    harris_zips = ["77002","77006","77007","77008","77009","77018","77019","77025"]
    harris_sub = result[result["zcta_id"].isin(harris_zips)]
    harris_with_claims = harris_sub["nfip_has_claims"].sum()
    log.info("Harris County spot check: %d/%d test ZCTAs have claims",
             harris_with_claims, len(harris_sub))
    if harris_with_claims == 0:
        log.warning("VALIDATION WARN: Zero claims in Harris County — "
                    "possible ZIP normalization failure")
    else:
        log.info("  PASS: Harris County shows NFIP claims as expected")

    # --- Save and upload ---
    out_path = output_dir / "nfip_claims_zcta.parquet"
    result.to_parquet(out_path, index=False)
    log.info("Saved: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    _s3_upload(str(out_path), S3_OUTPUT_KEY)

    provenance = {
        "operation": "build_nfip_claims",
        "timestamp": timestamp,
        "source": OPENFEMA_URL,
        "pages_fetched": pages_done,
        "total_paid_claims": int(result["nfip_claim_count"].sum()),
        "total_loss_usd": float(result["nfip_total_loss"].sum()),
        "n_zctas": len(result),
        "n_zctas_with_claims": int(n_with),
        "harris_county_spot_check": int(harris_with_claims),
    }
    prov_path = output_dir / "nfip_claims_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    _s3_upload(str(prov_path), S3_PROVENANCE_KEY)

    # --- Temporal outputs (--temporal flag) ---
    if args.temporal and temporal_agg:
        log.info("=== TEMPORAL PIPELINE ===")
        long = make_long_zcta_nfip(temporal_agg, xwalk)

        long_path = output_dir / "nfip_claims_long.parquet"
        long.to_parquet(long_path, index=False)
        log.info("Saved: %s (%.1f KB)", long_path, long_path.stat().st_size / 1024)
        _s3_upload(str(long_path), S3_LONG_KEY)

        wide = make_wide_epochs_nfip(long)
        wide_path = output_dir / "nfip_claims_wide.parquet"
        wide.to_parquet(wide_path, index=False)
        log.info("Saved: %s (%.1f KB)", wide_path, wide_path.stat().st_size / 1024)
        _s3_upload(str(wide_path), S3_WIDE_KEY)

        # Epoch summary
        for epoch, (y0, y1) in EPOCHS.items():
            mask = long["year"].between(y0, y1)
            active = long[mask & (long["nfip_claim_count"] > 0)]["zcta_id"].nunique()
            log.info("  %s (%d-%d): %d ZCTAs with claims", epoch, y0, y1, active)
        peak_zcta = wide.loc[wide["nfip_total_loss_peak_yr"].idxmax(), "zcta_id"]
        peak_val = wide["nfip_total_loss_peak_yr"].max()
        log.info("  Peak single-year loss ZCTA: %s ($%.1fM)", peak_zcta, peak_val / 1e6)
    elif args.temporal:
        log.warning("--temporal requested but temporal_agg is empty — skipping long/wide outputs")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    log.info("Done.")


if __name__ == "__main__":
    main()
