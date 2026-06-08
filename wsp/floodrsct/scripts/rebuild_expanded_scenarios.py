#!/usr/bin/env python3
"""
rebuild_expanded_scenarios.py -- Reassemble event feature parquets with expanded
county boundaries.

Houston expanded from Harris-only (132 ZCTAs) to 6-county metro.
SW Florida expanded to include Charlotte County.

This script reads configs, resolves expanded ZCTA sets via the county crosswalk,
joins all feature layers, and writes updated parquets locally (and optionally
uploads to S3).

Usage:
    python rebuild_expanded_scenarios.py --scenario houston
    python rebuild_expanded_scenarios.py --scenario southwest_florida
    python rebuild_expanded_scenarios.py --scenario both
    python rebuild_expanded_scenarios.py --scenario both --upload
"""

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
CONFIG_DIR = REPO_DIR / "configs"
OUTPUT_DIR = REPO_DIR / "outputs"

# S3 key templates
OUTPUT_KEYS: dict[str, str] = {
    "houston": "processed/houston/houston_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}

# Supplement S3 keys per scenario
SUPPLEMENT_KEYS: dict[str, dict[str, str]] = {
    "houston": {
        "nfip_historical": "processed/houston/houston_nfip_historical.parquet",
        "r1_supplement": "processed/houston/houston_r1_supplement.parquet",
        "r2_supplement": "processed/houston/houston_r2_supplement.parquet",
    },
    "southwest_florida": {
        "nfip_historical": "processed/southwest_florida/southwest_florida_nfip_historical.parquet",
        "r1_supplement": "processed/southwest_florida/southwest_florida_r1_supplement.parquet",
        "r2_supplement": "processed/southwest_florida/southwest_florida_r2_supplement.parquet",
    },
}

# Raw geocertdb2026 layer keys
RAW_KEYS: dict[str, str] = {
    "features_labels": "raw/geocertdb2026/zcta_features_labels.parquet",
    "crosswalk": "raw/geocertdb2026/zcta_county_crosswalk.parquet",
    "flood_zones": "raw/geocertdb2026/flood_zones_zcta.parquet",
    "svi": "raw/geocertdb2026/svi_zcta.parquet",
    "twi": "raw/geocertdb2026/twi_features_zcta.parquet",
    "adjacency": "raw/geocertdb2026/zcta_adjacency.parquet",
}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _get_s3_client():
    """Create S3 client using P18 credential triage."""
    import boto3
    from swarm_auth import get_aws_credentials
    return boto3.client("s3", **get_aws_credentials())


def s3_read(s3, key: str) -> Optional[pd.DataFrame]:
    """Read a parquet from swarm-floodrsct-data. Returns None if missing."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        log.warning("S3 key not found: %s", key)
        return None
    except Exception as e:
        log.warning("Could not read %s: %s", key, e)
        return None


def s3_upload(s3, df: pd.DataFrame, key: str) -> None:
    """Write a DataFrame as parquet and upload to S3."""
    local = Path(f"/tmp/{Path(key).name}")
    df.to_parquet(local, index=False)
    s3.upload_file(str(local), BUCKET, key)
    log.info(
        "Uploaded %d rows x %d cols to s3://%s/%s",
        len(df), len(df.columns), BUCKET, key,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(scenario: str) -> dict:
    """Load scenario YAML config.

    Args:
        scenario: One of 'houston', 'southwest_florida'.

    Returns:
        Parsed config dict.
    """
    path = CONFIG_DIR / f"{scenario}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ZCTA resolution via county crosswalk
# ---------------------------------------------------------------------------

def resolve_zctas(
    s3,
    county_fips_list: list[str],
    state_filter: Optional[str] = None,
) -> list[str]:
    """Resolve ZCTA set from county FIPS codes via crosswalk.

    Args:
        s3: boto3 S3 client.
        county_fips_list: List of 5-digit county FIPS codes.
        state_filter: Optional state abbreviation for secondary filter.

    Returns:
        Sorted list of ZCTA IDs.
    """
    crosswalk = s3_read(s3, RAW_KEYS["crosswalk"])
    if crosswalk is None:
        raise FileNotFoundError(
            "County crosswalk not found on S3. Upload "
            f"s3://{BUCKET}/{RAW_KEYS['crosswalk']} first."
        )

    # Identify column names (flexible)
    zcta_col = _find_col(crosswalk, ["zcta_id", "zcta", "ZCTA5CE20", "geoid_zcta"])
    fips_col = _find_col(crosswalk, ["county_fips", "county", "GEOID", "geoid_county", "fips"])

    if zcta_col is None or fips_col is None:
        raise ValueError(
            f"Cannot identify ZCTA/county columns in crosswalk. "
            f"Columns: {crosswalk.columns.tolist()}"
        )

    # Normalize to string for matching
    crosswalk[fips_col] = crosswalk[fips_col].astype(str).str.zfill(5)
    crosswalk[zcta_col] = crosswalk[zcta_col].astype(str).str.zfill(5)

    mask = crosswalk[fips_col].isin(county_fips_list)
    filtered = crosswalk[mask]

    zctas = sorted(filtered[zcta_col].unique().tolist())
    log.info(
        "Resolved %d ZCTAs from %d counties via crosswalk",
        len(zctas), len(county_fips_list),
    )
    return zctas


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first matching column name from candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_master_features(
    s3,
    zcta_ids: list[str],
    scenario: str,
) -> pd.DataFrame:
    """Load master features/labels table filtered to scenario ZCTAs.

    Tries scenario-specific subset first, then falls back to national.

    Args:
        s3: boto3 S3 client.
        zcta_ids: List of ZCTA IDs to filter.
        scenario: Scenario name.

    Returns:
        Filtered DataFrame.
    """
    # Try scenario-specific first
    key = f"raw/geocertdb2026/scenarios/{scenario}/zcta_features_labels.parquet"
    df = s3_read(s3, key)
    if df is None:
        log.info("Scenario subset not found; loading national features table")
        df = s3_read(s3, RAW_KEYS["features_labels"])
    if df is None:
        raise FileNotFoundError(
            "Master features table not found on S3. "
            "Run copy_geocertdb2026 job first."
        )

    zcta_col = _find_col(df, ["zcta_id", "zcta", "ZCTA5CE20", "geoid"])
    if zcta_col is None:
        raise ValueError(f"No ZCTA column in features table. Cols: {df.columns.tolist()}")

    df[zcta_col] = df[zcta_col].astype(str).str.zfill(5)
    df = df[df[zcta_col].isin(zcta_ids)].copy()
    if zcta_col != "zcta_id":
        df = df.rename(columns={zcta_col: "zcta_id"})

    log.info("Master features: %d ZCTAs x %d cols", len(df), len(df.columns))
    return df


def load_feature_layer(
    s3,
    key: str,
    zcta_ids: list[str],
    layer_name: str,
) -> Optional[pd.DataFrame]:
    """Load a single feature layer parquet, filtered to scenario ZCTAs.

    Args:
        s3: boto3 S3 client.
        key: S3 key for the parquet.
        zcta_ids: ZCTA filter set.
        layer_name: Human label for logging.

    Returns:
        Filtered DataFrame or None if missing.
    """
    df = s3_read(s3, key)
    if df is None:
        log.warning("Layer '%s' not found at %s", layer_name, key)
        return None

    zcta_col = _find_col(df, ["zcta_id", "zcta", "ZCTA5CE20", "geoid"])
    if zcta_col is None:
        log.warning("Layer '%s': no ZCTA column found. Cols: %s", layer_name, df.columns.tolist())
        return None

    df[zcta_col] = df[zcta_col].astype(str).str.zfill(5)
    df = df[df[zcta_col].isin(zcta_ids)].copy()
    if zcta_col != "zcta_id":
        df = df.rename(columns={zcta_col: "zcta_id"})

    log.info("Layer '%s': %d rows x %d cols", layer_name, len(df), len(df.columns))
    return df


def load_supplement(
    s3,
    key: str,
    zcta_ids: list[str],
    supplement_name: str,
) -> Optional[pd.DataFrame]:
    """Load a supplement parquet (NFIP historical, R1, R2).

    Supplements may be keyed on (zcta_id) or (zcta_id, event). Filters to
    the expanded ZCTA set and flags ZCTAs that are missing.

    Args:
        s3: boto3 S3 client.
        key: S3 key.
        zcta_ids: Expanded ZCTA set.
        supplement_name: Human label for logging.

    Returns:
        Filtered DataFrame, or None if missing entirely.
    """
    df = s3_read(s3, key)
    if df is None:
        log.warning(
            "SUPPLEMENT MISSING: '%s' at %s -- new ZCTAs will have NaN for "
            "these columns. Regenerate with the appropriate build script.",
            supplement_name, key,
        )
        return None

    zcta_col = _find_col(df, ["zcta_id", "zcta", "ZCTA5CE20", "geoid"])
    if zcta_col is None:
        log.warning("Supplement '%s': no ZCTA column. Cols: %s", supplement_name, df.columns.tolist())
        return None

    df[zcta_col] = df[zcta_col].astype(str).str.zfill(5)
    present = set(df[zcta_col].unique())
    missing = set(zcta_ids) - present
    if missing:
        log.warning(
            "SUPPLEMENT GAP: '%s' is missing %d/%d expanded ZCTAs. "
            "Regeneration needed for: %s",
            supplement_name, len(missing), len(zcta_ids),
            sorted(missing)[:10],
        )

    df = df[df[zcta_col].isin(zcta_ids)].copy()
    if zcta_col != "zcta_id":
        df = df.rename(columns={zcta_col: "zcta_id"})

    log.info(
        "Supplement '%s': %d rows (covers %d/%d ZCTAs)",
        supplement_name, len(df), len(present & set(zcta_ids)), len(zcta_ids),
    )
    return df


# ---------------------------------------------------------------------------
# Event grid construction
# ---------------------------------------------------------------------------

def build_event_grid(
    zcta_ids: list[str],
    events: dict[str, dict],
) -> pd.DataFrame:
    """Cross ZCTAs with events to produce the (zcta_id, event) grid.

    Args:
        zcta_ids: List of ZCTA IDs.
        events: Event config dict from YAML.

    Returns:
        DataFrame with columns [zcta_id, event] plus event metadata.
    """
    rows = []
    for event_name, ev_cfg in events.items():
        for zcta in zcta_ids:
            row = {
                "zcta_id": zcta,
                "event": event_name,
                "disaster_declaration": ev_cfg.get("disaster_declaration", ""),
                "event_start_date": ev_cfg.get("start_date", ""),
                "event_end_date": ev_cfg.get("end_date", ""),
                "nhc_storm_id": ev_cfg.get("nhc_storm_id", ""),
            }
            rows.append(row)

    grid = pd.DataFrame(rows)
    log.info(
        "Event grid: %d ZCTAs x %d events = %d rows",
        len(zcta_ids), len(events), len(grid),
    )
    return grid


# ---------------------------------------------------------------------------
# Join logic
# ---------------------------------------------------------------------------

def join_features(
    grid: pd.DataFrame,
    master: pd.DataFrame,
    layers: dict[str, Optional[pd.DataFrame]],
    supplements: dict[str, Optional[pd.DataFrame]],
) -> pd.DataFrame:
    """Join all feature sources onto the (zcta_id, event) grid.

    Static layers (master, flood zones, SVI, TWI) join on zcta_id only.
    Supplements may join on (zcta_id, event) if they have an event column,
    otherwise on zcta_id only.

    Args:
        grid: The (zcta_id, event) base grid.
        master: Master features/labels table.
        layers: Dict of layer_name -> DataFrame (flood_zones, svi, twi).
        supplements: Dict of supplement_name -> DataFrame (nfip_hist, r1, r2).

    Returns:
        Assembled DataFrame.
    """
    result = grid.copy()

    # Join master features (static, zcta_id only)
    # Drop event column if present in master to avoid confusion
    master_cols = [c for c in master.columns if c != "event"]
    master_dedup = master[master_cols].drop_duplicates(subset=["zcta_id"])
    result = result.merge(master_dedup, on="zcta_id", how="left")
    log.info("After master join: %d cols", len(result.columns))

    # Join each spatial layer
    for layer_name, layer_df in layers.items():
        if layer_df is None:
            continue
        layer_cols = [c for c in layer_df.columns if c != "event"]
        layer_dedup = layer_df[layer_cols].drop_duplicates(subset=["zcta_id"])
        # Avoid duplicate columns (master may already have some)
        overlap = set(result.columns) & set(layer_dedup.columns) - {"zcta_id"}
        if overlap:
            log.info(
                "Layer '%s': dropping %d overlapping cols: %s",
                layer_name, len(overlap), sorted(overlap)[:5],
            )
            layer_dedup = layer_dedup.drop(columns=list(overlap))
        result = result.merge(layer_dedup, on="zcta_id", how="left")
        log.info("After '%s' join: %d cols", layer_name, len(result.columns))

    # Join supplements (may be event-keyed or static)
    for sup_name, sup_df in supplements.items():
        if sup_df is None:
            continue
        if "event" in sup_df.columns:
            join_keys = ["zcta_id", "event"]
        else:
            join_keys = ["zcta_id"]
            sup_df = sup_df.drop_duplicates(subset=["zcta_id"])

        overlap = set(result.columns) & set(sup_df.columns) - set(join_keys)
        if overlap:
            log.info(
                "Supplement '%s': dropping %d overlapping cols: %s",
                sup_name, len(overlap), sorted(overlap)[:5],
            )
            sup_df = sup_df.drop(columns=list(overlap))
        result = result.merge(sup_df, on=join_keys, how="left")
        log.info("After '%s' join: %d cols", sup_name, len(result.columns))

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(
    df: pd.DataFrame,
    scenario: str,
    config: dict,
    old_row_count: Optional[int],
) -> None:
    """Print validation summary and check data quality.

    Args:
        df: Assembled output DataFrame.
        scenario: Scenario name.
        config: Scenario config dict.
        old_row_count: Row count of the previously assembled parquet (if available).
    """
    n_zctas = df["zcta_id"].nunique()
    n_events = df["event"].nunique()
    n_rows = len(df)

    log.info("=" * 70)
    log.info("VALIDATION SUMMARY: %s", scenario)
    log.info("=" * 70)

    # Size comparison
    if old_row_count is not None:
        old_zctas = old_row_count // n_events if n_events > 0 else old_row_count
        log.info(
            "ZCTAs:  old=%d  new=%d  (delta=+%d)",
            old_zctas, n_zctas, n_zctas - old_zctas,
        )
        log.info(
            "Rows:   old=%d  new=%d  (delta=+%d)",
            old_row_count, n_rows, n_rows - old_row_count,
        )
    else:
        log.info("ZCTAs: %d", n_zctas)
        log.info("Rows:  %d", n_rows)

    log.info("Events: %s", sorted(df["event"].unique().tolist()))
    log.info("Columns: %d", len(df.columns))

    # Duplicate check
    dupes = df.duplicated(subset=["zcta_id", "event"], keep=False)
    if dupes.any():
        n_dupes = dupes.sum()
        log.error(
            "DUPLICATE (zcta_id, event) ROWS DETECTED: %d rows. "
            "This is a data integrity violation.",
            n_dupes,
        )
    else:
        log.info("Duplicate check: PASS (no duplicate zcta_id x event)")

    # Required columns check
    mvd = config.get("mvd", {})
    required = mvd.get("required_columns", [])
    if required:
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            log.warning(
                "MISSING REQUIRED COLUMNS: %s (not in output)",
                missing_cols,
            )
        present_required = [c for c in required if c in df.columns]
        for col in present_required:
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_zctas = df.loc[df[col].isna(), "zcta_id"].unique()
                log.warning(
                    "NULL in required column '%s': %d/%d rows (%d ZCTAs: %s)",
                    col, null_count, n_rows, len(null_zctas),
                    sorted(null_zctas)[:10],
                )
            else:
                log.info("Required column '%s': OK (no nulls)", col)

    # Overall null summary
    null_pcts = df.isnull().mean().sort_values(ascending=False)
    high_null = null_pcts[null_pcts > 0.5]
    if not high_null.empty:
        log.warning("Columns with >50%% nulls:")
        for col, pct in high_null.items():
            log.warning("  %s: %.1f%%", col, pct * 100)

    # Min ZCTA check
    min_zctas = mvd.get("min_zctas", 0)
    if min_zctas and n_zctas < min_zctas:
        log.warning(
            "MVD VIOLATION: need %d ZCTAs, have %d",
            min_zctas, n_zctas,
        )
    elif min_zctas:
        log.info("MVD min_zctas: PASS (%d >= %d)", n_zctas, min_zctas)

    log.info("=" * 70)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(scenario: str, upload: bool = False) -> pd.DataFrame:
    """Run the full rebuild pipeline for one scenario.

    Args:
        scenario: 'houston' or 'southwest_florida'.
        upload: If True, upload result to S3.

    Returns:
        Assembled DataFrame.
    """
    log.info("=" * 70)
    log.info("REBUILDING: %s (expanded boundaries)", scenario.upper())
    log.info("=" * 70)

    s3 = _get_s3_client()
    config = load_config(scenario)

    county_fips = config["county_fips_list"]
    events = config["events"]
    log.info("Counties: %s", county_fips)
    log.info("Events: %s", list(events.keys()))

    # Step 1: Resolve expanded ZCTA set
    zcta_ids = resolve_zctas(
        s3,
        county_fips,
        state_filter=config.get("zcta_state_filter"),
    )

    # Step 2: Load old parquet for comparison
    old_key = OUTPUT_KEYS[scenario]
    old_df = s3_read(s3, old_key)
    old_row_count = len(old_df) if old_df is not None else None
    if old_df is not None:
        old_zctas = old_df["zcta_id"].nunique() if "zcta_id" in old_df.columns else None
        log.info(
            "Old parquet: %d rows, %d ZCTAs",
            len(old_df), old_zctas or -1,
        )
    else:
        log.info("No existing parquet found at %s", old_key)

    # Step 3: Build (zcta_id, event) grid
    grid = build_event_grid(zcta_ids, events)

    # Step 4: Load master features
    master = load_master_features(s3, zcta_ids, scenario)

    # Step 5: Load supplemental layers
    layers: dict[str, Optional[pd.DataFrame]] = {}
    for layer_name, key in [
        ("flood_zones", RAW_KEYS["flood_zones"]),
        ("svi", RAW_KEYS["svi"]),
        ("twi", RAW_KEYS["twi"]),
    ]:
        layers[layer_name] = load_feature_layer(s3, key, zcta_ids, layer_name)

    # Step 6: Load scenario-specific supplements
    supplements: dict[str, Optional[pd.DataFrame]] = {}
    scenario_sups = SUPPLEMENT_KEYS.get(scenario, {})
    for sup_name, sup_key in scenario_sups.items():
        supplements[sup_name] = load_supplement(s3, sup_key, zcta_ids, sup_name)

    # Step 7: Join everything
    result = join_features(grid, master, layers, supplements)

    # Step 8: Validate
    validate_output(result, scenario, config, old_row_count)

    # Step 9: Write local output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUTPUT_DIR / f"{scenario}_event_features.parquet"
    result.to_parquet(local_path, index=False)
    log.info("Wrote local: %s (%d rows x %d cols)", local_path, len(result), len(result.columns))

    # Step 10: Upload to S3
    if upload:
        s3_upload(s3, result, OUTPUT_KEYS[scenario])
        log.info("Uploaded to s3://%s/%s", BUCKET, OUTPUT_KEYS[scenario])
    else:
        log.info("Skipping S3 upload (pass --upload to enable)")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Rebuild event feature parquets with expanded county boundaries.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["houston", "southwest_florida", "both"],
        help="Which scenario to rebuild.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload rebuilt parquets to S3 (replaces existing).",
    )
    args = parser.parse_args()

    scenarios = (
        ["houston", "southwest_florida"]
        if args.scenario == "both"
        else [args.scenario]
    )

    results: dict[str, pd.DataFrame] = {}
    for scenario in scenarios:
        results[scenario] = run_scenario(scenario, upload=args.upload)

    # Final summary
    log.info("")
    log.info("=" * 70)
    log.info("REBUILD COMPLETE")
    log.info("=" * 70)
    for scenario, df in results.items():
        log.info(
            "  %s: %d ZCTAs x %d events = %d rows, %d cols",
            scenario,
            df["zcta_id"].nunique(),
            df["event"].nunique(),
            len(df),
            len(df.columns),
        )


if __name__ == "__main__":
    main()
