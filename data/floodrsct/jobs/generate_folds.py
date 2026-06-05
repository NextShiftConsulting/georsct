#!/usr/bin/env python3
"""
generate_folds.py -- Phase 1: Generate deterministic fold assignments.

Creates three fold assignment columns for a scenario's assembled parquet:
  1. fold_random:          random 80/20 split (seed-controlled)
  2. fold_spatial_blocked: 5-fold spatially-blocked CV (county or ZIP3 fallback)
  3. fold_leave_event_out: event name as fold ID

The spatial blocking strategy adapts to data:
  - If n_counties >= n_folds: block on county (ideal -- eliminates cross-county leakage)
  - If n_counties < n_folds:  block on ZIP3 prefix (postal region -- still spatially coherent)

This matters because Houston has ~1 county (Harris) but 5 ZIP3 prefixes (770-775).

Fold assignments are saved as a parquet keyed on (zcta_id, event).
Every representation level (R0, R1, R2) reads the SAME fold file.

Usage:
    python generate_folds.py --scenario houston --upload
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
from _coverage_common import (
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet, load_crosswalk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

SEED = 42
N_FOLDS = 5
TEST_FRACTION_RANDOM = 0.2
MIN_BLOCK_ZCTAS = 5   # Merge blocks with fewer ZCTAs into the largest remaining block
MIN_FOLD_ZCTAS = 5    # Reduce n_folds until every fold has at least this many ZCTAs


def assign_random_split(zctas: list[str], seed: int) -> dict[str, str]:
    """Assign ZCTAs to 'train' or 'test' (80/20 random split)."""
    rng = np.random.RandomState(seed)
    n_test = max(1, int(len(zctas) * TEST_FRACTION_RANDOM))
    indices = rng.permutation(len(zctas))
    return {
        zctas[idx]: ("test" if i < n_test else "train")
        for i, idx in enumerate(indices)
    }


def _merge_small_blocks(
    block_sizes: dict[str, int],
    min_size: int,
) -> dict[str, str]:
    """Merge blocks smaller than min_size into the largest remaining block.

    Returns a mapping from original block_id -> merged block_id.
    Small blocks are absorbed into the largest block to minimize
    disruption to the bin-packing balance.
    """
    merged_to = {b: b for b in block_sizes}
    if min_size <= 1:
        return merged_to

    # Sort: largest first so merge targets are stable
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: -x[1])
    large_blocks = [(b, s) for b, s in sorted_blocks if s >= min_size]

    if not large_blocks:
        # All blocks are small -- nothing to merge into, return as-is
        return merged_to

    for block, size in sorted_blocks:
        if size < min_size:
            # Merge into the largest block
            target = large_blocks[0][0]
            merged_to[block] = target
            log.info(
                "Merging small block %s (%d ZCTAs) into %s",
                block, size, target,
            )

    return merged_to


def _greedy_bin_pack(
    block_sizes: dict[str, int],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign blocks to folds via greedy bin-packing on item count.

    Shuffles blocks first (seeded), then assigns each to the fold
    with the fewest items so far.
    """
    rng = np.random.RandomState(seed)
    blocks = list(block_sizes.keys())
    rng.shuffle(blocks)

    fold_counts = np.zeros(n_folds, dtype=int)
    assignments = {}
    for block in blocks:
        best = int(np.argmin(fold_counts))
        assignments[block] = best
        fold_counts[best] += block_sizes[block]

    return assignments


def assign_spatial_blocked_folds(
    zctas: list[str],
    zcta_county: dict[str, str],
    n_folds: int,
    seed: int,
    min_block_zctas: int = MIN_BLOCK_ZCTAS,
    min_fold_zctas: int = MIN_FOLD_ZCTAS,
) -> tuple[dict[str, int], str, dict]:
    """Assign ZCTAs to folds via spatially-blocked greedy bin-packing.

    Returns (zcta->fold dict, blocking_strategy, fold_metadata).

    Ensures statistical validity:
      1. Merges blocks with < min_block_zctas into the largest neighbor.
      2. Caps n_folds at n_merged_blocks.
      3. Reduces n_folds until every fold has >= min_fold_zctas ZCTAs.

    Strategy selection:
      - county if n_unique_counties >= n_folds
      - ZIP3 prefix otherwise (first 3 digits of ZCTA ID)
    """
    # Determine blocking key per ZCTA
    counties = set(zcta_county.get(z, "unknown") for z in zctas)
    n_counties = len(counties - {"unknown"})

    if n_counties >= n_folds:
        strategy = "county"
        zcta_block = {z: zcta_county.get(z, "unknown") for z in zctas}
    else:
        strategy = "zip3"
        zcta_block = {z: z[:3] for z in zctas}
        n_zip3 = len(set(zcta_block.values()))
        log.info(
            "Only %d counties (need %d for %d-fold CV). "
            "Falling back to ZIP3 blocking (%d groups).",
            n_counties, n_folds, n_folds, n_zip3,
        )

    # Count ZCTAs per block
    block_sizes: dict[str, int] = {}
    block_zctas: dict[str, list[str]] = {}
    for z in zctas:
        b = zcta_block[z]
        block_sizes[b] = block_sizes.get(b, 0) + 1
        block_zctas.setdefault(b, []).append(z)

    # Step 1: Merge small blocks
    merge_map = _merge_small_blocks(block_sizes, min_block_zctas)
    merged_sizes: dict[str, int] = {}
    merged_zctas: dict[str, list[str]] = {}
    for block, target in merge_map.items():
        merged_sizes[target] = merged_sizes.get(target, 0) + block_sizes[block]
        merged_zctas.setdefault(target, []).extend(block_zctas[block])

    n_merged = len(merged_sizes)
    n_blocks_removed = len(block_sizes) - n_merged
    if n_blocks_removed > 0:
        log.info(
            "Merged %d small blocks -> %d effective blocks",
            n_blocks_removed, n_merged,
        )

    # Step 2: Cap n_folds at n_merged_blocks
    effective_folds = min(n_folds, n_merged)
    if effective_folds < n_folds:
        log.info(
            "Reduced n_folds from %d to %d (only %d merged blocks)",
            n_folds, effective_folds, n_merged,
        )

    # Step 3: Bin-pack, then reduce n_folds until min fold size met
    while effective_folds >= 2:
        block_fold = _greedy_bin_pack(merged_sizes, effective_folds, seed)

        # Compute fold sizes in ZCTAs
        fold_zcta_counts = [0] * effective_folds
        for block, fold in block_fold.items():
            fold_zcta_counts[fold] += merged_sizes[block]

        min_fold = min(fold_zcta_counts)
        if min_fold >= min_fold_zctas:
            break
        log.info(
            "Min fold has %d ZCTAs (need %d). Reducing from %d to %d folds.",
            min_fold, min_fold_zctas, effective_folds, effective_folds - 1,
        )
        effective_folds -= 1
    else:
        # 2-fold is the floor -- accept whatever we get
        block_fold = _greedy_bin_pack(merged_sizes, 2, seed)
        effective_folds = 2
        floor_sizes = [0] * 2
        for block, fold in block_fold.items():
            floor_sizes[fold] += merged_sizes[block]
        if min(floor_sizes) < min_fold_zctas:
            log.warning(
                "2-fold floor reached but min fold (%d ZCTAs) < threshold (%d). "
                "Per-scenario paired tests will be underpowered.",
                min(floor_sizes), min_fold_zctas,
            )

    # Map back to ZCTAs
    assignments = {}
    for block, fold in block_fold.items():
        for z in merged_zctas[block]:
            assignments[z] = fold

    # Log fold sizes
    fold_sizes = [0] * effective_folds
    for fold in assignments.values():
        fold_sizes[fold] += 1
    log.info(
        "Spatial-blocked (%s, %d-fold) fold sizes (ZCTAs): %s",
        strategy, effective_folds, fold_sizes,
    )

    fold_meta = {
        "strategy": strategy,
        "n_folds_requested": n_folds,
        "n_folds_effective": effective_folds,
        "n_blocks_original": len(block_sizes),
        "n_blocks_merged": n_merged,
        "blocks_removed": n_blocks_removed,
        "fold_sizes_zctas": fold_sizes,
    }

    return assignments, strategy, fold_meta


def generate_folds(
    df: pd.DataFrame,
    zcta_county: dict[str, str],
    seed: int = SEED,
    n_folds: int = N_FOLDS,
) -> tuple[pd.DataFrame, dict]:
    """Generate fold assignments for a scenario dataframe.

    Returns (folds_df, metadata_dict).
    Callable from other scripts (train_r0_baseline imports this).
    """
    df = df.copy()
    df["zcta_id"] = df["zcta_id"].astype(str)
    all_zctas = sorted(df["zcta_id"].unique())

    # 1. Random 80/20
    random_map = assign_random_split(all_zctas, seed)

    # 2. Spatial-blocked CV (with adaptive merging and fold reduction)
    blocked_map, block_strategy, fold_meta = assign_spatial_blocked_folds(
        all_zctas, zcta_county, n_folds, seed,
    )

    # 3. Build folds aligned with df rows
    folds_df = pd.DataFrame({
        "zcta_id": df["zcta_id"].values,
        "event": df["event"].astype(str).values if "event" in df.columns
                 else ["unknown"] * len(df),
        "fold_random": [random_map[z] for z in df["zcta_id"]],
        "fold_spatial_blocked": [blocked_map[z] for z in df["zcta_id"]],
        "fold_leave_event_out": df["event"].astype(str).values if "event" in df.columns
                                else ["unknown"] * len(df),
    })

    metadata = {
        "seed": seed,
        "n_folds_requested": n_folds,
        "n_folds_effective": fold_meta["n_folds_effective"],
        "n_zctas": len(all_zctas),
        "n_rows": len(folds_df),
        "n_events": folds_df["event"].nunique(),
        "block_strategy": block_strategy,
        "block_merging": {
            "n_original": fold_meta["n_blocks_original"],
            "n_after_merge": fold_meta["n_blocks_merged"],
            "blocks_removed": fold_meta["blocks_removed"],
        },
        "random_split": folds_df["fold_random"].value_counts().to_dict(),
        "spatial_blocked_folds": folds_df["fold_spatial_blocked"]
            .value_counts().sort_index().to_dict(),
        "spatial_blocked_fold_sizes_zctas": fold_meta["fold_sizes_zctas"],
        "leave_event_out_folds": folds_df["fold_leave_event_out"]
            .value_counts().to_dict(),
    }

    return folds_df, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate deterministic fold assignments"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    print(f"\n=== PHASE 1: FOLD GENERATION -- {args.scenario} ===\n")

    df = load_processed_parquet(s3, args.scenario)

    # Load crosswalk
    try:
        xwalk = load_crosswalk(s3)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
        zcta_county = dict(zip(xwalk["zcta_id"], xwalk["county_fips"].astype(str)))
    except Exception as e:
        log.warning("Crosswalk unavailable (%s), blocking on ZIP3 only", e)
        zcta_county = {}

    folds_df, metadata = generate_folds(df, zcta_county, args.seed, args.n_folds)

    # Validate
    for col in ["fold_random", "fold_spatial_blocked", "fold_leave_event_out"]:
        n_null = folds_df[col].isna().sum()
        if n_null > 0:
            log.error("VALIDATION FAIL: %d NaN in %s", n_null, col)
            sys.exit(1)

    print(json.dumps(metadata, indent=2))

    key = f"folds/{args.scenario}_folds.parquet"
    if args.upload:
        buf = io.BytesIO()
        folds_df.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.read())
        log.info("Uploaded s3://%s/%s", BUCKET, key)

        s3.put_object(
            Bucket=BUCKET,
            Key=f"folds/{args.scenario}_folds_meta.json",
            Body=json.dumps(metadata, indent=2).encode(),
            ContentType="application/json",
        )
    else:
        folds_df.to_parquet(f"/tmp/{args.scenario}_folds.parquet", index=False)
        log.info("Wrote /tmp/%s_folds.parquet", args.scenario)


if __name__ == "__main__":
    main()
