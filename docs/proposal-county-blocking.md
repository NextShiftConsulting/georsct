# Proposal: County Blocking Robustness Check

**Date:** 2026-06-07
**Context:** DOE Appendix R — Spatial blocking geometry robustness
**PAR Review:** Completed (2 independent reviewers, aggregated below)

## Executive Summary

County blocking is **already implemented** in `generate_folds.py` as the primary fold strategy. Three metros (NYC, New Orleans, SW Florida) already use county blocking; Houston's folds are stale (132 vs 216 ZCTAs) and will auto-switch to county blocking upon regeneration.

**PAR Verdict: Publishable only with within-metro design (not cross-metro comparison).**

## Current State

| Metro | Current Strategy | Counties | ZCTAs | Notes |
|-------|-----------------|----------|-------|-------|
| Houston | zip3 (STALE) | 6 | 216 | Will switch to county on regeneration |
| New Orleans | county | 5 | 66 | Active |
| NYC | county | 5 | 211 | Active |
| Riverside | zip3 | <5 | 86 | Correct fallback |
| SW Florida | county | 6 | 202 | Active |

## PAR Findings (Aggregated)

### Critical (Both Reviewers Agree)

1. **Cross-metro comparison is a fatal confound.** Comparing uplift direction across metros that happen to use different blocking schemes conflates metro heterogeneity (coastal vs riverine, insurance penetration, urban density) with blocking-scheme effects. A reviewer will note: "You cannot attribute consistent uplift to blocking robustness when blocking and metro co-vary perfectly." Both reviewers flagged this independently.

2. **Harris County mega-block split produces a hybrid strategy, not pure county blocking.** Harris County holds ~130/216 ZCTAs (60%). The mega-block split at 40% (line 172-199 of generate_folds.py) will activate, splitting Harris into ZIP4 sub-blocks. Reporting this as "county blocking" would be misleading — it is county + ZIP4 hybrid.

3. **Stale folds invalidate comparison.** DOE Amendment v1.12 states "all prior Houston and SW Florida results invalidated." There is no valid ZIP3 baseline for Houston to compare the new county-blocked folds against.

### Serious (Both Reviewers Agree)

4. **The paper already concludes ZIP3 is "the only viable scheme" (07b line 236).** Publishing county blocking results contradicts this without acknowledging the revision.

5. **No formal statistical test.** "Same uplift direction" is qualitative, not a hypothesis test. A reviewer will ask for null hypothesis, test statistic, and power.

6. **Extreme fold imbalance.** 6 counties for 216 ZCTAs with Harris at 60% means one fold dominates. Wilcoxon signed-rank on 5 folds with wildly different sizes is unreliable.

## The Valid Design (Reviewer Consensus)

**Within-metro comparison:** Run Houston under BOTH county blocking AND ZIP3 blocking, compare uplift direction. The code already supports forcing either strategy.

This is a clean, within-subject robustness test that holds the data-generating process constant.

## Implementation

### What Already Exists
- `generate_folds.py` — full county blocking logic with crosswalk loading
- `zcta_county_crosswalk.parquet` on S3
- Mega-block splitting for oversized counties
- ZIP3 fallback path

### New Code Needed

A dual-strategy fold generator that produces BOTH fold sets for the same metro:

```python
"""
generate_dual_folds.py — Within-metro robustness fold generator.

Produces two fold assignments for the same ZCTA set:
  1. County-blocked folds (primary strategy from generate_folds.py)
  2. ZIP3-blocked folds (forced, regardless of county availability)

This enables within-metro comparison of uplift direction under
different blocking geometries, holding the data constant.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
N_FOLDS = 5
MIN_BLOCK_ZCTAS = 5
MEGA_BLOCK_PCT = 0.40  # Split blocks exceeding this fraction


def assign_county_folds(
    zctas: list[str],
    crosswalk: pd.DataFrame,
    n_folds: int = N_FOLDS,
) -> dict[str, int]:
    """Assign ZCTAs to folds using county boundaries.

    Args:
        zctas: List of ZCTA IDs in the metro.
        crosswalk: DataFrame with columns [zcta_id, county_fips].
        n_folds: Target number of folds.

    Returns:
        Dict mapping zcta_id -> fold_index (0-based).
    """
    # Filter crosswalk to metro ZCTAs
    metro_xw = crosswalk[crosswalk["zcta_id"].isin(zctas)].copy()
    county_groups = metro_xw.groupby("county_fips")["zcta_id"].apply(list).to_dict()

    # Sort counties by size (descending) for bin-packing
    counties_sorted = sorted(county_groups.items(), key=lambda x: -len(x[1]))

    # Handle mega-blocks: split counties > 40% into ZIP4 sub-blocks
    blocks = {}
    for county_fips, county_zctas in counties_sorted:
        if len(county_zctas) / len(zctas) > MEGA_BLOCK_PCT:
            # Split by ZIP4 prefix (first 4 chars of ZCTA)
            zip4_groups = {}
            for z in county_zctas:
                prefix = z[:4]
                zip4_groups.setdefault(prefix, []).append(z)
            for prefix, group in zip4_groups.items():
                blocks[f"{county_fips}_{prefix}"] = group
        else:
            blocks[county_fips] = county_zctas

    # Merge small blocks (< MIN_BLOCK_ZCTAS) into largest neighbor
    final_blocks = _merge_small_blocks(blocks)

    # Greedy bin-packing into n_folds
    return _bin_pack_to_folds(final_blocks, n_folds)


def assign_zip3_folds(
    zctas: list[str],
    n_folds: int = N_FOLDS,
) -> dict[str, int]:
    """Assign ZCTAs to folds using ZIP3 prefix grouping.

    Args:
        zctas: List of ZCTA IDs in the metro.
        n_folds: Target number of folds.

    Returns:
        Dict mapping zcta_id -> fold_index (0-based).
    """
    # Group by first 3 characters
    zip3_groups = {}
    for z in zctas:
        prefix = z[:3]
        zip3_groups.setdefault(prefix, []).append(z)

    # Merge small groups
    final_blocks = _merge_small_blocks(zip3_groups)

    # Bin-pack
    return _bin_pack_to_folds(final_blocks, n_folds)


def generate_dual_folds(
    scenario: str,
    zctas: list[str],
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Generate both county and ZIP3 fold assignments for a metro.

    Returns:
        DataFrame with columns [zcta_id, fold_county, fold_zip3].
    """
    county_folds = assign_county_folds(zctas, crosswalk)
    zip3_folds = assign_zip3_folds(zctas)

    df = pd.DataFrame({"zcta_id": zctas})
    df["fold_county"] = df["zcta_id"].map(county_folds)
    df["fold_zip3"] = df["zcta_id"].map(zip3_folds)
    df["scenario"] = scenario

    # Metadata for auditability
    meta = {
        "scenario": scenario,
        "n_zctas": len(zctas),
        "n_folds": N_FOLDS,
        "county_blocks": len(set(county_folds.values())),
        "zip3_blocks": len(set(zip3_folds.values())),
        "county_balance": _balance_stats(county_folds),
        "zip3_balance": _balance_stats(zip3_folds),
    }

    return df, meta


def _merge_small_blocks(blocks: dict) -> dict:
    """Merge blocks with fewer than MIN_BLOCK_ZCTAS into largest block."""
    sorted_blocks = sorted(blocks.items(), key=lambda x: -len(x[1]))
    final = {}
    largest_key = sorted_blocks[0][0] if sorted_blocks else None

    for key, zctas in sorted_blocks:
        if len(zctas) < MIN_BLOCK_ZCTAS and key != largest_key:
            # Merge into largest
            if largest_key in final:
                final[largest_key].extend(zctas)
            else:
                final.setdefault(largest_key, []).extend(zctas)
        else:
            final.setdefault(key, []).extend(zctas)

    return final


def _bin_pack_to_folds(blocks: dict, n_folds: int) -> dict[str, int]:
    """Greedy bin-packing: assign blocks to folds, smallest-first into lightest fold."""
    # Sort blocks by size descending
    sorted_blocks = sorted(blocks.items(), key=lambda x: -len(x[1]))
    fold_sizes = [0] * n_folds
    fold_assignment = {}  # block_key -> fold_index

    for key, zctas in sorted_blocks:
        # Assign to lightest fold
        lightest = int(np.argmin(fold_sizes))
        fold_assignment[key] = lightest
        fold_sizes[lightest] += len(zctas)

    # Map ZCTAs to folds
    zcta_folds = {}
    for key, zctas in blocks.items():
        fold_idx = fold_assignment[key]
        for z in zctas:
            zcta_folds[z] = fold_idx

    return zcta_folds


def _balance_stats(folds: dict[str, int]) -> dict:
    """Compute fold balance statistics."""
    from collections import Counter
    counts = Counter(folds.values())
    sizes = list(counts.values())
    return {
        "min": min(sizes),
        "max": max(sizes),
        "ratio": max(sizes) / min(sizes) if min(sizes) > 0 else float("inf"),
    }
```

### Integration with Existing Training Pipeline

```python
# In the SageMaker job script, after generating dual folds:
from generate_dual_folds import generate_dual_folds

# Load crosswalk
crosswalk = pd.read_parquet("s3://swarm-floodrsct-data/raw/geocertdb2026/zcta_county_crosswalk.parquet")

# Generate both fold sets
df_folds, meta = generate_dual_folds("houston", zcta_list, crosswalk)

# Train R0/R1/R2 under county blocking
train_model(features, labels, folds=df_folds["fold_county"], strategy="county")

# Train R0/R1/R2 under ZIP3 blocking
train_model(features, labels, folds=df_folds["fold_zip3"], strategy="zip3")

# Compare uplift direction
# uplift_county = metric_R1_county - metric_R0_county
# uplift_zip3 = metric_R1_zip3 - metric_R0_zip3
# Robustness = sign(uplift_county) == sign(uplift_zip3)
```

## PhD-Level Execution Plan

1. **Regenerate Houston folds** with current 216-ZCTA dataset (will auto-switch to county)
2. **Force ZIP3 folds** for the same Houston dataset (override the county logic)
3. **Train R0, R1, R2** under both fold sets (6 models total)
4. **Report uplift direction** under both blocking schemes in a 2x3 table
5. **Statistical test:** Sign test or bootstrap CI on uplift difference across folds
6. **Extend to SW Florida** (6 counties, also supports both strategies)
7. **Report balance statistics** for both strategies (fold size ratio, min/max)

## Paper Integration

Revise `07b_spatial_diagnostics.tex` lines 236-238 from:
> "ZIP3 prefix blocking is the only viable spatial blocking scheme"

To:
> "ZIP3 prefix blocking and county blocking both produce non-degenerate folds for metros with >= 5 administrative units. Within-metro comparison (Houston: county vs ZIP3) confirms uplift direction is preserved across blocking geometries (Table X)."

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Harris County mega-split makes it "not really county" | Report honestly as "administrative blocking with refinement" |
| Only 1-2 metros support both strategies | Report as case study, not universal claim |
| Fold imbalance under county | Report balance statistics; use weighted metrics |
| No formal test at n=5 folds | Use bootstrap CI on per-fold uplift, not just sign |
