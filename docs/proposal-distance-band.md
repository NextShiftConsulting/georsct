# Proposal: Distance-Band Spatial Weights for Skater

**Date:** 2026-06-07
**Context:** DOE Appendix R — Spatial blocking geometry robustness
**PAR Review:** Completed (2 independent reviewers, aggregated below)

## Executive Summary

Replace Queen contiguity with distance-band weights to fix the disconnected graph problem in NYC (6 components) and SW Florida (3 components), then re-run Skater regionalization.

**PAR Verdict: NOT RECOMMENDED. Does not address root cause (feature homogeneity). Would not survive peer review.**

## PAR Findings (Aggregated)

### Critical (Both Reviewers Agree)

1. **Does not address the root cause.** The documented failure is feature HOMOGENEITY, not graph disconnection. Houston and Riverside already have fully connected Queen graphs (1 component each) yet produce 90-95% degenerate regions. Distance-band solves a secondary symptom (disconnection in 2/5 metros) while the primary pathology (one giant homogeneous region) remains untouched. A reviewer would reject in one sentence: "The authors propose solving a connectivity problem, but their own Table shows degeneracy occurs in fully-connected metros."

2. **Unprojected lat/lon degrees is geodetically invalid.** A fixed threshold of 0.5 degrees means different physical distances at different latitudes (48 km at lat 30 vs 42 km at lat 41). This introduces systematic latitude-dependent bias across metros. SIGSPATIAL reviewers will immediately flag this as a violation of isotropic distance assumptions.

### Serious (Both Reviewers Agree)

3. **Threshold selection is arbitrary and unjustifiable.** "3.4 avg neighbors comparable to Queen" is circular — Queen already fails. No sensitivity analysis proposed. KNN weights are strictly superior (deterministic degree, no threshold parameter) yet not discussed.

4. **Connects non-adjacent ZCTAs across geographic barriers.** 0.5 degrees (~55 km) connects Manhattan to New Jersey, barrier islands to mainland. These connections violate spatial coherence for flood-risk applications where hydrological barriers are real discontinuities.

5. **No publishable robustness result even if it works.** If DistanceBand still produces degenerate regions (likely for 3/5 metros), the paper gains nothing. If it works for NYC/SWFL only, you have a 2/5 partial result that raises more questions than it answers.

6. **Denser graph makes degeneracy WORSE.** More edges give Skater more freedom to merge similar-feature ZCTAs into one mega-region. For the 3 connected metros, Option 2 is guaranteed to reproduce or worsen the degenerate outcome.

7. **wlag NaN confound still applies.** Spatial lag features become NaN for interior test ZCTAs under any contiguous blocking, regardless of weight structure.

### Minor

- Coordinate data exists (latitude/longitude in `zcta_features_labels.parquet`) — no new data upload needed
- Code change is minimal (~10 lines to swap `build_weights_from_adjacency` for `DistanceBand`)
- KNN is the standard fix in spatial econometrics for disconnected graphs; not discussing it is a gap

## If Pursued Despite PAR Verdict

The ONLY defensible version would:
1. Use **projected coordinates** (EPSG:5070 Albers Equal Area, already used elsewhere in the pipeline)
2. Use **KNN weights** (K=8) instead of DistanceBand (no threshold parameter)
3. Combine with **Option 3** (larger K + merge) to address the balance problem
4. Include a **sensitivity analysis** across K in {4, 6, 8, 12}
5. Acknowledge in the paper that this addresses disconnection only, not homogeneity

### Code (If Pursued)

```python
"""
build_knn_weights.py — KNN spatial weights from ZCTA centroids.

Produces a connected W-matrix for Skater regionalization using
K-nearest-neighbor weights on projected coordinates (EPSG:5070).

NOTE: PAR review determined this does NOT address the root cause
of Skater degeneracy (feature homogeneity). Use only in combination
with balance-constrained regionalization (proposal-large-k-merge.md).
"""

import numpy as np
import pandas as pd
from libpysal.weights import KNN
from pyproj import Transformer


# WGS84 -> Albers Equal Area CONUS
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)


def build_knn_weights(
    zctas: list[str],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    k: int = 8,
) -> "libpysal.weights.W":
    """Build KNN spatial weights from ZCTA centroids.

    Args:
        zctas: ZCTA identifiers (used as W.id_order).
        latitudes: WGS84 latitude array.
        longitudes: WGS84 longitude array.
        k: Number of nearest neighbors.

    Returns:
        libpysal W object with KNN connectivity.
    """
    # Project to Albers Equal Area (meters)
    x_proj, y_proj = TRANSFORMER.transform(longitudes, latitudes)
    coords = np.column_stack([x_proj, y_proj])

    # Build KNN weights
    w = KNN(coords, k=k, ids=zctas, silence_warnings=True)

    # Symmetrize (KNN is asymmetric; Skater needs symmetric graph)
    w_sym = w.symmetrize()

    return w_sym


def load_metro_centroids(
    features_path: str,
    metro_zctas: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load lat/lon for metro ZCTAs from features parquet.

    Args:
        features_path: Path to zcta_features_labels.parquet.
        metro_zctas: List of ZCTA IDs in the metro.

    Returns:
        (latitudes, longitudes) arrays.
    """
    df = pd.read_parquet(
        features_path,
        columns=["zcta_id", "latitude", "longitude"],
    )
    df = df[df["zcta_id"].isin(metro_zctas)].set_index("zcta_id")
    df = df.loc[metro_zctas]  # Preserve order

    return df["latitude"].values, df["longitude"].values
```

### Integration Point

```python
# In compute_spatial_sidecar_regionalize.py, replace:
#   w = build_weights_from_adjacency(adj_df, metro_zctas)
# With:
from build_knn_weights import build_knn_weights, load_metro_centroids

lats, lons = load_metro_centroids(features_path, metro_zctas)
w = build_knn_weights(metro_zctas, lats, lons, k=8)
```

## Why This Is NOT the Recommended Path

| Problem | Distance-Band/KNN Fixes It? |
|---------|:---:|
| Disconnected graph (NYC, SWFL) | Yes |
| Feature homogeneity (ALL metros) | **No** |
| One giant region (Houston, Riverside, NOLA) | **No** |
| wlag NaN confound | **No** |
| Threshold/K sensitivity | Partially (KNN better than DistanceBand) |

**Bottom line:** This is engineering a fix for a secondary symptom. The publishable finding is that Skater regionalization fails due to feature homogeneity at metro scale — changing the weight structure cannot resolve this. Report the finding; don't chase a fix.

## Recommendation

**Do not pursue as standalone approach.** If any weight-structure work is done, combine with Option 3 (large-K merge) and use KNN on projected coordinates. But the county blocking within-metro comparison (proposal-county-blocking.md) is the higher-value, lower-risk path to a publishable robustness result.
