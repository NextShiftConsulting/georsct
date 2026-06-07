# Proposal: Larger K with Hierarchical Merge

**Date:** 2026-06-07
**Context:** DOE Appendix R — Spatial blocking geometry robustness
**PAR Review:** Completed (2 independent reviewers, aggregated below)

## Executive Summary

Run Skater at K=20 to over-partition the homogeneous mass, then hierarchically merge adjacent regions down to K=5 with a balance constraint (no fold > 40%).

**PAR Verdict: NOT RECOMMENDED. Produces pseudo-random partitions indistinguishable from random spatial blocking when features are homogeneous.**

## PAR Findings (Aggregated)

### Critical (Reviewer A — Confirmed by Reviewer B)

1. **Over-partitioning homogeneous features produces arbitrary splits indistinguishable from random spatial partitioning.** Skater minimizes within-cluster feature variance on an MST. When variance is near-zero across the graph (documented in findings), cuts at K=20 are determined by numerical noise in MST edge weights, not real feature gradients. A reviewer would correctly ask: "How does K=20 Skater + merge differ from randomly assigning contiguous blocks?" The answer: it doesn't, in any statistically distinguishable way.

2. **The merge step destroys any information from the over-partition.** Houston: K=5 produces {118, 8, 2, 2, 1}. K=20 would split the 118-ZCTA mass into ~15 arbitrary sub-regions of ~8 each. Merging "most-similar adjacent pairs" recombines them into... the same homogeneous mass (all sub-regions are equally similar by construction). Final K=5 result converges to the original degenerate partition.

3. **N=17 (New Orleans) makes K=20 impossible.** Any implementation must special-case small metros, meaning the methodology is not uniform across metros and cross-metro comparison is invalidated.

### Serious (Both Reviewers Agree)

4. **K_initial is a free parameter with no principled selection rule.** K=15 produces different final regions than K=20 or K=25. Since splits are noise-driven in homogeneous feature space, there is no convergence across K_initial values. A reviewer would demand sensitivity analysis that will show instability.

5. **No literature precedent for this specific failure mode.** "Over-partition + merge" exists in image segmentation (watershed) but has not been established as sound for spatial CV fold assignment. The regionalization literature (Duque et al. 2007, Assuncao et al. 2006) assumes sufficient spatial heterogeneity.

6. **Balance constraint (40%) may deadlock.** In Houston, K=20 with homogeneous features likely produces one region of ~100 ZCTAs + 19 singletons. Merging singletons produces one large region (100+ ZCTAs, >40%). The constraint blocks this, but no alternative merge path exists without a split operation (not specified).

7. **wlag NaN confound still applies.** Contiguous regions under any algorithm still produce NaN spatial lag features for interior test ZCTAs.

8. **The approach cannot answer the paper's core question.** Even if balanced folds are achieved, R1 hydrology models (which use wlag) will have missing predictors for test observations surrounded by other test ZCTAs.

### Minor

- "Hierarchical merge" is technically single-linkage agglomerative clustering with contiguity constraint — should be named as such
- The 40% balance threshold is not derived from any statistical principle
- spopt Skater API parameter name varies across versions (`n_clusters` vs `k`)

## If Pursued Despite PAR Verdict

The ONLY defensible version would need to:
1. **Use different features** with actual spatial heterogeneity (e.g., elevation, distance-to-coast, urbanization density)
2. **Prove non-randomness** via a permutation test comparing the partition's feature-homogeneity to random contiguous partitions
3. **Report K_initial sensitivity** showing stability across {10, 15, 20, 25, 30}
4. **Exclude New Orleans** (N=17 is too small for any regionalization)
5. **Address wlag NaN** by using only R0 features (no spatial lag) in the robustness comparison

### Code (If Pursued)

```python
"""
skater_overpartition_merge.py — Two-stage regionalization.

Stage 1: Over-partition with Skater at high K to fragment the
         homogeneous mass into smaller contiguous pieces.
Stage 2: Hierarchical merge of most-similar adjacent pairs down
         to target K, with balance constraint.

WARNING: PAR review determined this produces pseudo-random partitions
when features are homogeneous (the documented failure mode for all
5 metros). Use only if features are replaced with higher-variance
alternatives.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict


# --- Configuration ---
TARGET_K = 5
MAX_FOLD_FRACTION = 0.40  # No fold may exceed this fraction of total ZCTAs
MIN_ZCTAS_FOR_OVERPARTITION = 30  # Below this, skip to direct K assignment


def overpartition_and_merge(
    w,  # libpysal W object
    features: np.ndarray,
    zctas: list[str],
    k_initial: int = 20,
    k_final: int = TARGET_K,
    max_fold_frac: float = MAX_FOLD_FRACTION,
) -> dict[str, int]:
    """Two-stage regionalization: over-partition then merge.

    Args:
        w: Spatial weights (libpysal W, must be connected).
        features: Scaled feature matrix (n_zctas x n_features).
        zctas: ZCTA identifiers matching feature rows.
        k_initial: Number of initial Skater partitions.
        k_final: Target number of final regions.
        max_fold_frac: Maximum fraction of ZCTAs in any region.

    Returns:
        Dict mapping zcta_id -> region_index (0-based).

    Raises:
        ValueError: If n_zctas < MIN_ZCTAS_FOR_OVERPARTITION.
    """
    n = len(zctas)

    if n < MIN_ZCTAS_FOR_OVERPARTITION:
        raise ValueError(
            f"Too few ZCTAs ({n}) for over-partition. "
            f"Minimum is {MIN_ZCTAS_FOR_OVERPARTITION}."
        )

    # Clamp k_initial to feasible range
    k_initial = min(k_initial, n // 2)

    # Stage 1: Over-partition with Skater
    from spopt.region import Skater as SkaterAlgo
    from sklearn.preprocessing import StandardScaler

    # Skater needs a GeoDataFrame-like object with scaled attributes
    import geopandas as gpd
    from shapely.geometry import Point

    # Build minimal GeoDataFrame (Skater needs geometry for MST)
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(features, columns=[f"f{i}" for i in range(features.shape[1])]),
        geometry=[Point(0, 0)] * n,  # Dummy — weights override geometry
    )
    feature_cols = [f"f{i}" for i in range(features.shape[1])]

    model = SkaterAlgo(gdf, w, attrs_name=feature_cols, n_clusters=k_initial)
    initial_labels = np.array(model.labels_)

    # Stage 2: Hierarchical merge
    region_labels = _hierarchical_merge(
        initial_labels=initial_labels,
        features=features,
        w=w,
        zctas=zctas,
        k_final=k_final,
        max_fold_frac=max_fold_frac,
    )

    return dict(zip(zctas, region_labels))


def _hierarchical_merge(
    initial_labels: np.ndarray,
    features: np.ndarray,
    w,
    zctas: list[str],
    k_final: int,
    max_fold_frac: float,
) -> np.ndarray:
    """Merge regions greedily until k_final remain.

    Merge criterion: smallest feature-distance between adjacent regions.
    Balance constraint: merged region cannot exceed max_fold_frac * n.
    """
    n = len(zctas)
    max_size = int(max_fold_frac * n)

    # Build region state
    labels = initial_labels.copy()
    unique_labels = list(set(labels))

    # Precompute region centroids (mean feature vector per region)
    def compute_centroids(labels):
        centroids = {}
        for lbl in set(labels):
            mask = labels == lbl
            centroids[lbl] = features[mask].mean(axis=0)
        return centroids

    # Precompute region adjacency from W
    def compute_adjacency(labels, w, zctas):
        adj = defaultdict(set)
        for i, zi in enumerate(zctas):
            for j_idx in w.neighbors[zi]:
                j = zctas.index(j_idx) if isinstance(j_idx, str) else j_idx
                li, lj = labels[i], labels[j]
                if li != lj:
                    adj[li].add(lj)
                    adj[lj].add(li)
        return adj

    # Iterative merge
    while len(set(labels)) > k_final:
        centroids = compute_centroids(labels)
        adjacency = compute_adjacency(labels, w, zctas)
        region_sizes = {lbl: np.sum(labels == lbl) for lbl in set(labels)}

        # Find best merge pair (smallest distance, respecting balance)
        best_pair = None
        best_dist = float("inf")

        for lbl_a, neighbors in adjacency.items():
            for lbl_b in neighbors:
                if lbl_a >= lbl_b:
                    continue  # Avoid duplicates
                merged_size = region_sizes[lbl_a] + region_sizes[lbl_b]
                if merged_size > max_size:
                    continue  # Balance constraint
                dist = np.linalg.norm(centroids[lbl_a] - centroids[lbl_b])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (lbl_a, lbl_b)

        if best_pair is None:
            # No valid merge possible (all would violate balance)
            break

        # Execute merge: relabel lbl_b -> lbl_a
        lbl_a, lbl_b = best_pair
        labels[labels == lbl_b] = lbl_a

    # Relabel to 0-based contiguous
    unique_final = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique_final)}
    labels = np.array([remap[l] for l in labels])

    return labels


def validate_partition(
    labels: np.ndarray,
    zctas: list[str],
    k_target: int,
    max_fold_frac: float,
) -> dict:
    """Validate the partition meets constraints.

    Returns:
        Dict with validation results and balance statistics.
    """
    from collections import Counter
    counts = Counter(labels)
    n = len(zctas)
    sizes = sorted(counts.values(), reverse=True)

    return {
        "n_regions": len(counts),
        "target_k": k_target,
        "k_achieved": len(counts) == k_target,
        "sizes": sizes,
        "max_fraction": sizes[0] / n,
        "balance_ok": sizes[0] / n <= max_fold_frac,
        "min_fold_size": sizes[-1],
        "imbalance_ratio": sizes[0] / sizes[-1] if sizes[-1] > 0 else float("inf"),
    }
```

### Permutation Test for Non-Randomness

```python
"""
test_partition_nonrandomness.py — Permutation test to verify
that the over-partition + merge result is non-random.

Compare within-region feature variance of the actual partition
against null distribution from random contiguous partitions.
"""

import numpy as np
from scipy import stats


def random_contiguous_partition(w, n_zctas: int, k: int, n_iter: int = 100) -> list:
    """Generate random contiguous partitions via BFS growth.

    Starts k random seeds and grows regions by BFS until all
    ZCTAs are assigned. Balance is approximate.
    """
    import random
    partitions = []

    for _ in range(n_iter):
        zcta_ids = list(range(n_zctas))
        seeds = random.sample(zcta_ids, k)
        labels = [-1] * n_zctas
        queues = [[s] for s in seeds]

        for i, s in enumerate(seeds):
            labels[s] = i

        # BFS growth
        while any(l == -1 for l in labels):
            for i in range(k):
                if not queues[i]:
                    continue
                current = queues[i].pop(0)
                for neighbor in w.neighbors[current]:
                    if labels[neighbor] == -1:
                        labels[neighbor] = i
                        queues[i].append(neighbor)

        partitions.append(np.array(labels))

    return partitions


def permutation_test_variance(
    features: np.ndarray,
    actual_labels: np.ndarray,
    random_partitions: list[np.ndarray],
) -> dict:
    """Test whether actual partition has lower within-region variance than random.

    Returns:
        p-value and effect size.
    """
    def within_region_variance(labels):
        """Total within-region sum of squares."""
        wss = 0.0
        for lbl in set(labels):
            mask = labels == lbl
            if mask.sum() > 1:
                wss += features[mask].var(axis=0).sum() * mask.sum()
        return wss

    actual_wss = within_region_variance(actual_labels)
    null_wss = [within_region_variance(p) for p in random_partitions]

    # One-sided test: actual WSS should be LOWER than random
    p_value = np.mean([nw <= actual_wss for nw in null_wss])

    return {
        "actual_wss": actual_wss,
        "null_mean": np.mean(null_wss),
        "null_std": np.std(null_wss),
        "p_value": p_value,
        "effect_size": (np.mean(null_wss) - actual_wss) / np.std(null_wss),
        "is_nonrandom": p_value < 0.05,
    }
```

## Why This Is NOT the Recommended Path

| Criterion | Assessment |
|-----------|-----------|
| Addresses root cause? | **No** — feature homogeneity unchanged |
| Produces meaningful regions? | **No** — splits are noise-driven |
| Literature precedent? | **No** — not established for spatial CV |
| Stable across K_initial? | **No** — expected instability |
| Works for all metros? | **No** — N=17 excluded |
| Better than random? | **Unfalsifiable** — permutation test will likely fail |

## When This WOULD Work

This approach is valid when:
- Features have genuine spatial heterogeneity (e.g., elevation gradients, urban/rural transitions)
- The dataset is large enough (N > 100 ZCTAs) for K=20 to be feasible
- The permutation test confirms the partition is non-random (p < 0.05)

**For this dataset:** The three structural features (flood zone %, TWI, slope) are too homogeneous within metros to support meaningful regionalization at any K.

## Recommendation

**Do not pursue.** The county blocking within-metro comparison (proposal-county-blocking.md) is the only approach that produces a publishable robustness result for this specific dataset. Report the Skater degeneracy and large-K merge failure as evidence that feature-driven regionalization is inappropriate at metro scale — this is itself a publishable finding for SIGSPATIAL.

If a PhD student insists on pursuing this for methodological novelty, the permutation test code above provides the falsification criterion: if the actual partition's WSS is not significantly lower than random contiguous partitions, the approach is declared invalid and the negative result is reported.
