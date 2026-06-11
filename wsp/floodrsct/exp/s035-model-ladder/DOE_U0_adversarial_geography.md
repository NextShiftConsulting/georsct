# DOE: U0 Adversarial Geography -- Constructive Existence Proof for Gate 3B

**Experiment:** s035-model-ladder / Gate 3B Validation
**Role:** Constructive adversarial test -- build the pathology, verify the gate catches it
**Status:** DESIGNED
**Depends on:** R0 complete (baseline predictions), floodcaster.batch features wired (v24.009)
**Blocks:** Gate 3B promotion from EXPERIMENTAL to ENFORCED

---

## Motivation

Gate 3B (kappa_reconstruct) detects representations with non-planar implied
topology -- the "taxi flyover" pathology where a model asserts adjacency
between regions that cannot be neighbors in 2D geographic space.

The gate exists. The instrument (`kappa_reconstruct.py`) is implemented. But
we have no constructive proof that the gate fires on a controlled pathology
and passes on the real substrate. Without this, Gate 3B stays EXPERIMENTAL.

**The commitment:** FloodRSCT tasks are spatial-reasoning tasks. Geographic
topology constrains physically valid outcomes. A representation that cannot
recover that topology cannot reason about flood risk correctly.

---

## Hypothesis

**H_U0 (primary):** A substrate trained on permuted-geography features
(topology destroyed) achieves comparable forward accuracy to the real
substrate but produces significantly lower kappa_reconstruct.

**Formal:** Let G_real be the substrate with true centroid coordinates and
G_perm be the substrate with permuted centroid-to-coordinate assignments.

- H0: kappa_reconstruct(G_real) = kappa_reconstruct(G_perm)
- H1: kappa_reconstruct(G_real) > kappa_reconstruct(G_perm)

**Secondary (S_U0a):** The adversarial permutation orthogonality test
(`adversarial_geography_permutation`) yields `earns_gate=True` -- the
kappa_reconstruct drop under permutation is not explained by kappa_spatial
decline alone.

**Secondary (S_U0b):** Topology-derived features (HAND, GFI, SAR anomaly)
degrade more under permutation than covariate features (ACS demographics),
confirming that the permutation specifically attacks spatial reasoning, not
general statistical structure.

---

## Design

### Treatment Arms

| Arm | Name | Centroid Mapping | Topology-Derived Features | Expected kappa_reconstruct |
|-----|------|-----------------|--------------------------|---------------------------|
| A | Real | True ZCTA centroids | Valid HAND, GFI, SAR, Deltares | High (>0.5) |
| B | Permuted | Shuffled ZCTA-to-centroid | Nonsensical HAND, GFI, SAR, Deltares | Low (<0.3) |
| C | Block-permuted | Permuted within 200km blocks | Locally plausible, globally broken | Medium |
| D | Covariate-only | True centroids, no topology features | No HAND/GFI/SPI/TWI/SAR/Deltares | Baseline reference |

### Why These Arms

- **A vs B:** The core test. If kappa_reconstruct distinguishes A from B, the
  gate catches total topology destruction.
- **A vs C:** The sensitivity test. Block permutation preserves local
  autocorrelation (Tobler signal) but breaks drainage basin boundaries. If
  kappa_reconstruct still drops, it measures topology, not just correlation.
- **A vs D:** The absence test. Without topology features, does the substrate
  still achieve reasonable kappa_reconstruct from the W-matrix and spatial
  splits alone? Establishes which features carry the topology signal.

### Permutation Procedure

The permutation targets the centroid-to-ZCTA mapping used by
`floodcaster.batch` functions. All topology-derived features are recomputed
from the permuted coordinates:

```
Real pipeline:
  ZCTA_70112 -> centroid (30.0, -90.0) -> fetch DEM -> compute HAND=5.2m

Permuted pipeline (Arm B):
  ZCTA_70112 -> centroid (33.4, -84.1) [Atlanta coords] -> fetch DEM -> compute HAND=142.0m
  (ZCTA_70112 is in New Orleans but gets Atlanta's terrain)
```

**Implementation:** Override centroid DataFrame before passing to batch
functions. The permutation is applied at the centroid level, not the feature
level -- features are recomputed honestly from wrong coordinates, producing
physically impossible but internally consistent values.

### Features Affected by Permutation

| Feature | Source | Permutation Effect |
|---------|--------|-------------------|
| hand_mean_m | DEM flow routing | Completely wrong drainage |
| twi_mean | DEM slope + flow acc | Wrong terrain |
| gfi_mean | DEM flow acc / HAND | Doubly wrong |
| spi_mean | DEM flow acc * slope | Wrong slope |
| deltares_depth_ft_rp{10,50,100} | Deltares flood model | Wrong flood zone |
| deltares_max_depth_ft_rp{10,50,100} | Deltares flood model | Wrong flood zone |
| deltares_inundation_pct_rp{10,50,100} | Deltares flood model | Wrong flood zone |
| sar_water_pct | Sentinel-1 SAR | Wrong location entirely |
| sar_water_pct_anomaly | SAR - JRC baseline | Nonsensical difference |
| W-matrix spatial lags | Haversine neighbor weights | Wrong neighbors |

| Feature | Source | NOT Affected |
|---------|--------|-------------|
| obs_nfip_event_claims | NFIP claims data | Keyed on ZCTA, not centroid |
| population_total | ACS census | Keyed on ZCTA |
| svi_* | CDC SVI | Keyed on ZCTA |
| nwis_max_stage_ft | USGS gauges | Gauge-to-ZCTA uses centroids (affected in full permutation, but NOT in covariate-only arm) |

### Evaluation Metrics (per arm)

| Metric | Role | Source |
|--------|------|--------|
| kappa_reconstruct | **Primary gate** | `compute_kappa_reconstruct(embeddings, coords2d)` |
| forward_score | Control variable | R0 Spearman rho on held-out fold |
| kappa_spatial | Orthogonality check | Moran's I on residuals |
| n_crossings | Diagnostic | Gabriel edge crossing count |
| mantel_r | Correlational context | Distance matrix correlation |
| coordinate_lift_score | SatCLIP guard | Coordinate proximity explanation |
| feature_importance_shift | **Secondary S_U0b** | SHAP or permutation importance delta for topology features |

### Adversarial Permutation Wiring

Use `adversarial_geography_permutation()` from `kappa_reconstruct.py`:

```python
from georsct.domain.kappa_reconstruct import (
    adversarial_geography_permutation,
    compute_kappa_reconstruct,
)

def refit_and_embed_fn(trial_idx):
    """Adapter: retrain with permuted W-matrix, return embeddings + metrics."""
    # 1. Permute centroid-to-ZCTA mapping (seed=trial_idx for reproducibility)
    perm_centroids = permute_centroids(real_centroids, seed=trial_idx)
    # 2. Recompute topology features from permuted centroids
    perm_features = recompute_topology_features(perm_centroids, s3)
    # 3. Retrain R1 model with permuted features
    model, embeddings = train_r1(base_features, perm_features, folds)
    # 4. Compute metrics
    fwd = evaluate_forward(model, test_folds)
    ks = compute_kappa_spatial(model, test_folds)
    return embeddings, fwd, ks

result = adversarial_geography_permutation(
    refit_and_embed_fn=refit_and_embed_fn,
    baseline_reconstruct=real_kappa_reconstruct,
    baseline_spatial=real_kappa_spatial,
    coords2d=real_coords2d,
    n_trials=10,
)

# Primary outcome:
assert result.earns_gate is True, "Gate 3B does not fire -- kappa_reconstruct may be redundant"
```

---

## Pipeline

### Phase 0: Feature Extraction (per arm)

For Arms A, B, C: run `floodcaster.batch` with appropriate centroid DataFrame.

```python
# Arm A: real centroids (already cached from build_event_dataset)
deltares_real = build_deltares_depth_features(s3, zcta_ids)
hydrology_real = build_hydrology_features(s3, zcta_ids)

# Arm B: permuted centroids
# Override the centroid source before builder functions read it
perm_centroids = permute_zcta_centroids(real_centroids, mode="global", seed=42)
# Write permuted centroids to a temp parquet that the builder functions will read
# OR: call floodcaster.batch functions directly with the permuted DataFrame
deltares_perm = deltares_centroid_depth(perm_centroids, return_periods=[10, 50, 100])
hydrology_perm = hydrology_centroid_stats(perm_centroids)

# Arm C: block-permuted centroids
block_perm_centroids = permute_zcta_centroids(real_centroids, mode="block", block_km=200, seed=42)
deltares_block = deltares_centroid_depth(block_perm_centroids, return_periods=[10, 50, 100])
hydrology_block = hydrology_centroid_stats(block_perm_centroids)

# Arm D: no topology features (use R0 feature set only)
```

### Phase 1: Training (per arm x scenario x target x fold)

Use existing R1 training infrastructure. Only the feature matrix changes.

| Arm | Feature Set |
|-----|-------------|
| A | R0 + R1 universal + R1 scenario + **Planetary Computer features (real)** |
| B | R0 + R1 universal + R1 scenario + **Planetary Computer features (permuted)** |
| C | R0 + R1 universal + R1 scenario + **Planetary Computer features (block-permuted)** |
| D | R0 features only (no topology) |

Solver: HistGBDT (primary), Ridge (sensitivity).
Splits: spatial_blocked (primary), random (sensitivity).
Folds: same R0 fold assignments.

### Phase 2: Gate 3B Evaluation

For each arm, compute:

```python
result = compute_kappa_reconstruct(
    embeddings=shap_values,     # or PCA of learned feature importance
    coords2d=real_coords2d,     # ALWAYS real coords (we're testing the representation)
    n_baseline_trials=20,
    n_mantel_perms=999,
)
```

**Key:** `coords2d` is ALWAYS the real coordinates. We test whether the
representation implied by training on permuted features produces a non-planar
graph when drawn on true geography.

### Phase 3: Decision

| Outcome | kappa_reconstruct(A) | kappa_reconstruct(B) | Interpretation |
|---------|---------------------|---------------------|----------------|
| **Gate validated** | > 0.5 | < 0.3 | Gate catches topology destruction. Promote to ENFORCED. |
| Gate sensitive | > 0.5 | 0.3-0.5 | Gate detects but doesn't fire. Consider lowering threshold. |
| Gate insensitive | ~= B | ~= B | Gate measures something other than topology. Investigate. |
| Forward degraded | high | low (but forward also low) | Permutation killed forward accuracy. Not the quadrant we need. Arm C may be more informative. |

**Go/no-go for Gate 3B ENFORCED status (evaluated after Phase 1):**
1. kappa_reconstruct(A) - kappa_reconstruct(B) > 0.2 (effect size)
2. `adversarial_geography_permutation.earns_gate == True` (orthogonality)
3. forward_score(B) >= 0.8 * forward_score(A) (permutation didn't trivially kill prediction)

If all three pass: Gate 3B promoted to ENFORCED. Phase 2 (Arms C+D) runs
as sensitivity analysis.

If condition 3 fails: the permutation damaged the signal too much to test
the taxi-flyover quadrant. Proceed to Phase 2 -- Arm C (block permutation)
preserves more local signal and may reach the right quadrant.

---

## Permutation Implementation

```python
def permute_zcta_centroids(
    centroids: pd.DataFrame,
    mode: str = "global",
    block_km: float = 200.0,
    seed: int = 42,
    id_col: str = "zcta_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Permute centroid-to-ZCTA assignments.

    Args:
        centroids: DataFrame with ZCTA IDs and coordinates.
        mode: "global" (full shuffle) or "block" (permute within spatial blocks).
        block_km: Block size for block permutation.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with same ZCTA IDs but shuffled coordinates.
    """
    rng = np.random.default_rng(seed)
    out = centroids.copy()

    if mode == "global":
        # Full random permutation of coordinates
        perm_idx = rng.permutation(len(out))
        out[lat_col] = out[lat_col].values[perm_idx]
        out[lon_col] = out[lon_col].values[perm_idx]

    elif mode == "block":
        # Assign ZCTAs to spatial blocks, permute within blocks
        from sklearn.cluster import KMeans
        coords = out[[lat_col, lon_col]].values
        # Approximate number of blocks based on block_km
        # CONUS is ~4000km x 2500km, so area ~ 10M km^2
        # block area ~ pi * block_km^2
        n_blocks = max(2, int(10_000_000 / (3.14 * block_km ** 2)))
        n_blocks = min(n_blocks, len(out) // 3)
        km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
        labels = km.fit_predict(coords)
        for block_id in range(n_blocks):
            mask = labels == block_id
            block_idx = np.where(mask)[0]
            perm = rng.permutation(len(block_idx))
            out.loc[out.index[block_idx], lat_col] = out[lat_col].values[block_idx[perm]]
            out.loc[out.index[block_idx], lon_col] = out[lon_col].values[block_idx[perm]]

    return out
```

---

## Compute Budget

### Phased execution: A+B first, C+D only if needed

The critical comparison is A (real) vs B (permuted). Arms C (block-permuted)
and D (covariate-only) are sensitivity checks that only run if A vs B passes
the forward-score ratio threshold (condition 3 in go/no-go).

### Phase 1: A+B (mandatory)

| Phase | Work | Wall-Clock |
|-------|------|------------|
| Feature extraction (B) | 1 arm x 5 scenarios, STAC COG reads | ~5 min |
| Training (A+B) | 2 arms x 5 scenarios x 3 targets x 5 folds = 150 runs @ 3 min | ~45 min (5 parallel instances) |
| Gate 3B evaluation (A+B) | 2 arms x 5 scenarios = 10 evals @ ~5 sec | ~1 min |
| Adversarial permutation | 10 trials x 5 scenarios @ 3 min/trial | ~2.5 hrs (1 instance serial) |
| **Phase 1 total** | | **~3.5 hrs** |

### Phase 2: C+D (conditional on Phase 1 passing)

| Phase | Work | Wall-Clock |
|-------|------|------------|
| Feature extraction (C) | 1 arm x 5 scenarios | ~5 min |
| Training (C+D) | 2 arms x 150 runs | ~45 min |
| Gate 3B evaluation (C+D) | 10 evals | ~1 min |
| **Phase 2 total** | | **~1 hr** |

### Instance and resource assumptions

Based on s035 COMPUTE_MANIFEST.md -- all s035 jobs ran on ml.m5.xlarge or
ml.m5.large in 2-4 min each. HistGBDT on 60-70 features x 200-400 ZCTAs
is tiny. No GPU needed.

| Resource | Value | Rationale |
|----------|-------|-----------|
| **Instance** | ml.m5.xlarge (4 vCPU, 16 GB) | Same as all s035 training jobs. ml.m5.4xlarge is 4x cost for no benefit at ZCTA scale. |
| **Volume** | 10 GB | Features are small parquets. No raster I/O on SageMaker -- STAC COG windows are fetched into memory by floodcaster.batch. |
| **Image** | Standard SageMaker sklearn | Same as s035 R0/R1/R2. pip install floodcaster from git at job start. |
| **Threads** | n_jobs=-1 on HistGBDT | Standard convention. Gabriel graph + crossing detection is single-threaded but finishes in seconds at ZCTA scale (~300 regions, O(n^2) edges). |
| **Memory** | <2 GB working set | Feature matrices are ~400 rows x 70 cols. kappa_reconstruct distance matrices are 400x400 float64 = 1.2 MB. No memory pressure. |
| **Cache** | Arm A features already on S3 | build_event_dataset cached Deltares/hydrology/SAR features during pipeline run. Only Arms B and C require new STAC extraction. |
| **Parallelism** | 5 instances for training (1 per scenario) | Each scenario is independent. Adversarial permutation runs serial (10 trials, each retrains) on 1 instance per scenario. |

---

## Success Criteria

| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| kappa_reconstruct delta (A-B) | > 0.2 | Gate detects global permutation |
| kappa_reconstruct delta (A-C) | > 0.1 | Gate detects subtle topology damage |
| earns_gate (adversarial test) | True | Orthogonal to kappa_spatial |
| forward_score(B) / forward_score(A) | > 0.8 | Permutation tests the right quadrant |
| HAND importance drop (A vs B) | > 50% | Topology features carry the signal |

---

## Artifacts

| Artifact | S3 Key | Format |
|----------|--------|--------|
| Permuted centroids | processed/u0_adversarial/perm_centroids_{mode}_{seed}.parquet | Parquet |
| Permuted features | processed/u0_adversarial/{arm}/zcta_features_{scenario}.parquet | Parquet |
| Gate 3B results | results/u0_adversarial/gate_3b_{arm}_{scenario}.json | JSON |
| Adversarial permutation result | results/u0_adversarial/adversarial_perm_{scenario}.json | JSON |
| Feature importance | results/u0_adversarial/importance_{arm}_{scenario}.json | JSON |
| Summary table | results/u0_adversarial/summary.parquet | Parquet |

---

## Relationship to Existing Infrastructure

| Component | Status | U0 Role |
|-----------|--------|---------|
| `kappa_reconstruct.py` | Implemented, EXPERIMENTAL | Primary instrument |
| `adversarial_geography_permutation()` | Implemented | Secondary test |
| `gate_3b_decision()` | Implemented | Decision function |
| `floodcaster.batch` | Just landed (v24.009) | Feature extraction engine |
| R0 fold assignments | Complete | Reused |
| R1 training pipeline | DESIGNED | Training adapter |
| FEATURE_CONTRACT.yaml | 16 new features added | Feature registry |

**This experiment does not require new instruments.** It uses the existing
Gate 3B infrastructure with a controlled adversarial payload constructed from
the new Planetary Computer features. The only new code is the
`permute_zcta_centroids` function and the training adapter that swaps
feature matrices.
