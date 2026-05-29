# Acceptance Criteria: Six Geometry-Specific Kappa-Proxies

**Gate**: Each kappa-proxy below must independently pass ALL criteria before its row can be marked ACCEPTED. Partial credit is not a thing. A proxy either computes a distinct, tested, experimentally-validated quantity -- or it doesn't exist.

**Why this is hard to unapprove**: Once a kappa-proxy is labeled INSTANTIATED in a published venue, any future paper or patent amendment must be consistent with it. Retracting "we built this" is orders of magnitude harder than never claiming it.

---

## HOW THE PIPELINE ACTUALLY WORKS (context for all six)

`theory_certifier.py:certify_group()` runs a FOR loop over embeddings. At each iteration, per-sample data is live in memory:

```
Available per-sample (in memory, NOT on disk):
  solver_preds_test[emb]     shape (n_test,)    -- solver predictions
  solver_preds_train[emb]    shape (n_train,)   -- solver predictions  
  residuals_train            shape (n_train,)   -- |y - y_hat|
  probs_test                 shape (n_test, 3)  -- MLP P(R), P(S), P(N)
  kappa_per_sample_theory    shape (n_test,)    -- clip(D*/D, 0, 1) per sample
  _kappa_per_proxy           shape (n_test,)    -- R_i * (1 - N_i) per sample

Available cross-embedding (accumulated across loop iterations):
  solver_preds_test           dict {emb: ndarray}  -- all embeddings' predictions
  r2 scores                   dict {emb: float}    -- per-embedding test R2

Available on disk but NOT loaded by the pipeline:
  lat/lon                     zcta_features_labels.parquet (31,789 rows)
  queen's adjacency           zcta_adjacency.parquet (edge-list, in yrsn-experiments)
  county/state hierarchy      zcta_county_crosswalk.parquet

NOT available anywhere:
  per-sample residuals on disk (stripped from JSON export)
  semivariogram parameters
  periodicity structure
```

**Key constraint**: Any geometry-specific kappa must be computed INSIDE the FOR loop (where per-sample data is live) or from cross-embedding aggregates available AFTER the loop. Computing it after-the-fact from `s019d_results.json` is possible only for proxies that use aggregated statistics (like per-family R2), not per-sample data.

---

## HOW THE KAPPA REGISTRY WORKS (wiring for all six)

Each new proxy needs a directory in `yrsn/core/kappa/{name}/` with:

```python
# metadata.py
KAPPA_{NAME}_METADATA = KappaMetadata(
    name="kappa_{name}",
    formula="...",
    status=KappaStatus.EXPERIMENTAL,  # until validated
    input_contract={"param": "type [bounds]"},
    assumptions=["..."],
    ...
)

# compute.py  
def compute_kappa_{name}(**inputs) -> KappaResult:
    value = ...  # the actual math
    return KappaResult(name="kappa_{name}", value=value, ...)
```

Register in `registry.py:_populate_registry()`. The gate pipeline consumes `kappa_compat: float` from `CPGatekeeperInput`. To use a geometry-specific kappa for gating, either:
- **Option A**: Pass it as `kappa_compat` (simple, loses provenance)
- **Option B**: Add `kappa_geometry: Optional[float]` to `CPGatekeeperInput` and teach Gate 3 to prefer it when present (cleaner, requires controlplane changes)

---

## KAPPA-PROXY 1: SMOOTH / metric

**Claimed computation**: cross-family R2 convergence
**Claimed status**: INSTANTIATED

### What this proxy measures (proposed -- not yet defined)

"Cross-family R2 convergence" could mean several things:

| Interpretation | Formula | What It Captures |
|---|---|---|
| A. Variance of R2 across families | kappa = 1 - Var(R2_f) / Var_max | Low variance = all families agree = high compatibility |
| B. Rate of R2 stabilization as families added | Slope of cumulative mean R2 vs family count | Convergence speed to stable performance |
| C. Coefficient of variation | kappa = 1 - CV(R2_f) | Scale-invariant agreement |
| D. Min/max ratio | kappa = min(R2_f) / max(R2_f) | Worst-case vs best-case ratio |

Each is mathematically distinct. The figure says "cross-family R2 convergence" without resolving which. **This ambiguity must be resolved before implementation.**

### Data available NOW (no re-run needed)

`s019d_results.json` contains 810 rows: 27 tasks x 6 embeddings x 5 folds. Per row: `r2`, `embedding`, `target`. This is enough to compute any of interpretations A-D at task level.

Quick feasibility check (can be done in 10 minutes):

```python
import json, numpy as np
results = json.load(open("data/s019d/seed_42/s019d_results.json"))
# Group by task, compute per-task R2 variance across 6 embeddings
for task in tasks:
    r2s = [r["r2"] for r in results if r["target"] == task]
    smooth_kappa_A = 1 - np.var(r2s)  # interpretation A
    universal_kappa = np.mean([r["proxy_kappa"] for r in results if r["target"] == task])
    # Compare: if corr(smooth_kappa_A, universal_kappa) > 0.9, this proxy is redundant
```

### The real question

This proxy operates at TASK level (one kappa per task from 6 R2 values), but the gate pipeline operates at EMBEDDING level (one kappa per task-embedding pair). How does a task-level kappa feed into an embedding-level gate? Options:

1. **Same kappa for all embeddings on a task**: Every embedding gets the same smooth kappa. But then the gate can't distinguish embeddings -- it can only accept/reject the entire task.
2. **Leave-one-out**: Compute R2 convergence excluding the target embedding. This is per-embedding but requires >= 3 other families.
3. **Scale by embedding's R2 rank**: Multiply task-level convergence by the embedding's percentile rank. Contrived.

This isn't a nitpick -- it determines whether the proxy can function in the existing gate architecture.

### Acceptance criteria

| # | Criterion | What Must Be True | Status |
|---|-----------|-------------------|--------|
| S1 | Mathematical definition resolves the ambiguity (A/B/C/D above) | Paper equation with justification for choice | [ ] |
| S2 | Task-level vs embedding-level resolution is addressed | Design doc or paper paragraph explaining how task-level kappa feeds per-embedding gates | [ ] |
| S3 | Redundancy pre-check: Spearman(smooth_kappa, universal_kappa) < 0.9 on CONUS-27 | Correlation table from s019d_results.json | [ ] |
| S4 | `compute_kappa_smooth()` registered in yrsn kappa registry | File path + registry entry | [ ] |
| S5 | Unit tests with edge cases: (a) all families identical R2, (b) one family dominates, (c) 2 families only | Test file + pass log | [ ] |
| S6 | Run on all 27 CONUS-27 tasks with results persisted | Output JSON with smooth_kappa per task | [ ] |
| S7 | At least 1 task where smooth_kappa produces a different gate outcome than universal kappa | Comparison showing EXECUTE vs RE_ENCODE flip | [ ] |

**Pre-implementation gate (do this first, takes 10 min)**:
Compute S3 from existing data. If smooth_kappa correlates > 0.9 with mean proxy_kappa across 27 tasks, STOP. The proxy is redundant and the figure's premise is wrong for this geometry type.

---

## KAPPA-PROXY 2: SPATIAL / dependence

**Claimed computation**: semivariogram range
**Claimed status**: INSTANTIATED

### What this proxy measures

A semivariogram models spatial autocorrelation as a function of distance. The "range" is the distance at which autocorrelation drops to zero (observations become independent). A short range means spatial structure is local; a long range means it's global.

Mapping range to kappa: unclear. Is short range = high kappa (easy, local structure) or low kappa (hard, structure doesn't help at distance)? The answer depends on the spatial footprint of test observations relative to the range.

### Implementation requirements

**Step 1: Compute semivariogram from OOF residuals**

This requires:
- Per-sample residuals (in memory during `certify_group()`, not on disk)
- Per-sample spatial coordinates (NOT loaded by current pipeline)
- A semivariogram fitting function (NOT in any dependency)

Semivariogram fitting is non-trivial:

```python
# Pseudocode for empirical semivariogram
def empirical_semivariogram(residuals, coords, n_bins=15):
    """Compute gamma(h) = 0.5 * E[(Z(s) - Z(s+h))^2] for lag bins."""
    dists = pairwise_distances(coords)  # O(n^2) -- 31,789^2 = 1 billion pairs
    diffs = (residuals[:, None] - residuals[None, :]) ** 2
    # Bin by distance, compute mean semivariance per bin
    ...

# Model fitting (spherical, exponential, or Matern)
def fit_spherical(h, gamma_empirical):
    """gamma(h) = c0 + c1 * [1.5*(h/a) - 0.5*(h/a)^3] for h <= a"""
    # Nonlinear least squares: estimate nugget (c0), sill (c0+c1), range (a)
    ...
```

**Computational cost**: 31,789 ZCTAs = ~500 million distance pairs per fold. Either subsample or use spatial indexing (KD-tree with distance bins). This is not a 20-line function.

**Library options**:
- `scikit-gstat`: Pure Python, handles fitting. Not in current deps.
- `gstools`: Cython-accelerated, full geostatistics. Not in current deps.
- Hand-rolled: Possible but error-prone (fitting convergence, nugget handling, anisotropy).

### Unresolved mathematical choices

| Choice | Options | Impact |
|---|---|---|
| Variogram model | Spherical / Exponential / Matern / Power | Power model has no finite range. Which to use? |
| Anisotropy | Isotropic (one range) / Anisotropic (directional ranges) | CONUS spans 25 deg lat, 57 deg lon. Isotropy assumption is wrong. |
| Nugget | Estimate freely / Fix at zero | Nugget absorbs measurement error. Fixing at zero biases range upward. |
| Residual source | OOF residuals from one embedding / Cross-family pooled | Per-embedding gives 6 ranges per task. Pooled gives 1. |
| Distance metric | Euclidean on lat/lon / Haversine / Projected | Lat/lon Euclidean is wrong at CONUS scale (1 deg lon != 1 deg lat). |
| Range -> kappa normalization | range / max_range / sigmoid(range) / percentile_rank | Each produces different kappa distributions. |

Six degrees of freedom, each changing the proxy's behavior. The figure says "semivariogram range" as if there's one answer.

### Acceptance criteria

| # | Criterion | What Must Be True | Status |
|---|-----------|-------------------|--------|
| SP1 | Mathematical definition resolves all 6 choices above | Paper appendix with justified choices | [ ] |
| SP2 | Spatial coordinates loaded into `certify_group()` pipeline | Modified function signature + loading code | [ ] |
| SP3 | Semivariogram fitting runs in < 60s per task-embedding (not O(n^2) brute force) | Benchmarked with subsample or KD-tree | [ ] |
| SP4 | Range-to-kappa normalization is monotonic and interpretable | Mathematical proof or empirical demonstration | [ ] |
| SP5 | `compute_kappa_spatial()` registered in yrsn kappa registry | File path + registry entry | [ ] |
| SP6 | Handles degenerate cases: (a) pure nugget (no spatial structure), (b) range > study area, (c) insufficient pairs in lag bin | Unit tests for each | [ ] |
| SP7 | Haversine or projected distances used (not Euclidean lat/lon) | Code review | [ ] |
| SP8 | Run on all 27 CONUS-27 tasks with results persisted | Output JSON with spatial_kappa + fitted range per task-embedding | [ ] |
| SP9 | At least 1 task where spatial_kappa produces a different gate outcome than universal kappa | Comparison showing gate decision flip | [ ] |
| SP10 | Redundancy check: Spearman(spatial_kappa, universal_kappa) < 0.9 | Correlation from S019D data | [ ] |

**This is the hardest of the six proxies.** It requires a new dependency, O(n^2) distance computation with optimization, nonlinear model fitting, and 6 unresolved mathematical choices. Estimated implementation: 2-3 days for a correct, tested version.

---

## KAPPA-PROXY 3: COMPOSITE / multi-structure

**Claimed computation**: cross-family residual agreement
**Claimed status**: INSTANTIATED

### What this proxy measures (proposed -- not yet defined)

"Cross-family residual agreement" asks: do different embedding families make the same errors? High agreement could mean:

- **Interpretation A (optimistic)**: The task's difficulty structure is consistent -- all representations capture the same signal, so the certificate is trustworthy.
- **Interpretation B (pessimistic)**: All models fail identically -- a shared blind spot (e.g., unmeasured confounder) that no representation can fix.

The sign of the kappa mapping REVERSES between these interpretations. This is not a minor detail -- it determines whether high agreement means "gate should pass" or "gate should block."

### Data availability

Per-sample residual vectors are computed inside `certify_group()` but NOT persisted:

```python
# Inside the FOR loop (line 93-189):
residuals_train = np.abs(y_train - solver_preds_train[emb_name])
# This exists in memory but is discarded after tercile bucketing
```

To compute cross-family residual agreement, you need residuals from MULTIPLE embeddings simultaneously. The current loop processes one embedding at a time, discarding residuals before the next embedding. **The loop structure must change** to accumulate residuals across embeddings before computing agreement.

### Implementation sketch

```python
# After the FOR loop over embeddings:
residual_matrix = np.column_stack([
    y_test - solver_preds_test[emb] for emb in emb_names
])  # shape (n_test, n_embeddings)

# Option 1: Pairwise Pearson correlation of residual columns
corr_matrix = np.corrcoef(residual_matrix.T)  # (n_emb, n_emb)
mean_pairwise_corr = corr_matrix[np.triu_indices(n_emb, k=1)].mean()
composite_kappa = mean_pairwise_corr  # high corr = high agreement

# Option 2: Kendall's W (concordance)
# Requires ranked residuals, more robust to outliers

# Option 3: Cosine similarity of residual vectors
cosines = [cosine_sim(residual_matrix[:, i], residual_matrix[:, j])
           for i, j in combinations(range(n_emb), 2)]
composite_kappa = np.mean(cosines)
```

### The fundamental problem

This proxy is CROSS-EMBEDDING. It compares residuals BETWEEN families. But the gate operates PER-EMBEDDING. What kappa does embedding X get from the composite proxy?

- **Option A**: All embeddings get the same composite_kappa (task-level metric). But then you can't use it to choose between embeddings.
- **Option B**: Leave-one-out: compute agreement among all OTHER embeddings. Higher agreement among alternatives = higher confidence that THIS embedding's failure is an outlier. But this inverts the semantics (high agreement among others could mean this embedding is uniquely bad OR uniquely good).
- **Option C**: Compute correlation of THIS embedding's residuals with the consensus (mean of others). High correlation = this embedding agrees with consensus = high compatibility. This is per-embedding and interpretable.

Option C is the most defensible but requires the consensus residual to be meaningful.

### Acceptance criteria

| # | Criterion | What Must Be True | Status |
|---|-----------|-------------------|--------|
| C1 | Interpretation resolved: does high agreement = high kappa or low kappa? | Paper paragraph with justification | [ ] |
| C2 | Agreement metric chosen with justification over alternatives (Pearson vs cosine vs Kendall's W) | Mathematical argument or empirical comparison | [ ] |
| C3 | Per-embedding resolution: how a cross-embedding metric produces per-embedding kappa | Design doc showing option A/B/C and choice | [ ] |
| C4 | Pipeline refactored to accumulate residuals across embeddings (current loop discards per-iteration) | Code change to theory_certifier.py | [ ] |
| C5 | `compute_kappa_composite_geom()` registered in yrsn kappa registry | File path + registry entry | [ ] |
| C6 | Unit tests: (a) identical residuals across families, (b) orthogonal residuals, (c) 2 families only, (d) one family with opposite-sign residuals | Test file + pass log | [ ] |
| C7 | Run on all 27 CONUS-27 tasks with results persisted | Output JSON | [ ] |
| C8 | Redundancy check: Spearman(composite_kappa, theory_sigma) < 0.9 -- because theory_sigma ALREADY measures cross-family kappa dispersion | Correlation table | [ ] |

**Critical redundancy warning**: `theory_sigma = std(kappa_per_sample)` from the existing pipeline already captures cross-family dispersion. If composite_kappa (residual agreement) correlates strongly with theory_sigma, the new proxy is measuring what sigma already measures. Check this BEFORE building.

---

## KAPPA-PROXY 4: PERIODIC / cyclic

**Claimed computation**: 2-pi seam continuity
**Claimed status**: SUPPORTED

### Why this proxy doesn't apply to CONUS-27

CONUS-27 is a cross-sectional dataset of 31,789 ZCTAs with 27 targets (health, socioeconomic, environmental). There is:
- No temporal dimension (no seasonality, no cycles)
- No angular/periodic features (no wind direction, no aspect)
- No antimeridian crossing (CONUS is ~66W to ~125W)
- No circular topology in the target space

The yrsn codebase has extensive toroidal geometry (`geometric_utils.py`, 1114 lines) with `normalize_angle()`, `wrapped_angle_distance()`, `t4_chordal_distance()`. But this is for the T4 certificate embedding space, not for data-domain periodicity.

### What "2-pi seam continuity" would mean

For a function f defined on a periodic domain [0, 2pi), seam continuity measures the jump discontinuity at the wraparound:
```
seam_discontinuity = |f(2pi - epsilon) - f(0 + epsilon)|
kappa_periodic = 1 - seam_discontinuity / max_discontinuity
```
High kappa = smooth wraparound. Low kappa = the model treats 359 degrees and 1 degree as far apart (angular blindness).

### Acceptance criteria (SUPPORTED tier -- definition only, no experimental results required)

| # | Criterion | What Must Be True | Status |
|---|-----------|-------------------|--------|
| P1 | Mathematical definition in paper appendix | Equation for seam continuity measure | [ ] |
| P2 | Input contract specified: what constitutes "periodic features" and where the wraparound point is | KappaMetadata.input_contract | [ ] |
| P3 | At least one example dataset identified where this proxy would apply (NOT CONUS-27) | Named dataset in paper or appendix | [ ] |
| P4 | The figure or caption notes that this proxy targets periodic data regimes, not the v1 benchmark | Caption language | [ ] |

**Honest assessment**: This proxy is architectural furniture for CONUS-27. It exists to make the taxonomy symmetric (2 per tier: Smooth+Spatial instantiated, Composite instantiated, Periodic+Hierarchical+Logical supported). If the paper is honest about this, it's fine. If it implies CONUS-27 exercises this proxy, it's misleading.

---

## KAPPA-PROXY 5: HIERARCHICAL / tree-like

**Claimed computation**: within-vs-across-level R2 gap
**Claimed status**: SUPPORTED

### What this proxy measures

CONUS-27 has a natural hierarchy: ZCTA -> county -> state. The pipeline uses `GroupKFold` on `county_fips` for spatial holdout. This proxy would measure whether performance is concentrated within hierarchy levels or generalizes across them.

```python
# Pseudocode
r2_within_county = mean([r2(y[county_i], y_hat[county_i]) for county_i in counties])
r2_across_county = r2(y_test, y_hat_test)  # already computed
gap = r2_within_county - r2_across_county
# If gap >> 0: model works within counties but fails across (spatial overfitting)
# If gap << 0: model generalizes well across counties
kappa_hierarchical = sigmoid(-gap)  # or some normalization
```

### Data availability

Everything needed is already in the pipeline:
- `y_test`, `solver_preds_test[emb]` -- already computed
- `county_fips` per ZCTA -- in `zcta_county_crosswalk.parquet`
- `state` per ZCTA -- in same crosswalk

**This is the most buildable of all six proxies.** The data exists, the math is straightforward, and the hierarchy is natural to the benchmark.

### Why it's labeled SUPPORTED and not INSTANTIATED

Unknown. No documentation explains the choice. Possible reasons:
1. It was considered but the hierarchy (3 levels) is too shallow for meaningful gap analysis
2. It was never considered -- the SUPPORTED label is aspirational
3. The gap is confounded with the GroupKFold holdout strategy (which already splits by county)

Reason 3 is concerning: if test data is held out by county, then `r2_within_county` on TEST data means "R2 within held-out counties" -- but each test county is only seen once (as a held-out group). The within-county R2 becomes a small-sample estimate with high variance.

### Acceptance criteria

| # | Criterion | What Must Be True | Status |
|---|-----------|-------------------|--------|
| H1 | Mathematical definition resolves: gap = difference, ratio, or effect size? | Paper equation | [ ] |
| H2 | Confound with GroupKFold addressed: how to measure within-county R2 when counties are the holdout unit | Statistical argument or alternative split | [ ] |
| H3 | Handles degenerate cases: (a) county with 1 ZCTA, (b) state with 1 county, (c) all counties same R2 | Unit tests | [ ] |
| H4 | Gap-to-kappa normalization specified and monotonic | Formula + justification | [ ] |
| H5 | If upgraded to INSTANTIATED: registered in kappa registry + run on CONUS-27 | Code + results | [ ] |

**Pre-implementation gate**: Compute the raw gap from existing `s019d_results.json` + crosswalk. If the gap has near-zero variance across tasks (all tasks show the same within-vs-across pattern), the proxy has no discriminative power. This takes 20 minutes.

---

## KAPPA-PROXY 6: LOGICAL / relational

**Claimed computation**: Fisher discriminant on adjacency
**Claimed status**: SUPPORTED

### What this proxy measures

Fisher's linear discriminant ratio for two groups (adjacent vs non-adjacent pairs):

```
J(w) = (mu_adj - mu_non_adj)^2 / (var_adj + var_non_adj)
```

Where `mu_adj` is the mean feature difference between adjacent ZCTAs, and `mu_non_adj` is the mean difference between non-adjacent ZCTAs. High J = adjacent ZCTAs are more similar than non-adjacent = spatial structure exists = the representation captures neighborhood effects.

### Relationship to Moran's I

Moran's I measures spatial autocorrelation:
```
I = (n / sum(w_ij)) * (sum(w_ij * (x_i - x_bar)(x_j - x_bar)) / sum((x_i - x_bar)^2))
```

Fisher discriminant on binary adjacency and Moran's I with binary weights are algebraically related -- both measure whether adjacent observations are more similar than expected. The key difference:
- Moran's I is a correlation (-1 to 1)
- Fisher discriminant is a variance ratio (0 to inf)
- Moran's I has well-understood statistical properties (permutation tests, normal approximation)
- Fisher discriminant on adjacency does not have established spatial statistics theory

**If you're going to measure spatial autocorrelation, why not use Moran's I?** It's better understood, has standard significance tests, and multiple implementations exist (PySAL). Using Fisher discriminant instead requires justifying why a less-established method is preferred over the standard.

### Data availability

- Queen's adjacency: `yrsn-experiments/.../zcta_adjacency.parquet` (edge-list format)
- Features/residuals: in memory during `certify_group()`, not on disk

### Acceptance criteria

| # | Criterion | What Must Be True | Status |
|---|-----------|-------------------|--------|
| L1 | Mathematical definition in paper: Fisher discriminant on adjacency, applied to what? (raw features? residuals? embeddings?) | Paper equation specifying input | [ ] |
| L2 | Justification for Fisher over Moran's I | Paper paragraph with mathematical or empirical argument | [ ] |
| L3 | J-to-kappa normalization specified (J ranges [0, inf), kappa must be [0, 1]) | Normalization formula | [ ] |
| L4 | Handles degenerate cases: (a) isolated ZCTA (no neighbors), (b) complete graph, (c) bipartite structure | Unit tests | [ ] |
| L5 | If upgraded to INSTANTIATED: registered + run on CONUS-27 | Code + results | [ ] |

---

## CROSS-PROXY SYSTEM CRITERIA

These apply to the set of proxies as a system. They can only be evaluated after individual proxies exist.

| # | Criterion | What Must Be True | Why It Matters |
|---|-----------|-------------------|----------------|
| X1 | **Non-redundancy**: Pairwise Spearman(kappa_i, kappa_j) < 0.9 for all pairs on CONUS-27 | If two proxies are redundant, one should be dropped. Six redundant proxies is worse than one honest one. |
| X2 | **Non-redundancy with existing metrics**: Spearman(geometry_kappa, theory_sigma) < 0.9 AND Spearman(geometry_kappa, proxy_kappa) < 0.9 | theory_sigma already captures cross-family instability. N_ceiling already captures task difficulty. New proxies must add information beyond what's already measured. |
| X3 | **Taxonomy justification**: Why these six? Why not five (drop Periodic since CONUS-27 has no periodic targets)? Why not seven (add a Temporal geometry for panel data)? | An arbitrary taxonomy is worse than no taxonomy. The count must follow from principles, not aesthetics. |
| X4 | **Dispatch rule**: How does the pipeline know which geometry type a task belongs to? Is it user-specified? Auto-detected? One task = one geometry, or can a task have mixed geometry? | Without dispatch, the figure's branching is hypothetical. |
| X5 | **Backward compatibility**: Existing S019D results remain reproducible with universal kappa. Geometry-specific kappa is additive (new column in results), not a replacement. | Can't invalidate published results. |
| X6 | **Relationship to 12-mode taxonomy**: The paper has GEO-A through GEO-D (12 modes). The figure has 6 geometries. How do they relate? Do the 12 modes partition into the 6 geometries? Or are they orthogonal? | Two unrelated taxonomies in one paper is confusing. |

---

## WHAT TO DO BEFORE WRITING ANY CODE

### 10-minute pre-checks from existing data

These can be computed from `s019d_results.json` without any pipeline changes:

1. **Smooth redundancy**: Compute Var(R2) across 6 embeddings per task. Correlate with mean proxy_kappa. If rho > 0.9, Smooth is redundant.

2. **Hierarchical feasibility**: Load county crosswalk. Compute within-county vs across-county R2 gap per task-embedding. Check variance across tasks. If near-zero variance, proxy has no discriminative power.

3. **Composite vs sigma redundancy**: theory_sigma is std(per-sample kappa across embeddings). If cross-family residual agreement correlates with theory_sigma, Composite is redundant.

4. **Logical feasibility**: Load adjacency. Compute Moran's I on per-task residuals (using s019b_kappa_residual.npz if it has spatial structure). Check if I varies meaningfully across tasks.

**If any proxy fails its redundancy check, it should be dropped from the figure, not built.**

### 30-minute design decisions (resolve before implementation)

1. **Task-level vs embedding-level**: Smooth and Composite are naturally cross-embedding (task-level). The gate is per-embedding. How does a task-level kappa feed a per-embedding gate?

2. **Replace vs augment**: Does geometry kappa REPLACE `kappa_compat` in Gate 3? Or is it an additional field that informs but doesn't gate?

3. **Dispatch**: Who decides the geometry type? The user? An auto-detector? The data itself?

4. **The sign problem**: For Composite, does high residual agreement mean high kappa (consistent signal) or low kappa (shared blind spot)?

---

## HONEST DIFFICULTY RANKING

| Proxy | Math Defined? | Data Exists? | Redundancy Risk | Computational Cost | Implementation Days | Verdict |
|---|---|---|---|---|---|---|
| Smooth | NO (4 interpretations) | YES (R2 in JSON) | HIGH (may correlate with existing kappa) | Trivial | 0.5 | BUILD ONLY IF non-redundant |
| Hierarchical | NO (gap undefined) | YES (crosswalk on disk) | MEDIUM (confounded with GroupKFold) | Low | 0.5 | BUILD ONLY IF gap varies across tasks |
| Composite | NO (agreement undefined, sign problem) | PARTIAL (residuals in memory only) | HIGH (theory_sigma already measures dispersion) | Low but requires pipeline refactor | 1 | BUILD ONLY IF orthogonal to sigma |
| Logical | NO (Fisher vs Moran unjustified) | YES (adjacency on disk) | MEDIUM (related to Moran's I) | Medium | 1 | JUSTIFY Fisher over Moran first |
| Spatial | NO (6 free parameters) | PARTIAL (coords yes, residuals no) | LOW (unique measurement) | HIGH (semivariogram fitting) | 2-3 | Most novel but hardest to build |
| Periodic | NO | NO DATA | N/A | N/A | N/A | DROP for CONUS-27 |

**Bottom line**: The honest path is:
1. Run the 10-minute redundancy pre-checks
2. Kill any proxy that fails
3. Resolve the math for survivors
4. Build and test
5. Update the figure to reflect reality

The dishonest path is to build all six, label them INSTANTIATED, and hope reviewers don't ask about redundancy. That path ends at Reviewer 2.
