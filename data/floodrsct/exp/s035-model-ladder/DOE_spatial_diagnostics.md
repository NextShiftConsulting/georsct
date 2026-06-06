# DOE: Spatial Residual & Structure Diagnostics — Non-Locking Sidecar

**Experiment:** s035-model-ladder / Phase 0.6 (GWR probe) + Phase 4d (residual LISA/Geary) + Robustness Appendix R (regionalization)
**Role:** Descriptive spatial diagnostics and figures that sit ALONGSIDE the locked ladder. None is a hypothesis test. None alters folds, solvers, features, or hyperparameters.
**Status:** DESIGNED (DOE phase, not launched)
**Amends:** Nothing. This is additive. It does not modify DOE_LOCKED.md, the v1.2 amendment, or the kappa cascade.
**Depends on:** kappa_geom (Phase 0.5) for GWR triangulation; R0/R1/R2 predictions parquets for residual diagnostics.

---

## Why This Exists (and what it deliberately does NOT do)

The locked ladder establishes whether representation upgrades help (H2a fold-level
Wilcoxon) and whether pre-training geometry predicts where they help (kappa_geom,
H2b). It reports spatial error structure only as a single global scalar
(`diag_residual_spatial = 1 - kappa_spatial`, i.e. one Moran's I per cell).

That scalar answers "do errors cluster?" but not "WHERE, and does the cluster
dissolve after the fix?" — which is the spatial-explicit version of the cascade's
central promise. This sidecar adds four diagnostics that answer the spatial
question and produce the paper's spatial figures, WITHOUT touching anything locked.

The governing constraint, stated once and binding on all four sections:

| Constraint | Why |
|-----------|-----|
| No new primary or secondary hypothesis test | n=9 cells; the exploratory family is already at the Holm-Bonferroni power floor. These are descriptive. |
| Model-derived diagnostics stay in `diag_*` space | LISA, Geary's C, and GWR all touch the target or residuals. Per v1.8, kappa_geom must remain model-free. NONE of these may enter kappa_geom. |
| No change to locked folds, solvers, features, hyperparameters | The ladder result must remain exactly as registered. |
| Separate S3 output namespace | `results/s035/sidecar/` — never overwrites or feeds the money table. |
| Pre-registration where a knob exists | GWR feature subset, kernel, criterion; LISA/Geary permutation seed; regionalization attributes — all frozen here, before launch. |

---

## Section 1 — LISA Residual Clusters (Phase 4d) — PRIMARY DELIVERABLE

### What it computes

For each (scenario, target, level ∈ {R0,R1,R2}, solver = histgbdt) cell:

1. Read out-of-fold predictions from `results/s035/r{L}_{scenario}_predictions.parquet`
   (spatial_blocked only — these are the only leakage-safe per-row predictions).
2. Compute residuals on the modeling scale:
   - Regression (NFIP, log1p): `e_i = y_true_i - y_pred_i`
   - Binary (311, HWM): deviance residual, or `(y_i - p_i)` if probabilities are stored
3. Aggregate to ZCTA: mean out-of-fold residual per `zcta_id` across that scenario's
   events (W is ZCTA-keyed, so the LISA cross-section must be ZCTA-keyed).
4. Build Queen contiguity W from `zcta_adjacency.parquet`, restricted to the
   scenario's ZCTAs, row-standardized.
5. Compute Local Moran's I per ZCTA via `esda.Moran_Local`, conditional
   permutation inference (permutations=999, seed=42).
6. Classify each ZCTA: HH / LL / HL / LH / not-significant, using Benjamini-Hochberg
   FDR across ZCTAs (`esda` `Moran_Local.p_sim` + BH at q=0.05). FDR, not raw p,
   because there are one test per ZCTA.

### Outputs per cell

**ZCTA-level table** (`results/s035/sidecar/lisa_{scenario}_{target}_{level}.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| zcta_id | str | ZCTA identifier |
| residual | float | Mean out-of-fold residual |
| local_moran_I | float | Local Moran's I_i |
| local_moran_p_fdr | float | BH-adjusted pseudo p-value |
| cluster_label | str | HH / LL / HL / LH / ns |

**Cell-level rollup** (one row in `results/s035/sidecar/lisa_rollup.parquet`):

| Field | Definition |
|-------|-----------|
| frac_HH, frac_LL | Share of ZCTAs in positive-autocorrelation clusters (systematic, co-located miss) |
| frac_outlier | Share in HL/LH (model wrong relative to neighbors — Scout candidates) |
| frac_significant | Share with FDR-significant local I |
| global_moran_I | Reported alongside for consistency with `diag_residual_spatial` |

### The flag-clearing metric (descriptive, not a test)

The cascade claim is that clusters present at R0 attenuate after R1's W-matrix
features. Define per cell:

```
lisa_cluster_frac[L] = frac_HH[L] + frac_LL[L]
delta_R0_R1 = lisa_cluster_frac[R1] - lisa_cluster_frac[R0]   # expect < 0
delta_R1_R2 = lisa_cluster_frac[R2] - lisa_cluster_frac[R1]
```

Report the sign of delta per cell. If a directional summary across cells is
wanted, report the fraction of cells where clusters attenuated with a
Clopper-Pearson binomial CI, **explicitly labeled decorative at n=9**. No
cell-level Spearman against uplift enters the Holm-Bonferroni family.

### Figure (Paper Figure 5)

Three-panel choropleth (R0 | R1 | R2) for the primary cell
(houston, obs_nfip_event_claims, histgbdt). Each ZCTA colored by cluster_label
(HH = red, LL = blue, HL/LH = orange/purple outliers, ns = light grey). Panel
subtitles carry global Moran's I and `lisa_cluster_frac`. The intended read:
a red HH block at R0 (model under-predicts in a contiguous flood corridor) that
thins by R1. Appendix grid: the same triptych for every modelable cell.

This is the highest-certainty win in the sidecar. It is a richer rendering of an
already-registered diagnostic, so it carries zero lock risk and produces the
single most legible spatial figure in the paper.

### GeoRSCT audit linkage

Maps to A.1 (autocorrelation) and A.2/B.1 (Moran's I on residuals). LISA is the
local-explicit form of the existing `diag_residual_spatial` global scalar.

---

## Section 2 — GWR / MGWR Non-Stationarity Probe (Phase 0.6) — TRIANGULATION + FIGURE

### What it computes

Runs in Phase 0.6, AFTER kappa_geom (Phase 0.5), BEFORE R0 training. Uploaded to
S3 with a timestamp preceding Phase 1, preserving the "predicted before observed"
property the kappa cascade relies on.

For each (scenario, primary target = obs_nfip_event_claims), fit a Geographically
Weighted Regression of the target on a **pre-registered, frozen** subset of R0
features (below). This is a SEPARATE diagnostic regression — it never uses the
HistGBDT/Ridge solver outputs, so it is solver-independent.

**Pre-registered GWR specification (frozen here, no tuning):**

| Choice | Value | Note |
|--------|-------|------|
| Feature subset | `flood_pct_zone_a`, `twi_acc_twi`, `slope_basin_slope`, `acs_median_hh_income`, `svi_overall`, `population` | 6 features; z-scored so local coefficients are comparable |
| Kernel | Adaptive bisquare | Standard; robust to uneven ZCTA density |
| Bandwidth selection | AICc via `mgwr.sel_bw.Sel_BW` | Deterministic given kernel+criterion — no researcher DOF |
| Package | `mgwr` (`GWR`, `Sel_BW`); `MGWR` optional, primary cell only | |
| Coordinates | ZCTA centroids (EPSG:5070 equal-area for distance) | |
| Seed | 42 (for any MC stationarity test) | |
| Min-N guard | Skip if n_zcta < 50; report "insufficient support" | GWR is unstable at small n (e.g. NYC partial) |

### Diagnostic quantities per cell

| Quantity | Definition | Reading |
|----------|-----------|---------|
| `gwr_delta_aicc` | AICc(global OLS) − AICc(GWR) | > 0 ⇒ relationship varies in space |
| `gwr_local_r2_cv` | CV of local R² across ZCTAs | Higher ⇒ fit quality is spatially heterogeneous |
| `gwr_frac_nonstationary` | Fraction of coefficients failing the GWR parameter-stationarity test | Higher ⇒ more covariates act locally |
| `gwr_nonstationarity` | Cell summary = mean normalized local-coefficient std | Single scalar for ranking |

MGWR (optional, primary cell only) reports per-covariate bandwidths, exposing
WHICH features are global vs local — a richer readout for the figure caption.

### The triangulation (this is the legal framing)

You now hold two independent pre-training signals about cell difficulty:

- **kappa_geom** (Phase 0.5): model-free, from geometry + data support.
- **gwr_nonstationarity** (Phase 0.6): model-bearing but solver-independent,
  from spatial variation in the feature→target relationship.

The diagnostic value is their AGREEMENT, and how both relate to the observed
R0→R1 uplift. Report a rank table and a scatter:

```
scenario | kappa_geom | kappa_rank | gwr_nonstationarity | gwr_rank | obs_uplift_R0_R1
```

Report Spearman rho among the three with a bootstrap 95% CI, **explicitly labeled
as observed triangulation, not a hypothesis test, and NOT a member of the
Holm-Bonferroni family.** The narrative claim is qualitative: cells the geometry
flags as hard and the GWR flags as non-stationary are the cells where spatial
features earn the most uplift — if the ranks agree. Report honestly if they do not;
divergence between a geometry signal and a relationship signal is itself a finding.

### Figure (Paper Figure 6)

Local-R² choropleth for the primary cell (houston): where the global feature→NFIP
relationship holds vs collapses. Paired small-multiple of local coefficient
surfaces for two or three covariates (e.g. flood-zone share, TWI, SVI). Caption
ties low-local-R² regions to the LISA HH clusters from Figure 5 and to R1 uplift.

### Hard guardrails

- GWR is **model-bearing** (fits the target). It therefore stays a diagnostic and
  **must NOT enter kappa_geom** — doing so reintroduces exactly the
  model-dependence v1.8 removed. State this in the paper.
- GWR uses the full cross-section (no fold split). It can describe STRUCTURE; it
  cannot support any generalization claim. Frame strictly as a structural
  description, never as predictive performance.

### GeoRSCT audit linkage

Triangulates A.1/B.1 and kappa_geom. Does not claim a new audit code.

---

## Section 3 — Regionalization Robustness (Appendix R) — ROBUSTNESS ONLY

### What it computes

Reviewer risk: the headline uplift is an artifact of county-blocked fold geometry.
Robustness check: re-run the SAME pipeline under an alternative spatially-coherent
block definition, and verify the qualitative finding survives.

1. Build regions with spatially-constrained clustering on the ZCTA adjacency graph
   (`spopt.region.Skater` or `MaxPHeuristic`), using **structural geography only**:
   flood-zone composition + TWI + a hydrologic proxy. Set K = 5 to match the locked
   fold count (apples-to-apples), or use max-p with a population floor and report
   the resulting K and region sizes.
2. Assign each region as one CV block (the only thing that changes is the fold
   column).
3. Re-run R0 and R1 — the primary transition — with identical features, identical
   HistGBDT + Ridge, identical hyperparameters, identical target.
4. Compare the R0→R1 fold-level Wilcoxon under (a) locked county blocks and
   (b) regionalized blocks.

### Output

`results/s035/sidecar/robustness/region_blocked_uplift.json`, plus an appendix
table:

```
cell | uplift_county_blocked | wilcoxon_p_county | uplift_region_blocked | wilcoxon_p_region | direction_agrees?
```

The locked county-blocked Wilcoxon REMAINS the registered primary inference. This
appendix asks only whether the qualitative direction is sensitive to block geometry.

### Hard guardrails

| Constraint | Why |
|-----------|-----|
| Build regions from structural geography ONLY — never the target, never NFIP history | Target-derived regions reintroduce leakage |
| Separate namespace `…/sidecar/robustness/`; never feeds the money table | Primary result must stay untouched |
| Identical features/solvers/hyperparameters; only the fold column changes | This is a robustness re-run, not a new arm |
| Report region sizes; treat as descriptive | At tens–hundreds of ZCTAs, regions are uneven |

This is the most delicate of the four because DOE_LOCKED says "do NOT change fold
assignments." That rule binds WITHIN the locked benchmark. This is a separate,
clearly-labeled appendix that does not alter, overwrite, or replace the locked
folds or the locked primary number.

### GeoRSCT audit linkage

Directly addresses B.1 (MAUP) as a robustness demonstration.

---

## Section 4 — Geary's C Companion (Phase 4d) — MINOR

### What it computes

A companion to global Moran's I on the same per-ZCTA residual series used in
Section 1. Geary's C uses squared neighbor differences, so it is more sensitive to
local/short-range dissimilarity than Moran's I. Agreement between the two
strengthens the autocorrelation finding; divergence flags scale-dependent
structure (Moran broad clustering vs Geary local dissimilarity).

`esda.Geary` on residuals, same Queen W, permutations=999, seed=42, per
(scenario, target, level).

### Output

One extra column block in the diagnostics rollup:

| Field | Note |
|-------|------|
| geary_C | C ≈ 1 no autocorrelation; C < 1 positive; C > 1 negative |
| geary_p | Pseudo p-value |
| geary_delta_R0_R1 | Companion to the Moran Δ |

Report C raw with the interpretation legend (note the inverted scale vs Moran's I).
Optional small twin-bar of Moran's I and (1 − C) per level; no standalone figure.
No hypothesis. This is a one-line robustness companion, nothing more.

### GeoRSCT audit linkage

A.1 (autocorrelation), companion to `diag_residual_spatial`.

---

## Phasing & Sequencing

| Phase | Section | Timing | Reads | Model-bearing? | Touches lock? |
|-------|---------|--------|-------|----------------|---------------|
| 0.6 | GWR probe | After kappa_geom, before R0 train | Assembled R0 parquet, centroids | Yes (diagnostic regression, solver-independent) | No |
| 4d | LISA + Geary | After each level's predictions exist | r{L} predictions parquets, adjacency | Yes (residuals) | No |
| App. R | Regionalization | After R0/R1 locked results exist | Assembled parquet, adjacency | No (re-runs locked solvers) | No (separate namespace) |

GWR runs pre-training so its non-stationarity prediction is timestamped before the
R0→R1 uplift it triangulates — same temporal-ordering discipline as the kappa cascade.

---

## Outputs

| Artifact | S3 Key |
|----------|--------|
| GWR cell summaries | `results/s035/sidecar/gwr_nonstationarity.json` |
| GWR local-R² surface (primary cell) | `results/s035/sidecar/gwr_local_r2_houston.parquet` |
| LISA per-ZCTA tables | `results/s035/sidecar/lisa_{scenario}_{target}_{level}.parquet` |
| LISA cell rollup | `results/s035/sidecar/lisa_rollup.parquet` |
| Geary's C rollup | `results/s035/sidecar/geary_rollup.parquet` |
| Regionalization robustness | `results/s035/sidecar/robustness/region_blocked_uplift.json` |
| Figure 5 (LISA triptych) | `results/s035/sidecar/fig5_lisa_clusters.svg` |
| Figure 6 (GWR local-R²) | `results/s035/sidecar/fig6_gwr_nonstationarity.svg` |

---

## Dependencies

| Dependency | Used by |
|-----------|---------|
| `esda.Moran_Local`, `esda.Geary` | Sections 1, 4 |
| `mgwr` (`GWR`, `Sel_BW`, optional `MGWR`) | Section 2 |
| `spopt.region` (`Skater` / `MaxPHeuristic`) | Section 3 |
| `libpysal` (Queen W, kernels) | Sections 1–4 |
| `zcta_adjacency.parquet` | Sections 1, 3, 4 |
| R0/R1/R2 predictions parquets (spatial_blocked, per-row y_true/y_pred) | Sections 1, 4 |
| kappa_geom output (Phase 0.5) | Section 2 (triangulation) |

Note: `esda.Moran_Local` and `esda.Geary` are NEW calls for s035 — the locked
pipeline only uses the global `yrsn.core.kappa.spatial.compute` Moran's I. `mgwr`
and `spopt` are new package dependencies; pin versions before launch.

---

## Interpretation Criteria (descriptive — no pass/fail)

| Signal | Supportive pattern | Honest null |
|--------|-------------------|-------------|
| LISA cluster fraction | Attenuates R0→R1 in majority of cells | Clusters persist ⇒ W-matrix didn't decorrelate errors spatially |
| GWR ↔ kappa_geom ↔ uplift ranks | Ranks broadly agree | Divergence ⇒ geometry and relationship signals capture different things (still reportable) |
| Region-blocked vs county-blocked direction | Direction agrees | Disagreement ⇒ result is block-geometry-sensitive; report as a limitation, not a failure |
| Geary's C vs Moran's I | Agree on direction | Divergence ⇒ scale-dependent error structure |

---

## Skip / Guard Rules

- GWR: skip any cell with n_zcta < 50; report "insufficient support" rather than an
  unstable fit.
- LISA/Geary: if a level's predictions parquet is missing, skip that level's row;
  do not impute.
- Regionalization: if Skater/max-p cannot form K contiguous regions of viable size,
  report the achievable K and note the constraint; do not force a degenerate split.

---

## DO NOT Constraints

- Do NOT let LISA, Geary's C, or GWR enter kappa_geom (v1.8 independence rule).
- Do NOT add any of these as a primary or secondary hypothesis test.
- Do NOT place any sidecar Spearman in the Holm-Bonferroni exploratory family.
- Do NOT change locked folds, solvers, features, or hyperparameters in any section.
- Do NOT build regionalization blocks from the target or NFIP history.
- Do NOT tune the GWR feature subset, kernel, or criterion after seeing results.
- Do NOT report GWR as predictive performance — it is a structural description on
  the full cross-section, with no fold split.
- Do NOT overwrite or feed the money table from `results/s035/sidecar/`.
