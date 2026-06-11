# Pre-Registration: Floodcaster Story DOE

**Experiment:** s035-model-ladder / Floodcaster Diagnostic Stories
**Lock date:** 2026-06-11
**Status:** LOCKED (DOE phase)
**Depends on:** All s035 phases COMPLETED; Planetary Computer features wired (beb18dc)
**Decision rule:** Results determine paper placement (281 section vs companion vs appendix)

---

## Governing Principle

> No memorable assertion appears in the paper unless it maps to a DOE row,
> a metric, a comparison, and a permitted conclusion.

> No experiment runs until this file defines: DOE ID, assertion, information
> regime, feature sets, frozen row universe, split protocol, primary metric,
> pass condition, demotion language, figure/table output, and code/artifact path.

---

## Narrative Spine

**Construct -> Physics -> Hazard Shape -> Observation**

| Act | Story | Title | Evidence class | Ship status |
|-----|-------|-------|----------------|-------------|
| I | Construct | Insurance Is Not Damage | Descriptive divergence | Ship as main result |
| II | Physics necessity | Is Topology Load-Bearing? | Ablation / necessity test | Ship as robustness |
| IIb | Reconstruction | The Model's Impossible Watershed | Implied-field recovery | Register, do not claim |
| III | Hazard shape | The Floodplain Is Not a Scalar | Multi-RP representation test | Ship as main result |
| IV | Observation | Wet Today, or Always Wet? | Event-observation value | Ship in nowcast arm only |

---

## Literature Prior Table

Pre-registered expected compatibility ordering for physics-constrained solvers.
Derived from prior hydrologic evidence, NOT from FloodRSCT outcomes.
Maps to `PREREG_PHYSICS_RANK` in `physics_compat_probe.py`.

| Geometry | Rank | Literature evidence (our prior) | Source |
|----------|-----:|--------------------------------|--------|
| Smooth/metric | 6 | Process-based continuous streamflow; rainfall-runoff on smooth physical fields | feng2022, song2025 |
| Spatial/dependence | 5 | River-network routing (Muskingum-Cunge); mass-conserving GNN message passing | bindas2024, taghizadeh2025 |
| Composite | 4 | Multiphysical outputs; ungauged transfer — physics partially carries | feng2022 |
| Periodic | 3 | Storm/seasonal climatology plausible but under-represented in cited corpus | *flagged as uncovered* |
| Hierarchical | 2 | Watershed regionalization exists as parameter sharing, not relational hierarchy | feng2022, song2025 |
| Logical/relational | 1 | Monotonicity is the only relational-ish constraint; weak — not adjacency/policy | zhang2025 |

**Interpretation:** The field's own papers predict an ordering they never measured.
We measure it. If kappa_compat does NOT rank physics high-on-smooth / low-on-logical,
that is a real result against our own prediction.

**Hard dependency:** kappa=0.0 pin must be fixed before this ordering can be tested.
`physics_compat_probe.py` raises `DegenerateKappaError` under the current pin.

---

## DOE-C1: Construct Divergence — FAST vs NFIP

**Assertion:** FAST and NFIP encode different spatial constructs.

**Information regime:** N/A (descriptive, not forecast/nowcast)

**Data required:**
- `results/s035/fast_validation.json` (COMPLETED)
- `processed/{scenario}/*_event_features.parquet` (COMPLETED)
- FAST ZCTA aggregates: `processed/{scenario}/{scenario}_fast_zcta_*.parquet` (COMPLETED)
- Policies-in-force: optional denominator (not currently on S3)

**Feature sets:**
- `fast_total_loss_usd`, `fast_mean_loss_per_sqft`, `fast_pct_damaged` (FAST signal)
- `nfip_event_claim_count`, `nfip_event_total_loss` (NFIP signal)

**Frozen rows:** Same ZCTA-event universe as s035 R0.

**Split protocol:** N/A (descriptive comparison, not model evaluation)

**Primary metrics:**
- Spearman/Kendall correlation (FAST vs NFIP per event)
- Quadrant mass distribution (2x2: high/low FAST x high/low NFIP)
- Bivariate LISA (if spatial clustering pattern exists)

**Pass condition:** Correlation is significantly different from +1.0 (i.e., the two
constructs do not agree spatially). The Houston anti-correlation (rho = -0.189 to
-0.329) already suggests this.

**Demotion language:** If correlation is positive and strong (rho > 0.5), downgrade to
"FAST and NFIP show spatial agreement for this scenario; construct divergence not
demonstrated."

**Allowed conclusion:**
> FAST and NFIP diverge spatially and should not be treated as interchangeable
> flood-damage labels.

**Forbidden conclusion unless PIF denominator available:**
> High FAST / low NFIP means under-insurance.

**Quadrant labels (descriptive, not causal):**

| Quadrant | Label |
|----------|-------|
| High FAST / High NFIP | Construct agreement: high physical + high administrative signal |
| High FAST / Low NFIP | Physical-dominant divergence |
| Low FAST / High NFIP | Claims-dominant divergence |
| Low FAST / Low NFIP | Construct agreement: low signal |

**Figure output:** Bivariate construct divergence map (Houston, 3 events).

**Code path:** `jobs/compute_construct_divergence.py` (to be created)

**S3 artifact:** `results/s035/story/construct_divergence_{scenario}.json`

**Status: DATA EXISTS. Run next.**

---

## DOE-P1: Topology Necessity — HAND/GFI Load-Bearing Test

**Assertion:** HAND/GFI are load-bearing topology-derived features for flood prediction.

**Information regime:** Forecast (all features are pre-event)

**Data required:**
- R0 baseline results (COMPLETED)
- HAND, GFI, TWI, SPI features from `build_hydrology_features()` (wired, not yet fetched for all scenarios)
- R0 fold assignments (COMPLETED)

**Feature sets:**

| Arm | Features | Purpose |
|-----|----------|---------|
| baseline | R0 + HAND + GFI + TWI + SPI | Full topology features |
| no_topology | R0 only (remove HAND/GFI/TWI/SPI) | Ablation |
| shuffled_topology | R0 + shuffled(HAND/GFI/TWI/SPI) | Feature-association test |

**Frozen rows:** Same ZCTA-event rows as R0 baseline. Same folds. Same seed (42).

**Split protocol:** Spatial-blocked CV (same as R0). No feature-dependent row loss.

**Primary metrics:**
- Delta RMSE (baseline vs no_topology)
- Delta RMSE (baseline vs shuffled_topology)
- Paired bootstrap CI on deltas
- Fold stability (all 5 folds must agree on direction)

**Pass condition:** Performance degrades when topology features removed AND when shuffled.
Both deltas negative (worse without topology), CI excludes zero.

**Demotion language:** "HAND/GFI were not load-bearing under this representation
and solver. Topology features are redundant with existing R0 covariates."

**Allowed conclusion:**
> Drainage-derived features are load-bearing for flood prediction in this substrate.

**Forbidden conclusion:**
> The model predicts in a city where water runs uphill.

(That requires DOE-R1.)

**Figure output:** Ablation table (3 arms x 5 scenarios x primary metric).

**Code path:** `jobs/run_topology_necessity.py` (to be created)

**S3 artifact:** `results/s035/story/topology_necessity_{scenario}.json`

**Status: NOT RUN. Requires hydrology features fetched for all scenarios.**

---

## DOE-H1: Hazard Shape — Multi-Return-Period Deltares Depths

**Assertion:** Multi-return-period depth vectors capture hazard structure beyond a
single 100-year depth scalar.

**Information regime:** Forecast (Deltares depths are pre-event modeled hazard)

**Data required:**
- Deltares depths at RP 10, 50, 100: `processed/shared/zcta_deltares_depth.parquet` (wired, cache status unknown)
- R0 baseline results (COMPLETED)
- R0 fold assignments (COMPLETED)

**Feature sets:**

| Arm | Features | Purpose |
|-----|----------|---------|
| scalar_100yr | R0 + deltares_depth_ft_rp100 only | Single tail-risk value |
| level_gradient | R0 + deltares_depth_level + deltares_depth_gradient | Level + shape |
| full_vector | R0 + deltares_depth_ft_rp{10,50,100} + max + inundation_pct | Full RP vector |

**Derived features (computed from Deltares):**
- `deltares_depth_level` = mean(rp10, rp50, rp100)
- `deltares_depth_gradient` = (rp100 - rp10) / rp100 (normalized slope)
- `deltares_depth_curvature` = rp50 - (rp10 + rp100)/2 (deviation from linear)

**Frozen rows:** Same ZCTA-event rows as R0. Same folds. Same seed.

**Split protocol:** Spatial-blocked CV (same as R0).

**Primary metrics:**
- Delta RMSE: scalar_100yr vs full_vector
- Delta RMSE: scalar_100yr vs level_gradient
- Ranking stability (Kendall tau-b across arms)
- Fold stability

**Pass condition:** Vector or level+gradient outperforms scalar. CI excludes zero.

**Demotion language:** "No evidence that return-period shape improved this task
beyond a single depth scalar."

**Allowed conclusion:**
> Multi-return-period modeled depths preserve hazard structure beyond a single
> tail-risk scalar.

**Important wording:** "modeled hazard structure" not "observed risk."

**Figure output:** Return-period curves for representative ZCTAs + spatial gradient map.

**Code path:** `jobs/run_hazard_shape.py` (to be created)

**S3 artifact:** `results/s035/story/hazard_shape_{scenario}.json`

**Status: NOT RUN. Requires Deltares features fetched.**

---

## DOE-O1: Observation Value — SAR Anomaly in Nowcast Arm

**Assertion:** SAR anomaly adds event-time observation value beyond static exposure.

**Information regime:** NOWCAST / DAMAGE ASSESSMENT ONLY.

SAR anomaly is event-time evidence. It is forbidden in the forecast arm unless
the SAR acquisition timestamp is strictly before the prediction decision time
AND inside a declared observation window.

**Data required:**
- SAR features from `build_sentinel1_event_features()` (wired, per-event)
- JRC baseline from `build_jrc_water_features()` (wired, shared)
- R0/R2 results (COMPLETED)
- Event peak_window metadata

**Feature sets:**

| Arm | Features | Information regime |
|-----|----------|--------------------|
| pre_event | R0 + R1 + JRC baseline | Forecast |
| nowcast_sar | R0 + R1 + JRC baseline + sar_water_pct + sar_water_pct_anomaly | Nowcast |

**Timestamp discipline:**
- Each SAR acquisition must have timestamp metadata
- Acquisition must fall within declared event `peak_window`
- `sar_acquisition_lag_days` must be reported
- If lag is negative (SAR before event peak), flag as pre-event eligible

**Frozen rows:** Same ZCTA-event rows. Same folds. Same seed.

**Split protocol:** Spatial-blocked CV (same as R0).

**Primary metrics:**
- Delta RMSE: pre_event vs nowcast_sar
- Scenario-level lift breakdown
- Timestamp eligibility audit (what % of SAR features are truly event-time)

**Pass condition:** Nowcast arm outperforms pre-event arm. CI excludes zero.
Timestamp audit confirms >80% of SAR features are event-time.

**Demotion language:** "SAR anomaly did not improve nowcast performance beyond
static exposure, OR timestamp audit failed (leakage risk)."

**Allowed conclusion:**
> SAR anomaly separates permanent water from event-specific inundation and
> improves nowcast/damage assessment.

**Forbidden conclusion:**
> SAR anomaly proves forecast representation lift.

**Figure output:** Raw SAR vs anomaly maps (2-panel) + lift by scenario.

**Code path:** `jobs/run_observation_value.py` (to be created)

**S3 artifact:** `results/s035/story/observation_value_{scenario}.json`

**Status: NOT RUN. Requires SAR features fetched per-event.**

---

## DOE-R1: Reconstruction — kappa_reconstruct (Registered Extension)

**Assertion:** The model's implied drainage field can be recovered and tested for
physical coherence.

**Status: REGISTERED. DO NOT CLAIM UNTIL IMPLEMENTED.**

**This is the only DOE row that earns the taxi-route analogy.**

See `DOE_U0_adversarial_geography.md` for the full design (4 arms: real,
permuted, block-permuted, covariate-only). The U0 DOE already covers the
constructive adversarial test. DOE-R1 adds the implied-field recovery step.

**Minimum viable design:**
1. Synthetic planted DEM with known D8 flow
2. Known HAND/GFI field
3. Planted target from topology-consistent hydrology
4. Corrupted negative control (planted uphill violations)
5. Train same model ladder on both
6. Recover model-implied HAND-like field from response surface
7. Test downstream monotonicity violations
8. Report kappa_reconstruct = 1 - violation_rate

**Existing infrastructure:**
- `georsct/domain/kappa_reconstruct.py` (untracked, exists locally)
- `adversarial_geography_permutation()` (implemented)
- `gate_3b_decision()` (implemented)
- DOE_U0 arms A/B/C/D (designed, not launched)

**Forbidden assertion without this DOE row completed:**
> The model predicts flood loss in a city where water runs uphill.
> The model's implied watershed is impossible.

---

## Paper Placement Decision Rule

Do not choose 281 vs companion based on narrative elegance.
Choose based on result outcomes.

| Outcome | Placement |
|---------|-----------|
| DOE-C1 only confirms | Add construct divergence subsection to 281 |
| DOE-C1 + DOE-H1 confirm | Strong 281 results expansion |
| DOE-C1 + DOE-H1 + DOE-P1 confirm | 281 gets diagnostic section |
| DOE-C1 + DOE-H1 + DOE-P1 + DOE-O1 confirm | 281 gets forecast/nowcast split, or companion starts to justify |
| DOE-R1 implemented and works | Companion paper becomes real |
| Without DOE-R1 | Taxi-route analogy stays in future work |

**The clean internal phrase:**
> Run the evidence funnel before choosing the paper container.

---

## Hard Guards

1. `frozen_rows_required: true` — Same ZCTA-event universe across all arms
2. `frozen_folds_required: true` — Same spatial-blocked fold assignments
3. `same_seed_required: true` — Seed 42 throughout
4. `no_feature_dependent_row_loss: true` — Missing features get NaN, not dropped rows
5. `forecast_nowcast_separation_required: true` — DOE-O1 is nowcast only
6. `sar_anomaly_forbidden_in_forecast: true` — Unless timestamp audit clears it
7. `taxi_route_claim_forbidden_without_kappa_reconstruct: true` — DOE-R1 gate

---

## Execution Priority

1. **DOE-C1** (Construct) — data exists, run first
2. **DOE-H1** (Hazard shape) — requires Deltares fetch, high paper value
3. **DOE-P1** (Topology necessity) — requires hydrology fetch, high diagnostic value
4. **DOE-O1** (Observation) — requires SAR fetch per-event, nowcast arm only
5. **DOE-R1** (Reconstruction) — registered extension, blocks taxi-route claim
