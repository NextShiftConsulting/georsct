# DOE: NFIP Bias Correction Ablation via Certificate Trajectory

## Experiment ID: s040a

**Status:** LOCKED
**Date:** 2026-07-02
**Series:** 040 — Data Quality Governance
**Predecessor pattern:** s019g (single-variable-under-test, certificate as instrument)

---

## Hypothesis

> Each layer of NFIP bias correction produces a measurable change in the
> full RSCT certificate vector. The certificate trajectory across layers
> diagnoses *which corrections change eligibility* (gate transitions),
> *which shift the binding constraint* (failure_reason changes), and
> *which are inert* (certificate unchanged).

**This experiment is NOT about accuracy.** The RSCT certificate is a
multi-gate eligibility system. A correction that improves R but crashes κ
is not "better" — it is still ineligible, now for a different reason.
The certificate diagnoses *why* data is ineligible and *which correction
fixes which gate*, not whether a single metric went up.

**Null hypothesis:** Bias corrections change model performance metrics
but the certificate gate structure cannot distinguish which corrections
are responsible — all corrections either pass all gates or fail the same gate.

**Alternative hypothesis:** Different corrections bind at different gates.
The gate-transition sequence (which gate blocks, which correction unblocks it)
is unique per scenario and per correction layer, making the certificate
trajectory a diagnostic instrument for data quality governance.

---

## Why Multi-Metric Eligibility (Not Accuracy)

Geospatial flood data violates the assumptions behind ordinary ML metrics:

1. **Space creates dependence.** Nearby ZCTAs are not independent. Spatial
   blocking is necessary, but creates degenerate folds.
2. **Rare events are spatially clustered.** A test fold can easily be all-negative
   or all-positive. accuracy=1.0 / f1=0.0 is not success — the fold cannot
   support the metric.
3. **True negatives dominate.** A model predicting "no flood everywhere" scores
   high on accuracy. Accuracy is useless here.
4. **Metric validity depends on fold contents.** ROC-AUC needs both classes.
   Jaccard needs a non-empty union. F1 needs positive support. The system must
   check *which metrics are measurable* before interpreting values.
5. **Scale changes the answer.** The same event can be visible at ZCTA level
   and disappear at county level (Modifiable Areal Unit Problem).
6. **Geography has topology.** Adjacency, upstream/downstream flow, levee
   boundaries — a model can be numerically right but spatially wrong.
7. **Observation is biased.** Missing positives may mean "no observation" not
   "no flood." Abstention is the honest answer, not model failure.
8. **Validation is part of the science.** A model that works under random CV
   may fail under spatial-blocked CV. Validation design is a scientific claim.

This forces evaluation to become **conditional**:

```
fold eligibility       →  can this fold support any metric?
metric-specific elig.  →  which metrics are measurable on this fold?
certificate elig.      →  does the full gate sequence pass?
```

The bias correction ablation tests whether corrections change eligibility
*at each of these three levels*, not just whether a single accuracy number
goes up. A correction that makes 5 more folds Jaccard-eligible is a
different (and potentially more important) improvement than one that
raises mean F1 by 0.02 on the same eligible set.

The clean line:

> The benchmark must first certify whether a metric is measurable
> before interpreting model performance. Geospatial data makes this
> conditional evaluation necessary, not optional.

---

## Design: Single Variable Under Test (s019g Pattern)

### What is held constant

| Dimension | Value | Why constant |
|-----------|-------|-------------|
| Embedding | GNN_v2 (best from s019d) | Isolate target effect |
| Solver | HistGBDT | Isolate target effect |
| Folds | F=5, blocked bootstrap | Same as s019d/s035 |
| Seeds | {42, 123, 456} | Cross-seed reproducibility |
| ZCTAs | All ZCTAs in scenario | Same geography |
| Features | Identical feature matrix | Same inputs |
| Gate thresholds | R≥0.3, S≥0.4, N≤0.5, κ≥0.7 | Standard gates |
| Certifier | Shared reference calibration | Per-arm trap avoided |

### What varies (the single variable)

The **target variable** (obs_nfip_event_claims) is progressively corrected
through 8 conditions. Each condition applies one additional correction layer
on top of all previous layers:

| Condition | Target Variable | Correction Applied |
|-----------|----------------|-------------------|
| C0 | `y_raw` | Raw NFIP claims (baseline) |
| C1 | `y_tobit` | + Tobit regression (right-censoring at $250K) |
| C2 | `y_ipw` | + IPW selection bias correction |
| C3 | `y_private` | + Private market adjustment (AM Best share) |
| C4 | `y_event` | + Event-specific fraud exclusion (Katrina/Sandy) |
| C5 | `y_break` | + Bai-Perron structural break adjustment (RR 2.0) |
| C6 | `y_credible` | + Buhlmann-Straub credibility weighting (HAZUS blend) |
| C7 | `y_fused` | + Bayesian multi-source fusion (NFIP + IA + SBA) |

**Critical constraint:** C0 through C7 are *cumulative*. C3 = Tobit + IPW + private.
This is an ablation ladder, not independent conditions.

---

## Certificate Vector Tracked Per Condition

### Per-Metric Eligibility (the primary diagnostic)

Each fold emits per-metric eligibility status. The correction layers change
not just metric values but **which metrics become measurable**. A correction
that makes more folds eligible for Jaccard is a different kind of improvement
than one that raises R on already-eligible folds.

```
Per fold, per metric family:
  eligibility_status: ELIGIBLE | SKIP_TRAIN_SINGLE_CLASS | SKIP_EMPTY_UNION | ...
  metric_status: MEASURED | SKIP_EMPTY_UNION | SKIP_TEST_SINGLE_CLASS | MEASURED_FALSE_ALARM_ONLY
  class_support: (train_pos, train_neg, test_pos, test_neg)
```

### Metric Families (all tracked per eligible fold)

```
classification:    balanced_accuracy, f1, mcc, precision, recall
spatial_overlap:   jaccard_iou, jaccard_subset_matched, dice
ranking:           auc_roc, auc_pr
regression:        r2, rmse, mae, nse  (for continuous targets like claims $)
spatial_structure:  morans_i_residual  (spatial autocorrelation in errors)
```

**Jaccard/IoU is a first-class metric, not secondary.** It answers a
different question than F1:

```
Accuracy:  How many labels were correct, including true negatives?
F1:        How well did positive detections balance precision and recall?
AUC-ROC:   Can the model rank positives above negatives?
Jaccard:   Does the predicted event geography overlap the observed event geography?
```

For rare, clustered hazards, Jaccard is the more geospatially meaningful
question. A correction that improves F1 but degrades Jaccard has improved
label accuracy but worsened spatial overlap — the certificate must
distinguish these.

### Jaccard Variants (three levels)

**1. Set-overlap Jaccard (ZCTA-level, primary)**
```
observed_positive_set  = ZCTAs with observed flood label (or claims > 0)
predicted_positive_set = ZCTAs predicted positive above threshold

Jaccard = |observed ∩ predicted| / |observed ∪ predicted|
```
This is set overlap over areal units. Honest about what it is: not true
flood-extent IoU, but spatial agreement on which ZCTAs are affected.

**2. Subset-matched Jaccard (per-event, secondary)**
```
For each predicted flood region, match to at most one observed region.
Compute IoU per matched pair. Penalize unmatched predictions (false alarms)
and unmatched observations (misses).
```
Prevents one large sloppy predicted region from getting credit for
covering multiple separate events. Adapted from arXiv:1611.06880
(plant image segmentation → flood polygon matching).

**3. Field Jaccard (continuous, future)**
```
For continuous targets (claims $, water depth): generalized Jaccard
over scalar fields, not binary masks. From arXiv:2110.09619.
Applicable when comparing corrected vs raw claims distributions
as continuous surfaces over ZCTA geography.
```
Not implemented in s040a. Noted for future extension when raster/polygon
flood extents replace ZCTA-level binary targets.

### Jaccard Eligibility Rules (distinct from F1 eligibility)

```
truth_pos > 0, pred_pos > 0:  MEASURED (Jaccard is meaningful)
truth_pos > 0, pred_pos = 0:  MEASURED (Jaccard = 0, meaningful miss)
truth_pos = 0, pred_pos > 0:  MEASURED_FALSE_ALARM_ONLY (Jaccard = 0)
truth_pos = 0, pred_pos = 0:  SKIP_EMPTY_UNION (do NOT score as perfect)
```

A fold can be eligible for F1 but not for Jaccard (or vice versa).
The certificate reports per-metric eligibility status.

### Eligibility Counts (per scenario × condition)

```
n_attempted, n_eligible, eligibility_ratio
n_eligible_f1, n_eligible_jaccard, n_eligible_auc_pr

Key diagnostic: Does the correction change the eligibility ratio?
  - C4 (Sandy exclusion) may make NYC folds MORE eligible (fewer degenerate folds)
  - C6 (credibility) may make Riverside folds MORE eligible (blended target is less sparse)
  - A correction that raises R but drops eligibility_ratio is suspect
```

### Core 6D (per fold, per seed)

```
R, S_sup, N, alpha, omega, tau
```

### Derived (computed from 6D)

```
alpha_omega, quality_score, uncertainty, health_score
simplex_sum, entropy, collapse_risk, sigma
```

### Gate Verdicts

```
gate_decision (flat), gate_decision (oobleck)
gate_reached, failure_reason
gate_1_integrity, gate_1b_n_ceiling
gate_2_consensus, gate_3_admissibility
kappa_compat (theory), proxy_kappa
oobleck_threshold, margin
```

### Temporal / Trajectory (C0 is baseline)

```
n_delta = N(Ci) - N(C0)
n_trend = sign(N(Ci) - N(Ci-1))
drift_detected = |n_delta| > 2*std(N across seeds at C0)
```

### Lyapunov Stability (C0 is baseline)

```
V = d_T4(cert(Ci), cert(C0))        # distance from baseline
V_dot = V(Ci) - V(Ci-1)             # derivative estimate
is_lyapunov_stable = (V_dot <= 0)   # correction is stabilizing
is_converging = (V_dot < -epsilon)  # correction is actively improving
lyapunov_margin = 1.5 - V           # headroom before instability
```

**Key insight:** V̇ ≤ 0 at every step means the entire correction pipeline
is Lyapunov stable — each correction moves the certificate closer to
eligibility. V̇ > 0 at any step means that correction *destabilized*
the certificate — it may have improved one gate while breaking another.
The Lyapunov trajectory detects corrections that trade eligibility at
one gate for ineligibility at another, which single-metric thinking misses.

---

## Scenarios (same as s035)

Run the full ablation ladder on each scenario to expose geography-specific
bias patterns:

| Scenario | Expected Pattern | Why |
|----------|-----------------|-----|
| Houston | R should increase steadily | Best NFIP penetration, 311 validates |
| NYC | S should jump at C4 (Sandy fraud removal) | Temporal break is dominant corruption |
| New Orleans | R may remain near zero until C7 (fusion) | NFIP data is fundamentally non-informative here |
| SW Florida | R should increase at C1 (Tobit) | High property values hit $250K cap |
| Riverside | R may remain low throughout | Sparse data, 1.2% penetration |

---

## Money Table (primary output)

### Table 1: Multi-Metric Eligibility Trajectory

```
Scenario × Condition × {eligibility_ratio, n_eligible_f1, n_eligible_jaccard,
                         mean_jaccard, mean_f1, mean_mcc, mean_bal_acc}

Rows: 5 scenarios × 8 conditions = 40 rows
Columns: elig% | n_f1 | n_jac | F1 | Jaccard | MCC | BalAcc | R | S | N | α | κ | gate | V | V̇
Values: mean ± bootstrap CI across 3 seeds × 5 folds (on eligible folds only)

Note: Metrics are reported ONLY on eligible folds. Eligibility counts are
reported alongside metric values. A correction that raises Jaccard by 0.05
but drops n_eligible_jaccard from 27 to 15 is NOT an improvement — it just
discarded the hard cases.
```

### Table 2: Gate Binding Constraint by Layer

```
For each scenario × condition:
  Which gate blocks?  (gate_reached + failure_reason)
  Did the blocker change from Ci to Ci+1?

Scenario | C0 blocker          | C1 blocker          | ... | C7 blocker
Houston  | gate_3_kappa_below  | gate_3_kappa_below  |     | ELIGIBLE
NYC      | gate_1_noise_high   | gate_1_noise_high   |     | gate_3_kappa_below
NOLA     | gate_1_noise_high   | gate_1_noise_high   |     | gate_1_noise_high
```

The binding constraint shifting from one gate to another IS the evidence
that a correction fixed the problem at the earlier gate. This is the
multi-metric eligibility signal — not "R went up" but "the *reason for
ineligibility* changed."

### Table 3: Gate Transition Events

```
Which correction layer causes a gate verdict change?

Scenario | Layer | Transition                         | Interpretation
Houston  | C2    | gate_3 BLOCK → gate_3 PASS         | IPW fixed κ-compatibility
Houston  | C2    | RE_ENCODE → EXECUTE                | Now fully eligible
NYC      | C4    | gate_1 BLOCK → gate_1 PASS         | Sandy exclusion reduced N
NYC      | C4    | failure shifts gate_1 → gate_3     | New binding constraint exposed
NOLA     | C7    | gate_1 BLOCK → gate_1 BLOCK        | Still noise-dominated even after fusion
NOLA     | --    | No transition at any layer          | NFIP is fundamentally non-informative here
```

### Table 4: Layer Marginal Certificate Shift

```
For each layer transition Ci → Ci+1:
  ΔR, ΔS, ΔN, Δα, Δκ, ΔV (Lyapunov)

Significant if |Δ| > 2 * pooled_bootstrap_SE

This table is SECONDARY to Tables 2-3. The gate transitions are the
primary evidence. The Δ values explain WHY a gate transition happened
(or didn't).
```

---

## Predictions (falsifiable)

### P1: Tobit (C1) has largest ΔR in high-value scenarios

SW Florida and NYC have expensive properties hitting the $250K cap.
Tobit correction should produce the largest R increase in these scenarios.
**Falsified if:** ΔR(C1) < ΔR(C1) in Houston (which has lower property values).

### P2: Event exclusion (C4) has largest ΔS in NYC

Sandy fraud is a temporal corruption. Removing it should stabilize the
time series, increasing S.
**Falsified if:** ΔS(C4) is not the largest ΔS among all layers for NYC.

### P3: New Orleans R remains near zero through C6

Corrections C1-C6 fix *insurance-side* biases. New Orleans's problem is
*data-side* (Katrina destroyed the population signal). Only C7 (multi-source
fusion with FEMA IA) can introduce new information.
**Falsified if:** R(C6, NOLA) > 0.15.

### P4: Lyapunov stability holds for Houston but breaks for Riverside

Houston has dense data — each correction should stabilize (V̇ ≤ 0).
Riverside has sparse data — corrections may amplify noise (V̇ > 0).
**Falsified if:** Houston shows V̇ > 0 at any layer, or Riverside shows V̇ ≤ 0
at all layers.

### P5: IPW (C2) causes the most gate transitions across scenarios

Selection bias is the most universal NFIP corruption. IPW should unblock
the most gates across the most scenarios.
**Falsified if:** Another single layer causes more gate transitions.

### P6: No correction layer makes Houston *less* eligible

Houston has the cleanest base data. No correction should introduce a new
gate failure that wasn't present before.
**Falsified if:** Any layer introduces a new gate BLOCK in Houston that
was PASS at the previous layer (regression in eligibility).

### P7: The binding constraint sequence differs across scenarios

If the certificate is diagnostic, different scenarios should show different
gate-blocking patterns because the *type* of data corruption differs.
**Falsified if:** All 5 scenarios show the same gate blocker at every layer
(would mean the certificate can't distinguish corruption types).

### P8: Eligibility ratio increases monotonically for New Orleans

New Orleans has 19/44 eligible folds (43%) — the worst ratio. Corrections
that reduce target sparsity (C6 credibility, C7 fusion) should make more
folds eligible by providing non-degenerate target values.
**Falsified if:** eligibility_ratio(C7, NOLA) < eligibility_ratio(C0, NOLA).

### P9: Jaccard and F1 can disagree on correction direction

A correction may improve F1 (better label classification) but degrade
Jaccard (worse spatial overlap) or vice versa. This would prove the
multi-metric system detects quality dimensions invisible to any single metric.
**Falsified if:** ΔJaccard and ΔF1 have the same sign at every layer for
every scenario (would mean the metrics are redundant).

### P10: Abstention patterns are diagnostic of corruption type

NYC should have metric_status=SKIP_EMPTY_UNION folds that become MEASURED
after C4 (Sandy exclusion removes the temporal artifact that created
degenerate targets). The *pattern* of which folds flip from abstained to
eligible tells you which ZCTAs were affected by the corruption.
**Falsified if:** The set of eligible folds does not change between C0 and C4
for NYC.

---

## Implementation Notes

### Data Construction Pipeline

Each correction layer requires specific data inputs:

| Layer | Input Data Required | Source |
|-------|-------------------|--------|
| C1 (Tobit) | Coverage limits per policy | OpenFEMA NFIP Policies |
| C2 (IPW) | SFHA status, income, mortgage, CRS class | NFHL + ACS + HMDA |
| C3 (Private) | State-level private market share | AM Best reports |
| C4 (Event) | Disaster IDs for Katrina/Sandy | FEMA declarations |
| C5 (Break) | Pre/post RR 2.0 regime indicator | Date cutoff Oct 2021 |
| C6 (Credibility) | HAZUS expected loss per ZCTA | HAZUS runs or pre-computed |
| C7 (Fusion) | FEMA IA registrations, SBA loans | OpenFEMA API, data.sba.gov |

### Practical Ordering

Not all layers are equally easy to implement:

| Tier | Layers | Effort | Blocking? |
|------|--------|--------|-----------|
| **Immediate** | C0, C4 (event exclusion) | Low — filter by disaster ID | No |
| **Week 1** | C1 (Tobit), C5 (break) | Medium — regression + date split | No |
| **Week 2** | C2 (IPW) | Medium — need ACS/HMDA join | Need data download |
| **Week 3** | C3 (private), C6 (credibility) | Medium — need AM Best + HAZUS | HAZUS may need runs |
| **Week 4** | C7 (Bayesian fusion) | High — need FEMA IA download + model | OpenFEMA API pull |

### Minimum Viable Ablation (if time-constrained)

Run C0 → C4 → C7 only (raw → event-corrected → fused). This captures
the two largest expected effects (fraud removal and multi-source fusion)
and proves the certificate trajectory concept with 3 points instead of 8.

---

## Independent Verification (_verify_logic.py analog)

Following s019g's pattern, build an independent checker that:

1. Takes the raw and corrected target distributions as input
2. Computes expected certificate direction (ΔR sign) from distribution
   properties alone (variance reduction, mean shift, outlier removal)
3. Compares against actual certificate trajectory
4. Flags any case where the certificate moves opposite to the distributional
   change — this would indicate a certifier bug, not a data quality signal

---

## Relationship to Papers

| Paper | What this ablation provides |
|-------|---------------------------|
| V5b (NeurIPS) | Certificate trajectory as data governance diagnostic — the multi-gate eligibility system exposes which bias correction fixes which gate, not just whether "accuracy improved" (§4, Table 2) |
| V7 (SIGSPATIAL 2027) | Per-construct binding constraint analysis — different flood constructs are blocked by different gates, proving the certificate is geometry-sensitive (§6) |
| V8 (AAAI-27) | Disagreement as eligibility divergence — two sources can both "improve accuracy" but bind at different gates, exposing incompatible quality dimensions (§5) |

### The claim this enables (novel)

> RSCT certificates are a multi-gate eligibility instrument, not an
> accuracy metric. When applied to progressively bias-corrected flood
> damage data, the gate-transition sequence diagnoses which corrections
> address which quality dimensions. A correction that improves R (relevance)
> but degrades κ (compatibility) does not increase eligibility — the
> binding constraint merely shifts. This is invisible to single-metric
> evaluation but explicit in the certificate.

---

## Stop Conditions

1. **C0 baseline certificates don't match s035 baselines** (±0.03) → data pipeline error, stop
2. **All ΔR ≈ 0 across all layers for all scenarios** → corrections are inert, hypothesis falsified
3. **V̇ > 0 for every layer in every scenario** → corrections systematically degrade, hypothesis falsified
4. **Correction C_i crashes or produces NaN** → fix code, don't skip layer

---

## Deliverables

1. `s040a_certificate_trajectories.json` — full certificate vector per condition × scenario × seed × fold
2. `s040a_money_table.csv` — Tables 1-3 above
3. `s040a_lyapunov_trajectories.png` — V(Ci) plots per scenario (5 panels)
4. `s040a_delta_heatmap.png` — ΔR/ΔS/ΔN heatmap (scenarios × layers)
5. `_verify_logic.py` — independent distributional checker
