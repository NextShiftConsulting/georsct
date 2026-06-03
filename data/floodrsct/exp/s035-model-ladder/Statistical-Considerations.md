# Statistical Considerations — S035 Model Ladder

**Experiment:** s035-model-ladder
**Purpose:** Consolidate all statistical design decisions in one place.
**Authority:** This document summarizes; the DOE docs remain authoritative.

---

## 1. Observation Unit and Sample Structure

| Level | Unit | Count | Source |
|-------|------|-------|--------|
| Row | (zcta_id, event) | 132-264 per scenario | Assembled parquet |
| Fold | Spatial-blocked 5-fold CV | 5 per cell | `generate_folds.py` |
| Cell | (scenario, target) | 9 total | EXPERIMENT_CONTRACT.yaml |

**Cells (9):**

| Scenario | Target | Task | n_rows |
|----------|--------|------|--------|
| Houston | obs_nfip_event_claims | regression | 396 |
| Houston | obs_has_311 | classification | 396 |
| Houston | obs_has_hwm | classification | 396 |
| SW Florida | obs_nfip_event_claims | regression | 606 |
| NYC | obs_nfip_event_claims | regression | 422 |
| NYC | obs_has_311 | classification | 422 |
| Riverside | obs_nfip_event_claims | regression | 172 |
| New Orleans | obs_nfip_event_claims | regression | 264 |
| New Orleans | obs_has_hwm | classification | 264 |

---

## 2. Primary Hypothesis Test (H2a)

**Test:** Wilcoxon signed-rank (paired, two-sided)

**Unit of analysis:** Fold-level metric deltas, pooled across all cells.

**Pairing:** Each R0 fold is paired with the R1 fold trained on the same
ZCTAs, same target, same solver. Fold assignment is fixed from Phase 1 —
only the feature set changes. This cancels fold-level difficulty (a hard
fold stays hard in both R0 and R1).

**Sample size:** 9 cells x 5 folds = 45 paired observations.

**Decision rule:** PASS requires BOTH:
- Wilcoxon p < 0.05
- Cohen's d > 0.2 (small effect threshold)

A statistically significant but tiny effect (d < 0.2) is reported as
"detectable but not practically meaningful." A large effect that fails
significance is reported with the CI.

**Multiple comparison correction:** None. H2a is a single pre-registered
primary test comparing R0 vs R1 on the pooled fold-level deltas. There
is no family of primary tests requiring correction.

**Why fold-level, not cell-level:** Cell-level analysis (n=9) has
inadequate power for confirmatory inference. Fold-level pooling treats
each fold-within-cell as a paired observation, giving 45 observations —
sufficient for Wilcoxon signed-rank. The tradeoff: pooling assumes the
representation effect is directionally consistent across cells. If R1
helps some cells and hurts others, the pooled test may miss both. The
cell-level exploratory analysis (Section 4) addresses this.

**Cohen's d formula:**
```
d = mean(deltas) / std(deltas)
```
where `deltas[i] = metric_R1_fold_i - metric_R0_fold_i` for matched folds.

---

## 3. Secondary Hypothesis Tests

### H1: Baseline Skill (Gate)

**Test:** R2 > 0 on spatial_blocked, HistGBDT, for >= 2 of 3 targets
(within Houston, which has all 3).

**No formal statistical test.** This is a descriptive gate: does the
model beat naive mean prediction? Reported as R2 score with per-fold
distribution.

### H3: R1 vs R2 (Temporal Treatment)

**Test:** Same structure as H2a — fold-level Wilcoxon signed-rank on
paired (R1_fold_metric, R2_fold_metric). Same alpha (0.05) and effect
size threshold (Cohen's d > 0.2).

**Sample size:** 45 paired observations (same pooling).

### H7: VLM Signal (Gate)

**Test:** Spearman rho(VLM_risk_score, obs_nfip_event_claims) > 0.3.
Computed per fold, reported as mean +/- std across 5 folds.

**Note on R4 grain and the broadcast question.** R4 VLM scores are
produced at (zcta_id) grain — the VLM assesses a ZCTA's flood risk from
its map and text, with no event input — so a given ZCTA receives one
event-invariant score. This score *can* be broadcast to (zcta_id, event)
grain, which reproduces the exact R0–R2 key, rows, and spatial-blocked
test folds, and a fold-level R2 can then be computed for R4 on those test
sets. The broadcast is therefore mechanically valid, and R4's fold-level
R2 is reported in the money table as a **zero-shot reference column**
(see §3.1 below), on the shared fold scaffold so its per-fold spread is
visually comparable to R0–R2.

**R4 is nonetheless excluded from the paired confirmatory family (H2a,
H3), for two reasons that compound:**

1. **Constant-prediction ties.** For a ZCTA appearing in k events, the
   broadcast emits the identical R4 value k times. Those k rows are one
   prediction stamped k times, not k independent observations of R4
   skill. R0–R2 vary across those same rows. A paired delta
   (R4_fold - R0_fold) is thus computed over a test set where one arm
   carries artificial zero within-ZCTA-across-event variance and the
   other does not. The Wilcoxon null assumes exchangeable paired
   differences; here the pairing is contaminated by a
   constant-prediction artifact rather than a representation effect, so
   the resulting p-value does not carry its usual interpretation.

2. **Regime confound.** R0–R2 are supervised, trained on the fold's
   training rows. R4 is zero-shot. "R4 beats R0 on the same fold"
   conflates *modality* (vision-language vs tabular) with *training
   regime* (zero-shot vs supervised), so a paired delta cannot attribute
   the difference to representation — which is the causal claim the
   ladder exists to support.

**Pre-registration consequence.** H2a is the single pre-registered
confirmatory test and owes no multiple-comparison correction precisely
because it is one comparison. Admitting R4 as a "fourth rung" in the
paired family would either (a) add a family member, forfeiting the
single-test story and requiring correction, or (b) introduce a second
confirmatory test post-hoc — which the negative-results protocol (§9)
explicitly forbids. R4's quarantine in the exploratory rank-correlation
gates (H7–H9) is itself a pre-registration commitment and is not
dissolved after observing the broadcast's convenience.

**What R4 may claim, therefore:** (i) per-fold Spearman rho vs observed
claims, mean +/- std across the 5 shared folds (H7); (ii) pairwise
inter-VLM agreement (H8); (iii) flagged-vs-unflagged comparison against
R0 kappa diagnostics (H9); (iv) a descriptive fold-level R2 reference
column in the money table, flagged zero-shot/event-invariant. **What R4
may not claim:** a paired Wilcoxon against any of R0/R1/R2, or
membership in the H2a/H3 family. The descriptive intuition — "R4 is
handicapped on temporal signal; if it still lands near R0 that is
notable, and the R2-R4 gap quantifies what temporal features add" — is
expressed through the reference column and prose, never through a paired
p-value.

### 3.1 R4 Reference Column Specification

R4's broadcast fold-level R2 appears in the money table as a fenced-off
reference column:

| Column | Source phase | Grain produced -> grain evaluated | Definition | Hypothesis served | Reporting flags |
|--------|-------------|-----------------------------------|------------|-------------------|-----------------|
| `r4_ref_r2` | R4.3 -> R4.5 | (zcta_id) broadcast to (zcta_id, event) | Best-VLM fold-level R2 of the broadcast VLM risk score against the cell's target, computed on the **same spatial_blocked test folds** as R0–R2; reported as mean across 5 folds. "Best-VLM" = the VLM with highest mean per-fold rho in H7 for that cell. | None confirmatory — descriptive reference only | `zero_shot=true`, `event_invariant=true`, `excluded_from_wilcoxon=true` |

Placement rules:
- Position after `R2_R2` and before the certificate fields (`R_cert | S_cert | N_cert`),
  visually separated so no reader mistakes it for a ladder rung
- It carries no `R0->R4_pct` or `R4->anything_pct` uplift cell — uplift
  columns imply paired comparison; R4 has none
- It uses R2 (not rho) only so it sits on the same axis as R0–R2 for
  eyeballing; inferential R4 numbers stay rho-based in the separate R4
  table (DOE_R4_vlm.md)

### H8: VLM Inter-Rater Agreement

**Test:** Pairwise Spearman rho > 0.7 across VLM pairs.

---

## 4. Exploratory Cell-Level Analysis

### H2b: kappa_geom Predicts Uplift

**Test:** Spearman rho(kappa_geom, R0-to-R1 uplift) across cells.

**Sample size:** n=9 cells. This is marginal for rank correlation.

**Reporting requirements:**
1. Effect size (rho) with bootstrap 95% CI (10,000 resamples, seed=42,
   percentile method)
2. Exact permutation p-value (not asymptotic — asymptotic Spearman
   p-values are unreliable at n < 10)
3. Scatter plot with labeled points (each cell identified by
   scenario-target)
4. Honest caveat: "With 9 scenario-target pairs, we report observed
   associations; confirmation requires more scenarios."

**Why kappa_geom and not diag_leakage:** diag_leakage shares the
R0_spatial metric with the uplift calculation, creating a room-to-improve
confound (cells with poor R0 spatial metrics have more room to improve).
kappa_geom is computed pre-training (Phase 0.5) with zero model
dependency.

### H4: Kappa Cascade + Diagnostic Prediction

**Tests:** 8 exploratory Spearman correlations:
- 4 kappa diagnostics x 2 transitions (R0->R1, R1->R2)

**Reported with:** Bootstrap 95% CIs, exact permutation p-values.

**Multiple comparison correction:** Holm-Bonferroni corrected p-values
are reported for transparency but noted as having no practical power at
n=9. Critical rho for Holm-Bonferroni at n=9 with 8 tests is ~0.88 — an
effect this large would be visible by inspection.

**Hit rate reporting:** Exact binomial (Clopper-Pearson) CI on fraction
of cells where kappa flag correctly predicted uplift direction.

### H5: DGM Routing

**Test:** Hit rate (DGM-recommended arm matches exhaustive best arm).
Reported with exact binomial CI.

**At n=9 this is descriptive.** If DGM matches 7/9, exact 95% CI is
[0.44, 0.95] — too wide for confirmatory inference.

---

## 5. Multiple Comparison Corrections

| Test Family | Tests | Correction | Rationale |
|-------------|-------|------------|-----------|
| Primary (H2a) | 1 | None | Single pre-registered comparison |
| Secondary (H3) | 1 | None | Single comparison, different transition |
| VLM gate (H7) | 1 per VLM | None per VLM | Each VLM is an independent gate |
| Cell-level exploratory (H4) | 8 | Holm-Bonferroni | Family of related correlations |
| Spatial lag ablation | 3 | None | Descriptive ablation, not hypothesis test |

**Holm-Bonferroni procedure (for 8 exploratory tests):**
1. Sort 8 p-values ascending: p_(1) <= p_(2) <= ... <= p_(8)
2. Reject p_(i) if p_(i) < alpha / (8 - i + 1)
3. Stop at first non-rejection

**Practical power:** At n=9, Spearman critical value for p < 0.00625
(first Holm step, alpha=0.05/8) is |rho| ~ 0.88. This is reported
honestly as a limitation: "the corrected test has little practical power
at this sample size."

---

## 6. Confidence Intervals

### Bootstrap CIs (cell-level effects)

**Method:** Non-parametric percentile bootstrap
- n_bootstrap = 10,000
- seed = 42
- Resample unit = cells (not rows or folds)
- Report 95% CI [2.5th, 97.5th percentile]

**What they are:** Uncertainty intervals on aggregate effect sizes
(mean uplift, mean kappa movement).

**What they are NOT:**
- NOT hypothesis tests (the Wilcoxon test is the hypothesis test)
- NOT population-level inference from 9 cells
- NOT a replacement for the pre-registered primary test

### Binomial CIs (hit rates)

**Method:** Clopper-Pearson exact binomial
- Used for: kappa flag prediction accuracy, DGM routing hit rate
- Report 95% CI

---

## 7. Split Protocols

| Split | Column | Purpose | Leakage Control |
|-------|--------|---------|-----------------|
| Random 80/20 | fold_random | Diagnostic for diag_leakage | None (intentional) |
| Spatial-blocked 5-fold | fold_spatial_blocked | **Primary benchmark** | County/ZIP3 blocking |
| Leave-event-out | fold_leave_event_out | Transfer test (D.3) | Hold out entire event |

**Spatial blocking algorithm:**
1. Map each ZCTA to its county FIPS
2. If n_counties >= 5: greedy bin-packing by county (balance fold sizes)
3. If n_counties < 5: fall back to ZIP3 prefix blocking
4. Seed = 42 (deterministic)

**Headline results use spatial_blocked only.** Random split results are
reported for diag_leakage computation. Leave-event-out results test
temporal transfer (especially relevant for R2).

---

## 8. Frozen Hyperparameters (No Tuning)

Hyperparameters are fixed by DOE. This is a controlled representation
intervention — the only factor that changes across arms is the feature
set.

| Solver | Parameter | Value |
|--------|-----------|-------|
| HistGBDT | max_iter | 200 |
| HistGBDT | max_depth | 6 |
| HistGBDT | learning_rate | 0.1 |
| HistGBDT | random_state | 42 |
| Ridge | alpha | 1.0 |
| Ridge | imputation | median (SimpleImputer) |
| Ridge | scaling | StandardScaler |

**No tuning per representation.** If R1 outperforms R0, it is because
of the features, not because of hyperparameter advantage.

---

## 9. Kill Rules

| Condition | Action | Severity |
|-----------|--------|----------|
| H1 FAIL on all 3 targets | STOP experiment | Fatal |
| H2a: p >= 0.05 or Cohen's d < 0.2 | Report as null result | Primary null |
| All uplift < 1% | Representation differences are noise; R0 sufficient | Primary null |
| All R1 uplifts negative | R1 features are noise; skip to R2 | Arm failure |
| All R2 uplifts negative | Temporal features are noise | Arm failure |
| wlag_nfip_claims ablation: all uplift from target lag | Report: "neighbor claims predict, not hydrology" | Confound |
| > 50% of runs fail | Fix data pipeline before continuing | Data quality |
| H7 FAIL on all 5 VLMs | VLMs cannot extract flood risk from maps | VLM null |
| Parse success < 50% on any VLM | Fix prompt engineering before rerun | VLM adapter |

**Negative results protocol:** If the primary hypothesis fails:
1. Report null result with effect size and CI
2. Report whether any exploratory tests showed signal
3. Report cascade movement table (kappa progression across levels)
4. Do NOT switch primary hypothesis post-hoc

---

## 10. Pre-Registration and Temporal Ordering

**Primary outcomes locked BEFORE Phase 1 results examined (v1.7):**
- H1: R2 score on obs_nfip_event_claims, spatial_blocked, HistGBDT
- H2a: Paired fold delta on same target/split/solver
- H3: Paired fold delta R1->R2

**S3 timestamp ordering:**
Each phase uploads results with `_s3_result.py` which records
`git_hash`, `timestamp`, and `upload_timestamp`. Kappa diagnostics
(Phase 4a/b/c) are uploaded BEFORE the next training phase begins.
S3 object timestamps constitute a tamper-evident ordering proof.

**Median split for flag threshold (pre-committed):**
No threshold tuning. Median split guarantees balanced groups (4-5 cells
per group at n=9) with zero researcher degrees of freedom.

---

## 11. Spatial Lag Leakage Protocol

R1 includes W-matrix features (spatial lags). Two features carry
elevated leakage risk:

### wlag_nfip_claims (target lag)

Spatial lag of the target variable. Could leak test-fold information
through neighbor target values.

**Mandatory mitigation:**
1. Compute spatial lag PER FOLD using TRAINING ZCTAs only
2. Test ZCTAs receive lag from training neighbors only (NaN if all
   neighbors are in test fold)
3. Report results WITH and WITHOUT wlag_nfip_claims (mandatory ablation)

### spatial_lag_residual_R0 (residual lag)

Uses out-of-fold R0 predictions to compute residuals, then spatially
lags them. Each ZCTA's residual comes from the fold where it was in the
test set (no train-on-self).

### Mandatory Ablation Table

| Variant | Features | Tests |
|---------|----------|-------|
| R1 full | All R1 features | Headline |
| R1 no-wlag | Remove all 8 W-matrix features | Point vs spatial structure |
| R1 no-target-lag | Remove wlag_nfip_claims only | Is target lag driving everything? |
| R1 wlag-only | R0 + 8 W-matrix features | Is hydrology needed beyond spatial? |

All 4 variants reported in the money table. If uplift is entirely from
wlag_nfip_claims, the narrative is "neighbor activity predicts" not
"hydrology features help."

---

## 12. Causal Framing

This experiment is **causal at the pipeline level**: same folds, same
solver, same target — only the representation changes. It is NOT causal
at the hydrology level (we do not randomize watersheds).

**Correct:** "Under controlled representation intervention, adding
hydrologic features improved prediction by X% (Wilcoxon p = Y,
Cohen's d = Z)."

**Incorrect:** "Adding hydrologic features causes better flood
prediction." (Confounded by feature quality, spatial coverage, etc.)

---

## 13. Temporal Gating (IBNR Boundary)

NFIP historical features (`nfip_historical_frequency`,
`nfip_historical_severity`) enforce a strict temporal boundary:

```
Include: claims with dateOfLoss < event.incidentBeginDate
Exclude: same-event claims (target leakage)
```

For multi-event scenarios (Houston 3 events, NOLA 4 events), the same
ZCTA gets different historical features per event because the temporal
cutoff differs. This is computed by `build_nfip_historical.py` and
joined at training time on `(zcta_id, event)`.

**Causal boundary rule:** Only `invariant`, `slow_drift`, and
`event_window` features are legitimate model inputs. `post_event`
features are labels/outcomes. See FEATURE_CONTRACT.yaml header.

---

## Change Log

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-06-02 | Initial consolidation from DOE documents |
| v1.1 | 2026-06-02 | R4 broadcast analysis: quarantine rationale + r4_ref_r2 reference column spec |
