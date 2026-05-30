# DOE: S035-Model-Ladder — Diagnosis-Driven Representation Ablation

**Experiment ID:** s035-model-ladder
**Domain:** GeoRSCT / Flood Risk Representation
**Status:** LOCKED
**Last Updated:** 2026-05-29
**Depends on:** S035 Data Lock A (assembled parquets), Stage 5 audit battery (18 checks)

---

## Abstract

Test whether GeoRSCT audit flags predict which representation fixes improve
flood risk prediction. The money table: B.1 flags MAUP -> R1 adds HUC features
-> R1 beats R0 = audit predicted the fix. This is a controlled representation
intervention: same folds, same solver, same target — only the feature set changes.

---

## Problem Statement

ZCTA-level flood risk models use tabular features (R0) that ignore hydrologic
boundaries (watersheds, catchments) and temporal dynamics (storm progression).
GeoRSCT audits flag these as failure modes (B.1 MAUP, C.1 vintage drift), but
do these flags actually predict where adding features helps?

---

## Hypotheses

### H1: R0 Baseline Establishes Measurable Skill

**Statement:** HistGBDT on R0 features achieves RMSE < naive mean baseline on
at least 2 of 3 targets under county-blocked CV.

| Variable Type | Description |
|---------------|-------------|
| Independent | Solver (HistGBDT, Ridge) |
| Dependent | RMSE (log1p targets), ROC-AUC (binary HWM) |
| Control | Fold assignments (fixed), feature set (R0) |

**Metrics:**
- `rmse_vs_baseline`: RMSE / naive_mean_RMSE < 1.0
- `roc_auc_hwm`: ROC-AUC for HWM presence > 0.55

**Evidence Required:** `evidence/h1_r0_baseline.json`

### H2: R1 Improves Over R0 When B.1 Is Active

**Statement:** When audit B.1 (MAUP) flags a scenario, adding HUC/catchment
features (R1) reduces RMSE by > 5% relative to R0 on the primary target.

| Variable Type | Description |
|---------------|-------------|
| Independent | Representation (R0 vs R1) |
| Dependent | Paired delta: metric_R1 - metric_R0, per fold |
| Control | Solver, folds, target (all fixed) |

**Metrics:**
- `uplift_when_flagged`: mean(metric_R1 - metric_R0) for B.1-flagged scenarios
- `uplift_when_not_flagged`: mean(metric_R1 - metric_R0) for B.1-clean scenarios
- `diagnostic_gain`: uplift_when_flagged - uplift_when_not_flagged

**Evidence Required:** `evidence/h2_r1_uplift.json`

### H3: R2 Improves Over R1 When Temporal Mismatch Is Active

**Statement:** When audit C.1 (vintage drift) or temporal features are
misaligned, adding temporal event features (R2) reduces RMSE by > 3%
relative to R1 on the primary target.

| Variable Type | Description |
|---------------|-------------|
| Independent | Representation (R1 vs R2) |
| Dependent | Paired delta: metric_R2 - metric_R1, per fold |
| Control | Solver, folds, target (all fixed) |

**Metrics:**
- `uplift_r2_vs_r1`: mean(metric_R2 - metric_R1)
- `diagnostic_gain_temporal`: conditional uplift when C.1 flagged vs not

**Evidence Required:** `evidence/h3_r2_uplift.json`

### H4: Audit Flags Predict Representation Uplift

**Statement:** Scenarios where specific audits flag a failure mode show
statistically larger representation uplift than scenarios where the same
audit passes.

| Variable Type | Description |
|---------------|-------------|
| Independent | Audit flag status (FAIL/WARN vs PASS) |
| Dependent | Representation uplift (R1-R0 for B.1, R2-R1 for C.1) |
| Control | Solver, folds, target |

**Metrics:**
- `audit_uplift_correlation`: Spearman rho between flag count and uplift
- `diagnostic_gain_significance`: bootstrap 95% CI excludes zero

**Evidence Required:** `evidence/h4_audit_uplift_link.json`

---

## Representation Bundles

| Bundle | Features | Added Over Prior |
|--------|----------|------------------|
| R0 | ZCTA tabular: NFIP, ACS, SVI, flood zone %, TWI, slope, impervious, 311 | -- |
| R1 | R0 + HUC/catchment area, NHDPlus stream density, HCFCD district, levee rating, sewershed | Hydrologic/infrastructure |
| R2 | R1 + peak 1h/3h rainfall, storm duration, surge-rain overlap, lag between peaks | Temporal event dynamics |

---

## Targets

| Target | Transform | Bias Profile | Split Sensitivity |
|--------|-----------|--------------|-------------------|
| NFIP loss (obs_nfip_event_claims) | log1p | Insurance penetration, wealth | Primary |
| 311 count (obs_311_flood_reports) | log1p | Reporting culture, phone access | Secondary |
| HWM presence (obs_hwm_present) | binary | Physical ground truth, sparse | Validation |

---

## Solvers

| Solver | Why | NaN Handling |
|--------|-----|-------------|
| HistGBDT (sklearn HistGradientBoostingRegressor) | Handles NaN natively, robust baseline | Built-in |
| Ridge (sklearn Ridge + StandardScaler pipeline) | Linear baseline, detects when tree isn't needed | Impute + scale |

---

## Split Protocols

| Split | Purpose | Implementation |
|-------|---------|----------------|
| Random 80/20 | Diagnostic for A.1 leakage measurement | seed=42, stratified |
| County-blocked 5-fold CV | Main benchmark, eliminates spatial leakage | Greedy bin-packing by county |
| Leave-one-event-out | Transfer test (D.3) | Hold out entire event |

All splits use **fixed fold IDs** saved as columns in a folds parquet file.
Every representation level reads the same fold assignments.

---

## Experiment Matrix

| Phase | Script | Input | Output | Instance | Est. Duration |
|-------|--------|-------|--------|----------|---------------|
| 0 | `stratified_coverage_audit.py --upload` | Assembled parquets | `evidence/qa/coverage_audit_{scenario}.json` | ml.m5.large | ~15 min |
| 1 | `train_r0_baseline.py` | Assembled parquet + crosswalk | Folds parquet + results JSONs + predictions | ml.m5.xlarge | ~5 min |
| 2 | `train_r1_hydrology.py` | Parquet + folds + HUC features | `results/r1_*.json` | ml.m5.xlarge | ~30 min |
| 3 | `train_r2_temporal.py` | Parquet + folds + temporal features | `results/r2_*.json` | ml.m5.xlarge | ~30 min |
| 4 | `compute_uplift_table.py` | All results JSONs + audit manifest | `evidence/h2-h4_*.json` + money table | local | ~2 min |

**Phase 0** reuses the existing audit orchestrator (no new script).
**Phase 1** generates folds inline then trains -- one SageMaker job, not three.

---

## Decision Tree

```
Phase 0: Data Lock Manifest
  |
  +-- Any FAIL on P1-P6 support probes?
  |     YES -> Fix data pipeline, do NOT proceed
  |     NO  -> Continue
  |
Phase 1: Generate Folds
  |
Phase 2: R0 Baseline (Houston first)
  |
  +-- H1 PASS (skill > naive)?
  |     YES -> Proceed to R1
  |     NO  -> Representation is too sparse; add features before R1
  |
Phase 2: R1 Hydrology
  |
  +-- H2 PASS (R1 > R0 when B.1 flagged)?
  |     YES -> Clean GeoRSCT story: audit predicted the fix
  |     NO  -> B.1 flag is noise, or HUC features don't help this target
  |
  +-- B.1 flagged AND H1 PASS?
  |     YES -> Conditional: run GNN on R1 adjacency graph
  |            (tests graph structure vs tabular HUC features)
  |     NO  -> Skip GNN
  |
Phase 3: R2 Temporal
  |
  +-- H3 PASS (R2 > R1)?
  |     YES -> Temporal collapse was real
  |     NO  -> Temporal features don't add beyond R1
  |
Phase 4: Uplift Table
  |
  +-- H4 PASS (audit flags predict uplift)?
        YES -> Paper core result: audits are diagnostic
        NO  -> Audits flag real issues but don't predict which fix helps
```

---

## Paper-Safe Wording

This experiment is **causal at the pipeline level**: same folds, same solver,
same target — only the representation changes. It is NOT causal at the
hydrology level (we do not randomize watersheds).

**Correct framing:** "Under controlled representation intervention, scenarios
flagged by audit B.1 showed X% larger improvement when hydrologic features
were added."

**Incorrect framing:** "Adding hydrologic features causes better flood
prediction." (Confounded by feature quality, spatial coverage, etc.)

---

## Success Criteria

| Hypothesis | Criterion | Status |
|------------|-----------|--------|
| H1 | RMSE/baseline < 1.0 on >= 2/3 targets | PENDING |
| H2 | diagnostic_gain > 0, 95% CI excludes 0 | PENDING |
| H3 | uplift_r2_vs_r1 > 0.03 on primary target | PENDING |
| H4 | audit_uplift_correlation rho > 0.3 | PENDING |

---

## Kill Rules

- H1 FAIL on all 3 targets -> Data quality too low for any modeling; stop
- H2 diagnostic_gain CI includes 0 -> Report as null result (audits don't predict R1 uplift)
- All uplift < 1% -> Representation differences are noise; R0 is sufficient

---

## DO NOT Constraints

- Do NOT change fold assignments between representation levels
- Do NOT tune hyperparameters per representation (use defaults)
- Do NOT run on scenarios without completed assembly (check manifest)
- Do NOT use GPU instances (tabular models only)
- Do NOT add features to R0 after baseline is locked
- Do NOT impute targets (drop rows with NaN target)

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-05-29 | Initial DOE (DRAFT) |
| v1.1 | 2026-05-29 | LOCKED for execution |
