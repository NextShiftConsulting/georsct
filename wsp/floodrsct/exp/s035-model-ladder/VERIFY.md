# VERIFY.md -- Verification Gates for s035-model-ladder
#
# Adapted from georsct-rerun/VERIFY.md for the flood risk model-ladder
# experiment. Every phase must pass its applicable gates BEFORE results
# are cited in the paper.
#
# Relationship to other docs:
#   EXPERIMENT_CONTRACT.yaml -- data dependency chain (packages, S3 artifacts)
#   CHECKLIST.md             -- execution tracker (checkboxes, dates, job names)
#   VERIFY.md (this file)    -- correctness gates (how we know results are valid)
#
# The contract covers ~60% of V0 (inputs exist). This file covers the rest.

---

## Gate V0: Pre-Launch Resource Review

Mandatory before EVERY launch and EVERY re-launch. Applies to all phases.

| # | Dimension | Verify | s035 defaults |
|---|-----------|--------|---------------|
| 1 | **Git state** | `git status` clean, `git push` done | Commit hash in execution record (CHECKLIST.md) |
| 2 | **Memory budget** | largest_parquet x folds <= 0.5 x instance_RAM | Houston ~606 ZCTAs x 5 events x 36 cols ~ small; ml.m5.large (8 GB) OK |
| 3 | **Volume sizing** | pip_install + temp_files + safety | 10 GB default (processing jobs) |
| 4 | **Worker count** | CPU-bound: n_jobs matches instance vCPUs | ml.m5.large = 2 vCPU; Ridge/HGBDT fit is fast |
| 5 | **Image + drivers** | Base image + pip packages match contract | `sklearn-0.3-1` image + `scipy scikit-learn` |
| 6 | **Instance quota** | `aws service-quotas` for ml.m5.large | Check us-east-1 processing quota |
| 7 | **Dry run** | `--dry-run` shows correct config | All launchers support `--dry-run` |
| 8 | **Checkpoint / idempotency** | Job skips completed scenarios; partial failure safe | Per-scenario JSON checkpoints on S3 |
| 9 | **Timeout** | MaxRuntimeInSeconds set; > 2x expected wall clock | 3600s (1h) default; jobs run ~5 min each |

**Contract cross-check:** `validate_experiment_readiness.py --dry-run` verifies
packages importable and S3 artifacts present. That covers dimensions 5 (partially)
and 8 (input side). Dimensions 1-4 and 6-9 are manual.

---

## Gate V0.5: AWS Monitoring (In-Flight)

Do not assume a launched job will succeed. Check within the session.

### Immediate (< 5 min after launch)

```bash
# Job accepted?
MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job \
  --processing-job-name JOB_NAME --region us-east-1 --profile nsc-swarm \
  --query 'ProcessingJobStatus'

# Container booting?
MSYS_NO_PATHCONV=1 aws logs tail /aws/sagemaker/ProcessingJobs \
  --log-stream-name-prefix JOB_NAME --follow --region us-east-1 --profile nsc-swarm
```

### Early (5-15 min)

| Check | What to look for |
|-------|------------------|
| Bootstrap succeeded | `pip install` lines, no `ModuleNotFoundError` |
| Data loaded | `Loaded N ZCTAs for {scenario}` log line |
| First target started | `[1/N] obs_nfip_event_claims` or similar |
| No encoding crash | No `UnicodeEncodeError` in first 50 log lines |

### Kill criteria

Stop immediately if ANY of these are true:

- `ModuleNotFoundError` or `ImportError` in first 5 min
- `UnicodeEncodeError` (non-ASCII in print/log)
- `KeyError: 'zcta_id'` (wrong parquet schema)
- Job past 2x expected runtime with no new S3 output
- OOM (`exit code: 137` or `Killed`)

---

## Gate V0.7: Checkpoint Data Integrity

### During run (spot-check via S3)

```bash
# Count completed result files
MSYS_NO_PATHCONV=1 aws s3 ls \
  s3://swarm-floodrsct-data/results/s035/ \
  --profile nsc-swarm | grep ".json"
```

| Check | Pass Criteria |
|-------|---------------|
| Result files appearing | New `.json` files over time |
| Files not empty | Each file > 100 bytes |
| Files parseable | `aws s3 cp ... - \| python -m json.tool` succeeds |

### After run

| Check | Pass Criteria |
|-------|---------------|
| Per-scenario files complete | One result JSON per scenario per level |
| Prediction parquets present | `r0_{scenario}_predictions.parquet` for all 5 scenarios |
| Fold files present | `folds/{scenario}_folds.parquet` for all 5 scenarios |
| No duplicate folds | Each fold file has unique (zcta_id, event) pairs |

---

## Gate V1: Job Completion

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| Exit code 0 | `describe-processing-job` | `ProcessingJobStatus: Completed` |
| No truncated output | CloudWatch last 50 lines | Final summary line present, no OOM/SIGKILL |
| All scenarios attempted | Count result files on S3 | 5 per-scenario files per level |
| Summary file exists | `aws s3 ls` | Result JSON present with `timestamp` field |

---

## Gate V2: Domain Invariants

These catch silent pipeline bugs. If any fail, results for that phase are INVALID.

### V2.1 Simplex Validity

Every certificate must satisfy R + S_sup + N = 1.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| Simplex closes | `abs(R + S_sup + N - 1.0)` per cell | < 1e-6 for all cells |
| No all-zero certificates | `min(R + S_sup + N)` | > 0.99 (no degenerate cells) |
| S_sup notation | grep results for bare `"S":` without `S_sup` | Zero occurrences of bare S in certificate context |

### V2.2 Simplex Discrimination

Certificates must vary across representation levels. If R/S/N are identical
at R0/R1/R2, the proxy is not measuring representation change.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| R varies across levels | `std(mean_R per level)` | std > 0.02 for at least 3/5 scenarios |
| Not all ~0.33 | `max(abs(R - 0.333))` across levels | At least one cell with R > 0.5 or R < 0.2 |
| R1 >= R0 for primary target | Compare mean R at R0 vs R1 | True for obs_nfip_event_claims in >= 3 scenarios |

**If R/S/N identical across levels**: proxy calibration is broken (per-arm tercile bug).

### V2.3 Kappa Geometry Independence

kappa_geom must be pre-training and level-invariant.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| kappa_geom present | `geometry_kappa.json` on S3 | File exists, not empty |
| Not all NaN | Count non-NaN kappa_geom values | >= 6/11 cells have finite kappa_geom |
| Level-invariant | Same kappa_geom for R0/R1/R2 of same (scenario, target) | Identical values (same source file) |
| Pre-training timestamp | S3 timestamp of geometry_kappa.json | Before earliest r0_{scenario}.json |
| No model dependency | grep compute_geometry_kappa.py for model/predict/fit | Zero occurrences |

**If kappa_geom = 0.0 everywhere**: geometry file missing or not loaded (known prior bug).

### V2.4 Gearbox Warmup Signals (Phase 0.75)

Quality signals must be data-grounded and independent of downstream training.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| tau in [0, 1] | `min(tau), max(tau)` across cells | All in [0, 1] |
| sigma >= 0 | `min(sigma)` | Non-negative for all cells |
| alpha_warmup in [0, 1] | `min(alpha), max(alpha)` | All in [0, 1] |
| collapse_risk in [0, 1] | `min(cr), max(cr)` | All in [0, 1] |
| coherence in [0, 1] | `min(coh), max(coh)` | All in [0, 1] |
| Signals vary | `std(tau)` across cells | std > 0.01 (not constant) |
| Gear assignments vary | `len(set(gear))` | >= 2 distinct gears assigned |
| No model artifacts read | grep compute_gearbox_warmup.py for `_load.*model\|predict` | Ridge CV only, no pre-trained model loading |

**If all cells get same gear**: calibration thresholds are too wide for N=15 cells.

### V2.5 Decision Vocabulary

Per ADR-029/034, internal and public decisions must use correct vocabularies.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| InternalDecision used | grep routing output for decision values | Only EXECUTE/REJECT/BLOCK/RE_ENCODE/REPAIR/WARN/FALLBACK |
| PublicDecision present | grep routing output for `public_decision` | EXECUTE/CAUTION/REFUSE only |
| No MorphType in output | grep for `verification\|ensemble\|pruning` in decision fields | Zero occurrences (MorphType is graph morphisms, not gatekeeper decisions) |
| Projection consistent | Map InternalDecision -> PublicDecision | REPAIR/BLOCK -> REFUSE; RE_ENCODE/WARN/FALLBACK -> CAUTION; EXECUTE -> EXECUTE |

---

## Gate V3: Cross-Fold Stability

s035 uses 5-fold CV (not 3 seeds). Stability checks operate on fold variance.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| Fold R2 spread reasonable | `max(fold_r2) - min(fold_r2)` per cell | Range < 0.30 for >= 80% of cells |
| Spatial vs random agreement | Rank correlation of mean metrics | Spearman rho > 0.6 between split types |
| Best arm stable across folds | Per-fold best arm assignment | Same best arm in >= 4/5 folds for majority of cells |
| sigma tracks instability | `corr(sigma, fold_r2_std)` | Positive correlation (sigma reflects actual variance) |

**If fold rankings wildly unstable**: spatial blocking may be too aggressive for small-N scenarios.

---

## Gate V4: Protocol-Specific Checks

### V4.1 Representation Ladder (R0 -> R1 -> R2)

| Check | How | Pass Criteria |
|-------|-----|---------------|
| Feature counts match contract | Column count per level | R0: 33-36, R1: 58-63, R2: 67-72 |
| R1 features superset of R0 | Set difference of column names | R0 columns all present in R1 |
| R2 features superset of R1 | Set difference | R1 columns all present in R2 |
| No accidental target leakage | grep assembled parquet columns | No `obs_*` in feature list (only in targets) |
| Fold reuse across levels | Compare fold files | Same `folds/{scenario}_folds.parquet` used at R0/R1/R2 |

### V4.2 Money Table + Hypothesis Tests (Phase 5)

| Check | How | Pass Criteria |
|-------|-----|---------------|
| All cells reported | Count cells in money_table.json | >= 9 cells (no cherry-picking) |
| H2a test present | `wilcoxon_p` field exists | Fold-level Wilcoxon signed-rank for R0 vs R1 |
| Negative results included | Check for cells where R1 < R0 | Reported honestly, not dropped |
| Bootstrap CIs computed | `ci_lower, ci_upper` fields | Present for uplift estimates |
| Exploratory tests labeled | `exploratory: true` flag | Cell-level associations marked exploratory |

### V4.3 DGM Routing (Phase 6)

| Check | How | Pass Criteria |
|-------|-----|---------------|
| Both strategies evaluated | `strategies.kappa_first` and `strategies.gear_informed` | Both present in output |
| Hit rate with CI | `hit_rate` + `binomial_ci` fields | Clopper-Pearson exact CI (not binom.interval) |
| Proof-of-concept framing | `n` in output | n < 15, framed as exploratory |
| InternalDecision vocabulary | Decision values in routing table | Not MorphType values |
| Actual best arm correct | Cross-check against money table | Same best arm identification method |

### V4.4 Gearbox Calibration (Phase 0.75)

| Check | How | Pass Criteria |
|-------|-----|---------------|
| MIN_CALIBRATION_SAMPLES >= 8 | grep source code | Threshold not lowered |
| Winsorization applied | `winsor_bounds` in output | Non-null for calibrated runs |
| Monotonicity guard | Gear boundaries monotonically ordered | FIRST < SECOND < THIRD thresholds |
| Oobleck params data-grounded | `sigma_c` derived from data | Not hardcoded constant |
| Cells dropped for <2 folds | Count cells with `folds < 2` | Logged as warning, excluded from calibration |

---

## Gate V5: Paper-Readiness

Final gate before citing results in the manuscript.

| Check | Pass Criteria |
|-------|---------------|
| All V1-V4 gates passed | No exceptions, no waivers |
| Git hash recorded | Commit hash in CHECKLIST.md execution record matches pushed commit |
| All 5 scenarios complete | R0 + R1 + R2 results for all 5 metro areas |
| Certificates at all 3 levels | certificates_r0/r1/r2.json on S3 |
| Money table final | money_table.json reflects latest run |
| DGM routing dual-strategy | Both kappa-first and gear-informed evaluated |
| No `kappa_gate` in output | `grep -r kappa_gate results/` returns empty |
| No bare `S` in certificates | Always `S_sup` in certificate fields |
| Wall clock documented | Duration in CHECKLIST.md execution record (SIGSPATIAL requirement) |
| Instance type documented | Instance in CHECKLIST.md execution record (SIGSPATIAL requirement) |
| InternalDecision->PublicDecision wired | Routing output includes public_decision field |

---

## Failure Protocol

If ANY gate fails:

1. **Do not cite the results.** Mark the run as invalid in CHECKLIST.md.
2. **Diagnose root cause.** Which assumption broke?
3. **Fix code, commit, push, relaunch.** No patching results post-hoc.
4. **Re-run full verification.** No partial re-checks.

Post-hoc adjustment of results to pass verification is FORBIDDEN.
The point of this file is to catch bugs, not to certify convenience.
