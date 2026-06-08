# VERIFY.md -- Pre-Launch, In-Flight, and Post-Flight Verification Gates

Full lifecycle verification: pre-launch resource review, in-flight monitoring,
checkpoint integrity, and post-flight domain validation. Every experiment must
pass ALL gates BEFORE results are cited in the paper.

---

## Gate V0: Pre-Launch Resource Review (9 sections)

Mandatory before EVERY launch and EVERY re-launch. A bug fix that changes
a dependency or data source can silently invalidate the old instance config.

| # | Dimension | Verify | georsct-rerun defaults |
|---|-----------|--------|------------------------|
| 1 | **Git state** | `git status` clean, `git push` done | Commit hash in summary.json |
| 2 | **Memory budget** | largest_object x workers <= 0.5 x instance_RAM | 32k rows x 106 cols x 6 emb ~ 6 GB; 64 GB instance OK |
| 3 | **Volume sizing** | pip_install + temp_files + safety | 30 GB (S019D/S018B), 10 GB (bootstrap) |
| 4 | **Worker count** | CPU-bound: n_jobs = n_cpus; logged at startup | `os.cpu_count()` logged; S019D uses 16 vCPU |
| 5 | **Image + drivers** | pytorch-cpu + scikit-image via bootstrap.sh | `pytorch-training:2.9.0-cpu-py312` + `pip install scikit-image` |
| 6 | **Instance quota** | `aws service-quotas` for target instance type | ml.m5.4xlarge (S019D/S018B), ml.m5.xlarge (bootstrap) |
| 7 | **Dry run** | `--dry-run` shows correct config | All launchers support `--dry-run` |
| 8 | **Checkpoint / idempotency** | Job skips completed targets; partial failure safe | S3 per-target checkpoints in run scripts |
| 9 | **Timeout** | MaxRuntimeInSeconds set; > 2x expected wall clock | S019A: 21600s (6h), S019D/S018B: 43200s (12h), bootstrap: 1800s (30m) |

**Timeout rules:**
- Never omit `StoppingCondition` -- a hung job with no timeout burns money indefinitely
- Set timeout > 2x expected runtime (safety for checkpointing and retries)
- Set timeout < 24h for processing jobs (cost cap)
- If a job previously hit timeout: increase AND investigate root cause (not just bump the number)

---

## Gate V0.5: AWS Monitoring (In-Flight)

Do not assume a launched job will succeed. Check status within the session.

### Immediate (< 5 min after launch)

```bash
# Job accepted?
MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job \
  --processing-job-name JOB_NAME --region us-east-1 --profile nsc-swarm \
  --query 'ProcessingJobStatus'

# Container booting? (first log lines appear within 2-3 min)
MSYS_NO_PATHCONV=1 aws logs tail /aws/sagemaker/ProcessingJobs \
  --log-stream-name-prefix JOB_NAME --follow --region us-east-1 --profile nsc-swarm
```

### Early (5-15 min)

| Check | What to look for |
|-------|------------------|
| Bootstrap succeeded | `pip install` lines, no `ModuleNotFoundError` |
| Data loaded | `Loaded XXXXX ZCTAs` log line |
| First target started | `[1/27] annual_checkup` or similar |
| No encoding crash | No `UnicodeEncodeError` in first 100 log lines |

### Periodic (every 30-60 min for long jobs)

| Check | How |
|-------|-----|
| Job still running | `describe-processing-job` shows `InProgress` |
| Progress advancing | New target checkpoint files appearing on S3 |
| No OOM | No `exit code: 137` or `Killed` in logs |
| Memory stable | No growing RSS if instance supports CloudWatch agent |

### Kill criteria

Stop the job immediately if ANY of these are true:

```bash
# Kill command
MSYS_NO_PATHCONV=1 aws sagemaker stop-processing-job \
  --processing-job-name JOB_NAME --region us-east-1 --profile nsc-swarm
```

- `ModuleNotFoundError` or `ImportError` in first 5 min (wrong code deployed)
- `UnicodeEncodeError` (missing `sys.stdout.reconfigure`)
- `KeyError: 'zcta_id'` (wrong .npz artifact version)
- Same target failing repeatedly (infinite retry without checkpoint)
- Job past 2x expected runtime with no new checkpoints

---

## Gate V0.7: Checkpoint Data Integrity

Applies during and after the run. Checkpoints are the crash-recovery mechanism.

### During run (spot-check via S3)

```bash
# Count completed targets
MSYS_NO_PATHCONV=1 aws s3 ls \
  s3://swarm-yrsn-datasets/rsct_curriculum/series_019_v2/results/s019d/seed_42/checkpoints/ \
  --profile nsc-swarm | wc -l
```

| Check | Pass Criteria |
|-------|---------------|
| Checkpoint count advances | New .json files appearing over time |
| Checkpoint not empty | Each file > 100 bytes (not `[]` or `{}`) |
| Checkpoint parseable | `aws s3 cp ... - \| jq length` returns > 0 |

### After run (before citing)

| Check | How | Pass Criteria |
|-------|-----|---------------|
| 27 checkpoint files | `aws s3 ls .../checkpoints/ \| wc -l` | Exactly 27 (one per CONUS target) |
| Total rows match | Sum of rows across checkpoints | S019D: 810 (6 emb x 27 targets x 5 folds) |
| No duplicate targets | Unique target names across all rows | 27 unique targets |
| Resume worked correctly | If job restarted: no duplicated rows from overlapping checkpoints | `len(results) == len(set((r['target'], r['fold'], r['embedding']) for r in results))` |

---

## Gate V1: Job Completion

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| Exit code 0 | `aws sagemaker describe-processing-job --processing-job-name JOB` | `ProcessingJobStatus: Completed` |
| No truncated output | Check CloudWatch last 50 lines | Final summary line present, no OOM/SIGKILL |
| All targets attempted | `jq '.n_failed' summary.json` | 0 failed targets |
| Result file exists | `aws s3 ls s3://bucket/prefix/` | `_results.json` + `_summary.json` present |

---

## Gate V2: Domain Invariants

These catch silent data pipeline bugs. If any fail, results are INVALID.

### V2.1 Noisy Control Floor

The `noisy_control` embedding is PCA32 + N(0,1) noise. Gates MUST reject it.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| R2 near zero | `mean(r2 where embedding == noisy_control)` | R2 < 0.15 across all targets |
| Gate rejection | `count(EXECUTE where embedding == noisy_control)` | flat: 0 EXECUTE, oobleck: 0 EXECUTE |
| Worse than all real embeddings | `max(noisy_r2) < min(real_emb_r2)` per target | No target where noise beats a real embedding |

**If noisy_control passes gates**: calibration is broken. Stop. Do not cite.

### V2.2 GNN Alignment Sanity

Catches the silent misalignment bug that invalidated original S018B/S019D.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| GNN R2 > 0 | `mean(r2 where embedding == gnn_v2)` | R2 > 0.10 on at least 20/27 targets |
| GNN != PCA | `abs(mean_r2_gnn - mean_r2_pca)` | Difference > 0.02 (not identical) |
| Alignment logged | grep CloudWatch for `reindexed by zcta_id` | Message present with count |
| zcta_id key used | grep CloudWatch for `missing canonical 'zcta_id'` | Message ABSENT (no fallback) |

**If GNN R2 ~ 0 everywhere**: zcta_id alignment failed silently. Check .npz artifact.

### V2.3 Simplex Discrimination

Catches the per-arm calibration bug (FC-1/FC-2) that forced R=S=N=0.33.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| R varies across embeddings | `std(mean_R per embedding)` | std > 0.05 |
| S_sup varies across embeddings | `std(mean_S_sup per embedding)` | std > 0.05 |
| Not all ~0.33 | `max(abs(R - 0.333))` across embeddings | At least one embedding with R > 0.5 or R < 0.2 |
| PCA > noise on R | `mean_R(pca_v1) > mean_R(noisy_control)` | True for all 3+ targets |

**If R/S/N ~ 0.33 for all embeddings**: `shared_boundaries=True` is not working. Terciles are per-arm.

### V2.4 Theory Kappa Consistency

| Check | How | Pass Criteria |
|-------|-----|---------------|
| kappa_compat present | `jq '.[0] | keys' results.json` | Field exists, not null |
| kappa_gate absent | grep results for `kappa_gate` | Zero occurrences |
| theory_kappa range | `min/max(theory_kappa)` | All in [0, 1], mean > 0.3 for best embeddings |
| Proxy vs theory | `corr(theory_kappa, R*(1-N))` | Pearson r > 0.7 (proxy tracks theory) |

---

## Gate V3: Cross-Seed Stability

Run AFTER all 3 seeds (42, 123, 456) complete.

| Check | How | Pass Criteria |
|-------|-----|---------------|
| Ranking preserved | Per-target embedding rank by mean R2 | Same top-2 embeddings across all 3 seeds |
| R2 std across seeds | `std(mean_r2_per_seed)` per target | std < 0.05 for 25/27 targets |
| Gate decision stable | `count(flips across seeds)` per cell | < 10% of cells flip EXECUTE/REJECT across seeds |
| N-ceiling consistent | `std(n_ceiling)` per target | std < 0.03 |

**If rankings flip across seeds**: fold assignment is unstable or sample size is too small for that target.

---

## Gate V4: Protocol-Specific Checks

### V4.1 S019A (Solver Invariance)

| Check | How | Pass Criteria |
|-------|-----|---------------|
| Embedding rank invariant | Spearman rho of embedding R2 ranking across solvers | rho > 0.8 for all 3 target pairs |
| No solver dominance | `max(mean_r2) - min(mean_r2)` across solvers | Delta < 0.10 per target |
| 315 fits total | `len(results)` | Exactly 315 (7 emb x 3 solvers x 3 targets x 5 folds) |

### V4.2 S019D (N-Ceiling Spectrum)

| Check | How | Pass Criteria |
|-------|-----|---------------|
| 810 fits total | `len(results)` | Exactly 810 (6 emb x 27 targets x 5 folds x 1 solver) |
| N-ceiling range | `min/max(n_ceiling)` | Spread across [0.1, 0.9] -- not all clustered |
| Oobleck stricter | `oobleck_pass <= flat_pass` | Oobleck never more permissive than flat |
| Flips exist | `n_flip > 0` | At least some certificates change under oobleck |
| Easy targets pass both | elevation, night_lights gate rates | > 80% EXECUTE under both gatekeepers |
| Hard targets rejected | smoking, binge_drinking gate rates | < 50% EXECUTE under oobleck |

### V4.3 S018B V2 (State-Holdout Extrapolation)

| Check | How | Pass Criteria |
|-------|-----|---------------|
| State column used | grep CloudWatch for `state holdout` | Holdout states listed in log |
| ~10 holdout folds | `len(holdout_states)` in summary | 9-11 states (20% of ~48 CONUS) |
| PDFM comparison valid | `mean_r2_all` in summary | Value is finite, not NaN |
| Extrapolation < interpolation | `mean_r2(s018b) < mean_r2(s019d)` | True (holdout states are harder) |
| No state leakage | Train/test state sets disjoint per fold | Verify from per-fold logs |

### V4.4 Bootstrap CIs

| Check | How | Pass Criteria |
|-------|-----|---------------|
| B = 1000 | `n_boot` in output | Exactly 1000 replicates |
| CIs non-degenerate | `ci_upper - ci_lower` per target | Width > 0.01 for all targets |
| Point estimate inside CI | `ci_lower <= point <= ci_upper` | True for all targets |
| CSV and JSON match | Compare row counts | Same number of entries |

---

## Gate V5: Paper-Readiness

Final gate before citing results in manuscript.

| Check | Pass Criteria |
|-------|---------------|
| All V1-V4 gates passed | No exceptions, no waivers |
| Git hash recorded | `git_hash` in summary.json matches a pushed commit |
| Three seeds completed | All 3 seed directories populated |
| Bootstrap CIs computed | s019d_bootstrap_cis.json exists |
| No kappa_gate in any output | `grep -r kappa_gate results/` returns empty |
| Wall clock documented | `elapsed_seconds` in summary (NeurIPS requirement) |
| Instance type documented | Launcher logs or summary (NeurIPS requirement) |

---

## Verification Script

```bash
# Quick post-flight check (run after downloading results)
python -c "
import json, sys
results = json.load(open(sys.argv[1]))
# V2.1: noisy control
noise_r2 = [r['r2'] for r in results if r.get('embedding') == 'noisy_control']
print(f'Noisy control mean R2: {sum(noise_r2)/len(noise_r2):.4f}' if noise_r2 else 'NO NOISY CONTROL')
# V2.3: simplex discrimination
import collections
emb_R = collections.defaultdict(list)
for r in results:
    emb_R[r.get('embedding','?')].append(r.get('R', 0))
for emb, vals in sorted(emb_R.items()):
    print(f'  {emb}: mean R = {sum(vals)/len(vals):.3f}')
# V2.4: kappa_gate absent
has_kg = any('kappa_gate' in r for r in results)
print(f'kappa_gate in results: {has_kg} (MUST be False)')
" results.json
```

---

## Failure Protocol

If ANY gate fails:

1. **Do not cite the results.** Mark the run as invalid.
2. **Diagnose root cause.** Which fix was ineffective? Which assumption broke?
3. **Fix code, commit, push, relaunch.** No patching results post-hoc.
4. **Re-run full verification.** No partial re-checks.

Post-hoc adjustment of results to pass verification is FORBIDDEN.
The point of this file is to catch bugs, not to certify convenience.
