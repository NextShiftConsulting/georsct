# Fixes Applied -- GeoRSCT Rerun

All fixes identified by MMAR (Multi-Model Adversarial Review) of GeoRSCT V3 paper,
cross-referenced against rsct-governance canonical docs.

## Fix 1: GNN zcta_id Alignment

**File**: `shared/representations.py`
**Issue**: GNN latent .npz files have arbitrary row order. Without reindexing by `zcta_id`,
embeddings are silently misaligned with the DataFrame, producing near-zero R2.
**Fix**: `_load_latents()` now requires `zcta_id` key in .npz, builds reindex mapping,
and logs alignment statistics.

## Fix 2: kappa_gate -> kappa_compat

**File**: `domain/certificate.py`
**Issue**: `kappa_gate` is RETIRED (ADR-020 D7, ADR-034). Code used the old field name.
**Fix**: Renamed field to `kappa_compat` (= R*(1-N), simplex proxy). Three occurrences:
field declaration, `to_yrsn_dict()` key, `from_yrsn_dict()` key.

## Fix 3: Pooled Tercile Calibration (shared_boundaries=True)

**Files**: `s019a_v2/run_s019a.py`, `s019d_v2/run_s019d.py`, `s018b_v2/run_s018b_extrapolation.py`
**Issue**: Per-arm tercile calibration forces R~=S~=N~=0.33 by construction (FC-1/FC-2).
**Fix**: `certify_group(..., shared_boundaries=True)` pools residuals across all embedding
arms so the simplex can discriminate embedding quality.

## Fix 4: S3 Prefix Isolation (series_019_v2/)

**Files**: All launchers and run scripts
**Issue**: Original runs wrote to `series_019/`. Rerun must not overwrite originals.
**Fix**: All S3 prefixes changed to `series_019_v2/` (code, results, checkpoints).

## Fix 5: Bootstrap Results Prefix

**Files**: `s019d_v2/run_s019d_bootstrap.py`, `s019d_v2/sagemaker_s019d_bootstrap.py`
**Issue**: Bootstrap script pointed at original results prefix.
**Fix**: RESULTS_PREFIX updated to `rsct_curriculum/series_019_v2/results/s019d`.

## Fix 6: S018B State-Holdout Extrapolation (New)

**Files**: `s018b_v2/run_s018b_extrapolation.py`, `s018b_v2/sagemaker_s018b_v2.py`, `s018b_v2/bootstrap.sh`
**Issue**: Original S018B used PCA32+GBDT only, no certificates, no theory kappa, no gatekeepers.
Table 2 cross-protocol comparison was invalid because it predated the GNN alignment fix.
**Fix**: New S018B V2 uses same 6-embedding certification framework as S019D but with
state-holdout protocol (PDFM sec 3.2: random 20% of states). Produces certificates with
theory kappa and dual gatekeeper evaluation (flat + oobleck) for valid cross-protocol comparison.

## Preflight Import Fix

**Files**: All `sagemaker_*.py` launchers, `preflight/` package
**Issue**: Launchers imported preflight from `../../series_018/shared/preflight.py` which
doesn't exist in this standalone repo.
**Fix**: Copied preflight module into `preflight/` package; updated all launcher imports
to `from preflight import ...`.
