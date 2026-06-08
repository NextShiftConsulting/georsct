# georsct-rerun

Fresh repo for rerunning GeoRSCT S018B/S019A/S019D experiments with all MMAR fixes applied.

Writes to `series_019_v2/` S3 prefix to preserve original results.

## Structure

```
georsct-rerun/
  shared/           # Shared modules (representations, theory_certifier, constants)
  domain/           # Certificate domain object (kappa_compat, not kappa_gate)
  preflight/        # S3 artifact pre-flight verification
  s019a_v2/         # Certificate invariance gradient (7 emb x 3 solvers x 3 targets)
  s019d_v2/         # Comprehensive theory kappa benchmark (6 emb x 27 targets)
  s018b_v2/         # State-holdout extrapolation benchmark (6 emb x 27 targets)
```

## Fixes Applied

See [FIXES_APPLIED.md](FIXES_APPLIED.md) for the full list of 6 fixes + preflight import fix.

## Launch Order

1. **S019A** (solver invariance) -- validates that embedding ranking is solver-independent
2. **S019D** (N-ceiling spectrum) -- 810 fits with dual gatekeepers
3. **S018B V2** (extrapolation) -- state-holdout for PDFM comparison
4. **S019D Bootstrap** (CIs) -- after all three seeds of S019D complete

## Usage

```bash
# Dry run (pre-flight only)
python s019a_v2/sagemaker_s019a.py --dry-run
python s019d_v2/sagemaker_s019d.py --dry-run
python s018b_v2/sagemaker_s018b_v2.py --dry-run

# Launch all seeds
python s019a_v2/sagemaker_s019a.py --all-seeds
python s019d_v2/sagemaker_s019d.py --all-seeds
python s018b_v2/sagemaker_s018b_v2.py --all-seeds

# Bootstrap CIs (after S019D completes)
python s019d_v2/sagemaker_s019d_bootstrap.py
```

## S3 Layout

All outputs under `s3://swarm-yrsn-datasets/rsct_curriculum/series_019_v2/results/`:
- `s019a/seed_{42,123,456}/` -- S019A results
- `s019d/seed_{42,123,456}/` -- S019D results
- `s019d_bootstrap/` -- Bootstrap CIs
- `s018b_v2/seed_{42,123,456}/` -- S018B V2 extrapolation results
