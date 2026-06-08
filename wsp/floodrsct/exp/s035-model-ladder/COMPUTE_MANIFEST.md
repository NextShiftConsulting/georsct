# Compute Manifest — s035-model-ladder

Generated: 2026-06-07
Source: AWS SageMaker job metadata (account 865679935554, us-east-1)

## Overview

All s035 computation ran as SageMaker Processing Jobs (no Training Jobs).
The experiment spanned 2026-05-28 through 2026-06-07 (~11 days).
Total jobs launched: ~160+ (across all pages); mix of completed, failed, and stopped.

## Instance Types Used

| Instance Type | vCPU | RAM (GiB) | Usage Pattern |
|---------------|------|-----------|---------------|
| ml.m5.xlarge  | 4    | 16        | Model training (R0/R1/R2), spatial diagnostics, ZCTA aggregation, data fetch |
| ml.m5.large   | 2    | 8         | Verdicts, VLM quality scoring, fast validation, uplift tables |

No GPU instances were used. All work was CPU-only.

## Processing Jobs by Category

### Phase 1: Data Fetch & Staging (May 28–Jun 1)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-fetch-mrms-* | ml.m5.xlarge | 100 | 3–5 min | ~8 |
| s035-fetch-tides-* | ml.m5.xlarge | 100 | 3–5 min | ~8 |
| s035-fetch-surge-* | ml.m5.xlarge | 100 | 3–5 min | ~6 |
| s035-fetch-dem-* | ml.m5.xlarge | 100 | 3–5 min | 5 |
| s035-fetch-nlcd-* | ml.m5.xlarge | 100 | ~28 min | 1 |
| s035-fetch-nsi-all | ml.m5.xlarge | 100 | 3–5 min | 1 |
| s035-fetch-nwis-* | ml.m5.xlarge | 10 | 3–5 min | ~4 |
| s035-fetch-openfema | ml.m5.xlarge | 10 | 3–5 min | 2 |
| s035-fetch-slosh | ml.m5.xlarge | 10 | 3–5 min | 1 |
| s035-stage-zcta-geometry | ml.m5.xlarge | 10 | 3–5 min | 1 |
| s035-stage-nlcd-geotiff | ml.m5.xlarge | 10 | 3–5 min | 1 |

### Phase 2: Event Building (May 29–Jun 6)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-build-events-{region} | ml.m5.xlarge | 10 | 2–4 min | ~20 |
| s035-build-nfip-hist-* | ml.m5.xlarge | 10 | 2–3 min | 2 |
| s035-build-r1-* | ml.m5.xlarge | 10 | 2–3 min | 2 |
| s035-build-r2-* | ml.m5.xlarge | 10 | 2–3 min | 1 |

### Phase 3: Model Training — R0 Baseline (May 30–Jun 6)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-r0-baseline-{region} | ml.m5.xlarge | 10 | 2–3 min | 2 (SW-FL, NYC) |

### Phase 4: Model Training — R1 Spatial (Jun 3–Jun 6)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-r1-full-{region} | ml.m5.xlarge | 10 | 2–3 min | 7 |
| s035-r1-nowlag-houston | ml.m5.xlarge | 10 | 2–3 min | 2 |
| s035-r1-notargetlag-houston | ml.m5.xlarge | 10 | 2–3 min | 2 |
| s035-r1-wlagonly-houston | ml.m5.xlarge | 10 | 2–3 min | 2 |

### Phase 5: Model Training — R2 Temporal (Jun 5)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-r2-temporal-{region} | ml.m5.xlarge | 10 | 2–3 min | 5 |

### Phase 6: Certification & Diagnostics (Jun 5)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-certificates-r0 | ml.m5.xlarge | 10 | 2–3 min | 4 |
| s035-certificates-r1 | ml.m5.xlarge | 10 | 2–3 min | 4 |
| s035-certificates-r2 | ml.m5.xlarge | 10 | 2–3 min | 4 |
| s035-diagnostics-r0/r1/r2 | ml.m5.xlarge | 10 | 2–3 min | 3 |
| s035-geometry-kappa | ml.m5.xlarge | 10 | 2–3 min | 3 |
| s035-dgm-routing | ml.m5.xlarge | 10 | 2–3 min | 5 |
| s035-gearbox-warmup | ml.m5.xlarge | 10 | 2–3 min | 1 |

### Phase 7: R3 Certificate-Gated Admission (Jun 5–Jun 6)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-r3-block-tests-{region} | ml.m5.xlarge | 30 | 2–3 min | 10 |
| s035-r3-order-robust-{region} | ml.m5.xlarge | 10 | 2–3 min | 5 |
| s035-r3-block-admission | ml.m5.xlarge | 10 | 2–3 min | 2 |
| s035-r3-certified-{region} | ml.m5.xlarge | 10 | 2–3 min | 9 |
| s035-r3-money-table | ml.m5.xlarge | 10 | 2–3 min | 3 |
| s035-uplift-table | ml.m5.large | 10 | 4–5 min | 2 |

### Phase 8: Spatial Diagnostics — LISA & GWR (Jun 4–Jun 7)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-lisa-r0/r1/r2-{region} | ml.m5.xlarge | 10 | 2–3 min | 15 |
| s035-fig5-lisa-houston | ml.m5.xlarge | 10 | 2–3 min | 3 |
| s035-fig6-gwr-houston | ml.m5.xlarge | 10 | 2–3 min | 3 |
| s035-sidecar-gwr | ml.m5.xlarge | 10 | 2–3 min | 1 |

### Phase 9: VLM / R4 Quality Assessment (Jun 2–Jun 7)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-vlm-gemini-flash-{region} | ml.m5.large | 10 | 3–8 min | ~5 |
| s035-vlm-comparison | ml.m5.large | 10 | 2–3 min | 2 |
| s035-quality-gpt4o-{region} | ml.m5.large | 10 | 2–3 min | 5 |
| s035-quality-gemini-flash-{region} | ml.m5.large | 10 | 2–3 min | 5 |
| s035-quality-jina-{region} | ml.m5.large | 10 | 2–3 min | 5 |
| s035-quality-nova-{region} | ml.m5.large | 10 | 2–3 min | 5 |
| s035-quality-qwen-{region} | ml.m5.large | 10 | 2–3 min | 5 |

### Phase 10: ZCTA Aggregation & Region Rendering (Jun 5–Jun 7)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-fast-zcta-houston | ml.m5.xlarge | 50 | 13 min | 1 |
| s035-fast-zcta-nyc | ml.m5.xlarge | 50 | 6 min | 1 |
| s035-region-all | ml.m5.xlarge | 10 | 2–3 min | 5 |
| s035-region-{region} | ml.m5.xlarge | 10 | 2–3 min | 5 |
| s035-fast-validation | ml.m5.large | 10 | 2–3 min | 1 |

### Phase 11: Verdicts (Jun 6–Jun 7)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-verdicts | ml.m5.large | 10 | 2–3 min | 2 |

### Visualization / Rendering (Jun 3)

| Job Pattern | Instance Type | Volume (GB) | Typical Duration | Count (completed) |
|-------------|--------------|-------------|------------------|-------------------|
| s035-render-oahu-* | ml.m5.xlarge | 10 | 2–5 min | ~5 |

## Total Compute

- **Total GPU hours:** 0 (all CPU-only)
- **Total CPU hours (estimated):** ~12–15 hours wall-clock across all completed jobs
  - ml.m5.xlarge: ~100 completed jobs x ~3 min avg = ~5 hours (20 vCPU-hours)
  - ml.m5.large: ~35 completed jobs x ~3 min avg = ~1.75 hours (3.5 vCPU-hours)
  - Data fetch (large volume): ~30 completed jobs x ~5 min avg = ~2.5 hours (10 vCPU-hours)
  - Longer jobs (NLCD fetch, ZCTA): ~1.5 hours (6 vCPU-hours)
- **Total vCPU-hours:** ~40
- **Estimated cost:** ~$15–20 (ml.m5.xlarge at $0.23/hr, ml.m5.large at $0.115/hr)
- **Region:** us-east-1
- **Account:** 865679935554 (nsc-swarm)

## Software Environment

- **Container image:** `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker`
- **Python:** 3.11
- **Base framework:** PyTorch 2.5.1 (CPU-only; used as base image for pip ecosystem)
- **Key packages (from requirements.txt):**
  - scikit-learn >= 1.4
  - numpy >= 1.24
  - pandas >= 2.0
  - geopandas >= 1.1
  - pyarrow >= 14.0
  - scipy >= 1.11
  - boto3 >= 1.28
- **Vendored wheels (from S3):**
  - yrsn (kappa spatial, certificates, DGM routing) — `s3://swarm-yrsn-datasets/rsct_code/wheels/20260506-162534/`
  - floodcaster (NSI fetch, Hazus, ZCTA aggregation) — installed via bootstrap

## Reproducibility

- All jobs record `S035_GIT_HASH` in environment variables
- Key git hashes observed across runs:
  - `f02cd93` (Jun 7 final region-all)
  - `c2764c8` (Jun 7 verdicts)
  - `0da21f8` (Jun 7 fast-validation)
  - `e555d37` (Jun 6–7 ZCTA jobs)
  - `5b7f50e` (Jun 6 fig5/fig6)
  - `6dc9a82` (Jun 6 quality scoring)
- `S035_GIT_DIRTY=true` on most jobs (dirty working tree at launch time)
- Code uploaded per-job to: `s3://swarm-floodrsct-data/code/s035/{job-name}/src/`

## Notes

- No SageMaker Training Jobs were used; all computation ran as Processing Jobs
- Individual job durations are short (2–13 min) because the workload is embarrassingly parallel across regions
- The ml.m5.xlarge (4 vCPU, 16 GiB) was sufficient for all model training; no memory-bound workloads required larger instances
- VLM quality scoring used ml.m5.large since it only orchestrates API calls to external LLM providers (GPT-4o, Gemini Flash, Jina, Nova, Qwen)
- Many failed jobs represent iteration on pipeline bugs, not wasted compute (most failed within 2–3 min of startup)
