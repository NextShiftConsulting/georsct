# Prithvi-EO-2.0 Satellite Embedding Extraction

**Date:** 2026-06-26
**Phase:** prithvi_embeddings (s035-model-ladder phase, not a standalone series)
**Commit:** 029b59a (v2 fixes), df3ede8 (status registration)

## Summary

Per-ZCTA satellite imagery embeddings extracted from IBM/NASA Prithvi-EO-2.0
(300M-param ViT) using Harmonized Landsat Sentinel-2 (HLS) data. 1024-dim
embeddings for all 5 flood scenarios.

## Key Design Decisions

### 1. Mean-pooled patch tokens, not CLS

v1 used CLS token (position 0 of ViT output). Quality analysis revealed
CLS is dominated by architectural bias:

- **v1 pairwise cosine:** min=0.992, mean=0.999 (Houston, 211 HLS pairs)
- All 212 embeddings nearly identical -- no spatial discrimination

v2 uses mean-pooled patch tokens (`x[:, 1:, :].mean(dim=1)`), which
capture spatial land-cover variation:

- **v2 pairwise cosine:** min=0.414 (Riverside), mean=0.932--0.967
- Norm std 0.51--1.06 (was 0.02 in v1)

**Paper implication:** When citing Prithvi embedding quality, note that
CLS tokens from foundation model ViTs are near-degenerate for geospatial
discrimination. Mean-pooling patch tokens is required.

### 2. NaN fallback instead of zero-chip encoding

v1 fed zero chips through the encoder, producing valid-looking embeddings
(norm ~23.7, cosine 0.987--0.993 with real HLS). These were invisible
poison -- downstream models could not distinguish fallback from real data.

v2 sets fallback rows to NaN. Downstream phases must filter or impute
explicitly. The `source` column marks each row as `hls` or `fallback_no_data`.

### 3. CMR retry with backoff

v1 had single 30s timeout on NASA CMR API queries. Transient failures
caused New Orleans to drop to 50% coverage.

v2 retries 3 times with 5/15/30s backoff. Coverage improvement:

| Scenario | v1 | v2 | Delta |
|----------|----|----|-------|
| New Orleans | 50.0% | 87.1% | +37.1pp |
| Riverside | 65.9% | 90.6% | +24.7pp |
| SW Florida | 73.0% | 96.2% | +23.2pp |
| Houston | 99.5% | 95.8% | -3.7pp* |
| NYC | 100% | 95.6% | -4.4pp* |

*Houston/NYC slight decrease due to chip quality gate rejecting >50%
nodata chips that v1 accepted as zeros.

## v2 Quality Metrics

| Scenario | ZCTAs | HLS | Fallback | Coverage | Cosine min | Cosine mean | Norm std |
|----------|-------|-----|----------|----------|-----------|------------|---------|
| Houston | 212 | 203 | 9 | 95.8% | 0.626 | 0.962 | 0.66 |
| NYC | 181 | 173 | 8 | 95.6% | 0.827 | 0.955 | 1.06 |
| New Orleans | 62 | 54 | 8 | 87.1% | 0.498 | 0.961 | 0.51 |
| Riverside | 85 | 77 | 8 | 90.6% | 0.414 | 0.933 | 0.80 |
| SW Florida | 211 | 203 | 8 | 96.2% | 0.764 | 0.967 | 0.80 |

All scenarios pass the 60% minimum HLS coverage quality gate.

## S3 Artifacts

```
s3://swarm-floodrsct-data/results/s035/prithvi_embeddings/
  {scenario}_prithvi_embeddings.parquet   # (zcta, prithvi_emb_0..1023, source)
  {scenario}_prithvi_meta.json            # coverage, quality stats, HLS provenance
```

## Data Contract

Added to `EXPERIMENT_CONTRACT.yaml`:
- `phase_id: prithvi_embeddings`
- `depends_on: [event_building]`
- `quality_gates: {min_hls_coverage_pct: 60, embed_dim: 1024, pooling: mean_patch, fallback_value: NaN}`

## Paper-Extractable Claims

1. **Prithvi-EO-2.0 CLS tokens are near-degenerate for geospatial tasks.**
   Pairwise cosine similarity >0.99 across all ZCTA pairs within a metro.
   Mean-pooling patch tokens reduces mean cosine to 0.93--0.97 with spread
   down to 0.41, indicating genuine spatial discrimination.

2. **CMR retry recovers 23--37 percentage points of HLS coverage** on
   scenarios affected by transient NASA API timeouts.

3. **Zero-chip fallback produces indistinguishable embeddings.** L2 norm
   and cosine similarity of zero-chip-through-ViT embeddings overlap with
   real HLS embeddings (cosine 0.987--0.993). NaN marking is required.

4. **Coverage ranges 87--96% across 5 US metros** with max cloud cover
   30%, temporal window 2023--2024, using HLS Sentinel-2 (preferred) with
   Landsat fallback.

## Downstream Dependencies

These embeddings feed into the R4.5 representation arm if activated.
Downstream consumers must:
- Filter rows where `source == "fallback_no_data"` (embeddings are NaN)
- Or impute using spatial neighbors (not recommended without validation)

## SageMaker Job IDs

- `s035-prithvi-embed-houston-20260626-173356`
- `s035-prithvi-embed-nyc-20260626-173403`
- `s035-prithvi-embed-new-orleans-20260626-173410`
- `s035-prithvi-embed-riverside-coachella-20260626-173418`
- `s035-prithvi-embed-southwest-florida-20260626-173425`
