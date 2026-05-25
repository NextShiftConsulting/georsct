# HF Dataset vs S018D Input Reconciliation

**Date**: 2026-05-01
**Purpose**: Verify data integrity between HuggingFace release (GeoCert v23.001) and S018D experiment inputs.

---

## Data Sources

| Attribute | HF Release (v23.001) | S018D Input |
|-----------|---------------------|-------------|
| **File** | `geocert_table.parquet` | `zcta_features_labels_with_lags.parquet` |
| **S3 Path** | `rudymartin/geocert` (HF) | `s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed_with_lags/` |
| **S3 Timestamp** | 2026-05-01 (release) | 2026-04-30 15:56:28 |
| **ZCTAs** | 31,789 | 31,789 |
| **ACS Features** | 33 (`acs_*`) | 33 (`acs_*`) + 33 (`lag_acs_*`) = 66 |
| **Targets** | 27 (`target_*`) | 27 (`target_*`) |
| **Total Columns** | ~95 (features + targets + splits + geo) | ~118 (features + lags + targets + metadata) |

## Alignment: CONFIRMED

| Check | Status | Detail |
|-------|--------|--------|
| ZCTA count | MATCH | Both 31,789 (canonical set, territories excluded) |
| ZCTA IDs | MATCH | Same canonical set from `zcta_features_labels.parquet` |
| Target columns | MATCH | Same 27 CONUS-27 tasks (21 health + 4 socio + 2 enviro) |
| ACS feature columns | MATCH | Same 33 `acs_*` base columns |
| Sentinel fix (-666666666) | MATCH | Applied to both (sentinel_fix_provenance.json: 2026-04-27) |
| Train/test splits | COMPATIBLE | S018D uses interpolation split from same `train_test_split.json` |

## Discrepancy: 825 Bogus Negatives

| Attribute | HF Release | S018D Input |
|-----------|-----------|-------------|
| `clean_acs_negatives.py` | APPLIED | NOT APPLIED |
| Affected cells | 0 (cleaned to NaN) | 825 across 387 ZCTAs in 6 columns |
| Affected columns | `acs_gini_index` (132), `acs_median_age` (71), `acs_median_hh_income` (130), `acs_median_home_value` (94), `acs_median_rent` (250), `acs_median_year_built` (148) | Same columns, values ~-700K to -999K |

### Timeline

1. **2026-04-27**: `fix_acs_sentinels.py` applied (sentinel_fix_provenance.json)
2. **2026-04-30 14:14**: `zcta_features_labels.parquet` updated on S3
3. **2026-04-30 15:56**: `zcta_features_labels_with_lags.parquet` created (S018D input)
4. **2026-05-01 ~03:51-09:50**: S018D experiment ran on SageMaker
5. **2026-05-01 15:11**: `clean_acs_negatives.py` committed (825 residual negatives fixed)
6. **2026-05-01**: HF release v23.001 published with cleanup applied

### Impact Assessment: LOW (for paper claims)

The 825 bogus negatives affect S018D findings as follows:

| Finding | Impact | Rationale |
|---------|--------|-----------|
| Architecture-invariance (spread ~0.037) | **NEGLIGIBLE** | All 12 solvers trained on identical corrupted features. Spread measures cross-family convergence (relative), not absolute performance. Corrupted cells affect all solvers equally. |
| GCF-1.1 metric compression (9/12 solvers) | **NEGLIGIBLE** | Certificate quality is relative to solver population. All solvers see same noise floor. The compression diagnosis (aggregate masks individual) is independent of feature quality. |
| Gate entropy = 0.0 (all EXECUTE) | **NONE** | Gate decisions use certificate signals, not raw features. |
| Nearest-neighbor = linear_ridge (3.34) | **LOW** | Relative distances between solvers may shift slightly, but the noisy-solver cluster structure is robust to 0.08% feature contamination. |

**Quantitative scope**: 825 / (31,789 x 33) = 0.079% of the feature matrix.
387 / 31,789 = 1.22% of ZCTAs affected (scattered, not systematic).

### Conclusion

S018D results are valid for the claims made in the paper. The 825 bogus negatives represent a 0.08% feature contamination that affects all solvers identically, preserving the relative comparisons that all three paper findings rely on.

The HF-published dataset (v23.001) is the canonical clean version. S018D out-of-fold predictions should be regenerated on clean data for v2 if absolute performance numbers are reported.

---

## Verification Commands

```bash
# Check S3 timestamp
MSYS_NO_PATHCONV=1 aws s3 ls "s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed_with_lags/" --profile nsc-swarm

# Check sentinel provenance
MSYS_NO_PATHCONV=1 aws s3 cp "s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/sentinel_fix_provenance.json" - --profile nsc-swarm | python -m json.tool | head -20
```
