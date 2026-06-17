# DOE-C3: Census Tract Resolution — Gate 3B Activation

**Experiment:** s035-model-ladder / DOE-C3
**Role:** Rebuild data pipeline at census tract resolution to activate kappa_reconstruct
**Status:** DESIGNED
**Depends on:** DOE-C1 (COMPLETED — established that kappa_reconstruct = 1.0 at ZCTA)
**Blocks:** DOE-C4 (morph dispatch needs discriminative Gate 3B)

---

## Motivation

DOE-C1 found kappa_reconstruct = 1.0 for all constructs at ZCTA resolution
(~900 ZCTAs per scenario). The Gabriel graph at this resolution is too coarse
for topology crossings to occur: the representation-implied neighborhood
always matches the geographic neighborhood. This is a documented limitation,
not a bug, but it means Gate 3B cannot discriminate between constructs.

Census tracts (~74,000 nationally, ~2,000-5,000 per metro) provide 3-5x
finer spatial resolution. At tract scale, representation-implied neighborhoods
should diverge from geographic neighborhoods for constructs that encode
institutional rather than physical structure (FEMA, NFIP).

---

## Hypothesis

**H-C3:** At census tract resolution, kappa_reconstruct < 1.0 for at least
one construct in at least two scenarios, and the construct ordering by
kappa_reconstruct differs from the ordering at ZCTA resolution.

**Null:** kappa_reconstruct remains 1.0 at tract resolution (spatial
structure is always preserved regardless of resolution).

**Diagnostic:** If kappa_reconstruct drops for FEMA/NFIP but stays high
for JRC/Deltares, that confirms the thesis: physical constructs preserve
topology, institutional constructs do not.

---

## Design Matrix

| Factor | Levels | Type |
|--------|--------|------|
| Scenario | houston, new_orleans | Fixed (2 — focus scenarios) |
| Construct | JRC, Deltares, FEMA, NFIP | Fixed (4 — FAST excluded: no tract-level NSI) |
| Resolution | ZCTA (control), census_tract (treatment) | Treatment (2 levels) |

**Total cells:** 2 scenarios x 4 constructs x 2 resolutions = 16 certifications

---

## Data Pipeline Requirements

### New data needed (not on S3):

1. **Census tract boundaries** -> tract_id assignment for each (lat, lon)
   - Source: Census Bureau TIGER/Line shapefiles
   - Processing: spatial join of event records to tract polygons

2. **Tract-level feature aggregation**
   - ACS features at tract level (more granular than ZCTA)
   - Source: Census API / ACS 5-year estimates
   - Key: census tract FIPS code

3. **Tract-level adjacency matrix**
   - Queen contiguity from tract polygons
   - Source: PySAL from TIGER/Line

4. **Tract-level construct targets**
   - JRC: aggregate 30m pixels to tract (mean occurrence)
   - Deltares: aggregate depth to tract (mean depth at RP)
   - FEMA: % of tract area in SFHA
   - NFIP: claims per tract per event (geocoded via property address)

### Existing data (reusable):
- JRC/Deltares rasters (just need new aggregation boundary)
- FEMA NFHL polygons (just need tract intersection)
- NFIP claims (need geocoding to tract — address-based)

---

## Method

1. Build tract-level event_features for houston and new_orleans.
2. Build tract-level folds (spatial-blocked, 5-fold).
3. Build tract-level adjacency CSR.
4. Run DOE-C1 protocol at tract resolution.
5. Compare kappa_reconstruct(tract) vs kappa_reconstruct(ZCTA) per construct.

---

## Acceptance Criteria

| ID | Criterion | Test |
|----|-----------|------|
| AC-C3-1 | kappa_reconstruct < 1.0 for at least one construct at tract | kr_tract < 0.95 |
| AC-C3-2 | FEMA/NFIP have lower kr than JRC/Deltares at tract | ordering test |
| AC-C3-3 | Tract-level forward_score is within 0.1 of ZCTA-level | resolution does not destroy signal |
| AC-C3-4 | Tract adjacency matrix has > 500 edges per scenario | sufficient connectivity |

---

## S3 Output Convention

```
results/s035/doe_c3/
  tract_resolution_{scenario}.json
  cache/
    certificates_tract_{scenario}.parquet
    certificates_zcta_{scenario}.parquet    # re-run for paired comparison
    kappa_resolution_comparison.parquet
processed/shared/
  tract_features_{scenario}.parquet
  tract_adjacency_{scenario}.npz
  tract_folds_{scenario}.parquet
```

---

## Resource Estimate

- Data pipeline: ml.m5.2xlarge (8 vCPU, 32 GB) — spatial joins are memory-heavy
- Certification: ml.m5.xlarge per scenario
- Wall clock: ~4 hours (pipeline) + ~2 hours (certification)
- Cost: ~$3-5

---

## Risk

This is the heaviest data engineering lift of the four experiments.
Census tract geocoding of NFIP claims requires address-to-tract mapping
that may have coverage gaps. Fallback: use tract-level NFIP aggregates
from FEMA's published OpenFEMA API (pre-aggregated by census tract for
recent disasters).
