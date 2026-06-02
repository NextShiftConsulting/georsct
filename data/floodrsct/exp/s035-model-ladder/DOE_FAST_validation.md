# DOE: FEMA FAST — Engineering Model Features + External Validation

**Experiment:** s035-model-ladder / Phase 2.5 (features) + Phase 7 (validation)
**Role:** Dual — engineering features for R1.5 AND independent external validation
**Status:** DESIGNED (DOE phase, not launched)
**Depends on:** NSI 2.0 on S3 (data team), FloodSimBench + SLOSH already on S3

---

## Hypothesis

**H6 (exploratory):** Model-ladder predictions at R2 correlate more strongly
with FAST engineering estimates than R0 predictions do, validating that
representation upgrades capture physical flood damage signal.

---

## What FAST Is

FEMA Flood Assessment Structure Tool — Hazus depth-damage functions applied
per structure to compute economic losses:
- Input: building inventory (NSI 2.0) + flood depth raster (.tif)
- Process: Hazus depth-damage functions per structure
- Output: per-structure economic loss ($), damage state, debris
- Speed: ~10,000 structures/second

### Implementation: floodcaster + sphere (NOT HazPy)

The Hazus engine is already implemented in the existing stack:

| Package | Role | Key Classes |
|---------|------|-------------|
| **sphere-flood** | Hazus depth-damage engine | `HazusFloodAnalysis`, `DefaultFloodVulnerability`, `SingleValueRaster` |
| **sphere-core** | Building schema | `FastBuildings`, `NsiBuildings` |
| **floodcaster** | Public API wrapping sphere | `run_flood_analysis()`, `fetch_overture_buildings()` |

s035 does NOT use HazPy directly. It calls floodcaster, which calls sphere.

---

## Scenario Scoping (verified against S3, 2026-06-01)

| Scenario | Depth Source | S3 Location | Coverage | Tier |
|----------|-------------|-------------|----------|------|
| **Houston** | FloodSimBench (7 tiles x 10 return periods) | `raw/floodsimbench/6hr_max/HOU00{1-7}_*_MaxDepth.tif` | Full metro, physics-based 2D sim | **Primary** |
| **NYC** | FloodSimBench (2 tiles x 10 return periods) | `raw/floodsimbench/6hr_max/NYC00{1-2}_*_MaxDepth.tif` | Manhattan only | **Primary** (partial) |
| **SW Florida** | SLOSH MOM Cat 1-5 | `raw/noaa_slosh/mom_national/us_Category{1-5}_MOM_Inundation_HIGH.tif` | Coastal surge only, no pluvial | **Stretch** |
| **Riverside** | None | N/A | No depth data | **Excluded** |

**Return periods available (FloodSimBench):**
1-yr (48mm), 2-yr (57mm), 5-yr (70mm), 10-yr (82mm), 25-yr (98mm),
50-yr (110mm), 100-yr (123mm), 200-yr (138mm), 500-yr (162mm), 1000-yr (181mm)

---

## Data Prerequisites

### Available on S3

| Data | S3 Key | Size | Status |
|------|--------|------|--------|
| FloodSimBench depth grids | `raw/floodsimbench/6hr_max/*.tif` | 3 GB (78 files) | READY |
| SLOSH MOM inundation | `raw/noaa_slosh/mom_national/*.tif` | 5.5 GB (5 cats) | READY |
| 3DEP DEM (Houston) | `raw/dem/3dep/v1/houston/*.tif` | ~4.5 GB (14 tiles) | READY |
| FEMA flood zones | `raw/geocertdb2026/flood_zones_zcta.parquet` | 0.4 MB | READY |

### NOT on S3 — Data Team Must Fetch

| Data | S3 Target | Source | Size | Priority |
|------|-----------|--------|------|----------|
| NSI 2.0 structures — Houston | `raw/nsi/v2/houston_structures.parquet` | USACE NSI API | ~50 MB | **Critical** |
| NSI 2.0 structures — NYC | `raw/nsi/v2/nyc_structures.parquet` | USACE NSI API | ~80 MB | **Critical** |
| NSI 2.0 structures — SW Florida | `raw/nsi/v2/southwest_florida_structures.parquet` | USACE NSI API | ~40 MB | Stretch |

NSI 2.0 required fields: `fd_id, x, y, occtype, sqft, found_type,
num_story, found_ht, val_struct, val_cont, med_yr_blt`

---

## Architecture Boundary: floodcaster vs s035

Generic flood analysis capabilities belong in `floodcaster` (open-source).
Experiment-specific logic stays in s035.

### floodcaster owns (reusable, open-source)

| Capability | Current State | What's Needed |
|------------|--------------|---------------|
| **Overture building fetch** (bbox) | DONE — `sources.py:fetch_overture_buildings()` | Nothing |
| **Hazus damage computation** | DONE — `analysis.py:run_flood_analysis()` via sphere | Nothing |
| **NSI 2.0 building fetch** (bbox or ZCTA) | NOT IN FLOODCASTER — exists as standalone `fetch_nsi_structures.py` in rsct-geocert | Migrate to `floodcaster/nsi_sources.py` |
| **NSI→sphere schema mapping** | NOT IN FLOODCASTER — sphere has `NsiBuildings` | Wire `NsiBuildings` into floodcaster as alternative to Overture |
| **ZCTA-level aggregation** | NOT IN FLOODCASTER | New `floodcaster/aggregation.py` — spatial join + group stats |
| **run_nsi_flood_analysis()** | NOT IN FLOODCASTER | New function: NSI buildings + depth raster -> per-building losses (mirrors `run_flood_analysis()`) |

### s035 owns (experiment-specific)

| Script | What It Does | Calls |
|--------|-------------|-------|
| `run_fast_zcta.py` | Runs FAST for s035 scenarios, extracts 6 ZCTA features | `floodcaster.analysis.run_nsi_flood_analysis()` + `floodcaster.aggregation.aggregate_by_zcta()` |
| `compute_fast_validation.py` | Spearman correlations across levels (H6) | Reads ZCTA aggregates from S3 |
| `train_r1_5_fast.py` | Trains with FAST features (conditional on NSI) | Reads ZCTA features from `run_fast_zcta` output |

### Data flow

```
floodcaster (reusable)                     s035 (experiment)
─────────────────────                      ─────────────────
nsi_sources.fetch_nsi()                    run_fast_zcta.py
  -> NsiBuildings (sphere)                   selects scenarios + return periods
    -> HazusFloodAnalysis (sphere)           calls floodcaster for each
      -> per-building losses                 extracts 6 ZCTA features
        -> aggregate_by_zcta()               uploads to S3
          -> ZCTA-level parquet
                                           compute_fast_validation.py
                                             reads ZCTA parquets + s035 predictions
                                             computes Spearman correlations (H6)
```

### What migrates from rsct-geocert to floodcaster

`data/floodrsct/jobs/fetch_nsi_structures.py` — NSI API fetch logic moves
to `floodcaster/nsi_sources.py`. The s035 script becomes a thin caller that
specifies which county FIPS codes to fetch and where to upload.

---

## Part A: FAST as Feature Source (R1.5)

### FAST-Derived ZCTA Features (6 columns)

Run FAST per structure, aggregate to ZCTA:

| Column | Aggregation | Signal |
|--------|-------------|--------|
| fast_mean_loss_per_sqft | Mean(structure_loss / sqft) per ZCTA | Average damage intensity |
| fast_max_loss_usd | Max(structure_loss) per ZCTA | Worst-case structure |
| fast_pct_damaged | Count(loss > 0) / Count(all) per ZCTA | Damage penetration rate |
| fast_total_loss_usd | Sum(structure_loss) per ZCTA | Total ZCTA exposure |
| fast_median_depth_ft | Median(flood_depth_at_structure) per ZCTA | Typical inundation |
| fast_n_structures | Count(structures) per ZCTA | Building density |

### Where R1.5 Sits

```
R0: Static tabular (36)
R1: + Hydrology + W-matrix (61-63)
R1.5: + FAST engineering (67-69)   <-- Optional, conditional on NSI
R2: + Temporal dynamics (76-78)
```

R1.5 is CONDITIONAL — only runs if NSI 2.0 is on S3 before launch. If not
ready, skip R1.5 and use FAST only for external validation (Part B).

### FloodSimBench Return Period Selection

For features, use the 100-yr return period (123mm/6hr) as the representative
depth grid. Rationale:
- 100-yr is the FEMA regulatory standard (1% annual chance)
- Matches the BFE (Base Flood Elevation) concept
- Consistent across Houston and NYC tiles

For validation (Part B), test across multiple return periods.

---

## Part B: FAST as External Validation Target (Phase 7)

### Validation Protocol

For each (scenario, return_period) where FAST outputs exist:

1. Run FAST → per-structure losses → aggregate to ZCTA → `fast_total_loss_zcta`
2. Compare s035 predictions at each level against FAST:
   - Spearman rho(pred_R0, fast_total_loss_zcta)
   - Spearman rho(pred_R1, fast_total_loss_zcta)
   - Spearman rho(pred_R2, fast_total_loss_zcta)
3. Compare observed NFIP claims against FAST:
   - Spearman rho(nfip_obs, fast_total_loss_zcta) — ceiling

### Validation Table (Paper Table 5)

```
scenario | return_period | n_zctas | rho(NFIP_obs,FAST) | rho(R0,FAST) | rho(R1,FAST) | rho(R2,FAST) | interpretation
```

### Expected Patterns

- rho(NFIP_obs, FAST) is the ceiling — NFIP and FAST measure different things
- rho(pred, FAST) increases R0→R1→R2 → ladder captures damage signal
- If rho(R2, FAST) > rho(NFIP_obs, FAST) → model generalizes beyond
  insurance filing patterns (strong result)
- If rho(pred, FAST) is flat across levels → engineering damage signal is
  already captured at R0; representation upgrades add non-FAST signal

### Multiple Return Period Robustness

Run validation at 10-yr, 50-yr, 100-yr, 500-yr return periods. If the
rho(pred, FAST) ranking across levels is consistent across return periods,
the finding is robust to severity assumptions.

---

## Outputs

| Artifact | S3 Key | Phase |
|----------|--------|-------|
| FAST ZCTA aggregates | `processed/{scenario}/{scenario}_fast_zcta.parquet` | 7a |
| Validation results | `results/s035/fast_validation.json` | 7b |
| R1.5 results (conditional) | `results/s035/r1.5_{scenario}.json` | 2.5 |

---

## Success Criteria

| Criterion | Threshold | Action if FAIL |
|-----------|-----------|----------------|
| rho(R2, FAST) > rho(R0, FAST) | Positive delta | Report as null — ladder doesn't improve FAST correlation |
| rho(NFIP_obs, FAST) > 0.3 | Spearman > 0.3 | If very low, NFIP and FAST measure fundamentally different things — validation inapplicable |
| Consistent across return periods | Same level ranking | If inconsistent, report sensitivity to severity |

---

## Kill Rules

- FAST outputs all zeros (no damage) for all ZCTAs → depth raster doesn't
  overlap with NSI structures, check spatial alignment
- rho(NFIP_obs, FAST) < 0 → NFIP claims anticorrelate with engineering
  damage, validation framework is invalid for this scenario
- NSI 2.0 not on S3 by launch date → skip R1.5, run Phase 7 only if NSI
  arrives before paper deadline

---

## Compute

| Resource | Value |
|----------|-------|
| Instance | ml.m5.xlarge (FAST is CPU-bound) |
| Est. duration | ~10 min per scenario per return period |
| Dependencies | NSI 2.0 on S3, floodcaster package (wraps sphere-flood Hazus engine) |
| GPU | NOT NEEDED |

---

## DO NOT Constraints

- Do NOT run FAST without NSI building inventory (it needs structures)
- Do NOT use FAST outputs as training target (it's validation only, or features)
- Do NOT cherry-pick return periods — report all that were run
- Do NOT change FAST damage functions (use Hazus defaults via sphere)
- Do NOT reimplement Hazus in s035 — call floodcaster, which calls sphere
- Do NOT reimplement NSI fetch in s035 — use floodcaster.nsi_sources
- Do NOT reimplement ZCTA aggregation in s035 — use floodcaster.aggregation
