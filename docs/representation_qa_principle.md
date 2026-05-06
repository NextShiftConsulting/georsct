# GeoRSCT Representation QA Principle

GeoRSCT requires representation QA wherever geography is transformed into a
model-facing quantity. CRS conversion, polygon overlay, raster aggregation,
spatial lag construction, key-based joins, and spatial splits are not neutral
preprocessing. They are measurement operations. If they are not validated,
downstream model performance cannot be cleanly attributed to solver quality,
target difficulty, or representation-solver compatibility.

## QA Taxonomy

| Area | Urgency | Type of QA |
|------|---------|------------|
| FEMA flood zones | Done | CRS, topology, overlay, area fraction |
| Raster features (elevation, tree cover, night lights) | Urgent | CRS, nodata, zonal stats, distribution |
| Spatial lags | Urgent | Graph alignment, W matrix, leakage |
| Train/test splits | Urgent | Spatial leakage / block validation |
| OSM features | Next | Geometry, missingness, density normalization |
| ACS / CDC joins | Next | Key integrity, vintage, duplicates, leakage |
| N-ceiling | Next | Solver comparability, bootstrap, uncertainty |
| R/S/N certificates | Next | Degeneracy, calibration, semantic label validity |
| Paper evidence tables | Next | Artifact lineage, stale-number audit |

## Priority 1: Raster / Zonal-Stat QA

Applies to: USGS elevation, Hansen tree cover, VIIRS night lights, NOAA layers.

Checks:
- CRS alignment between raster and ZCTA boundaries
- Raster bounds vs ZCTA bounds
- Nodata handling (masked vs zero vs NaN)
- Pixel resolution / resampling method
- Zonal-stat method: mean, median, max, area-weighted
- Coastal / partial-pixel behavior
- Distribution sanity by known regions

Risk: Bad raster aggregation corrupts representation. Model appears weak or
strong for the wrong reason.

## Priority 2: Spatial Lag / W-Matrix QA

Applies to: all lag_acs_* columns, adjacency matrix.

Checks:
- W matrix row count = ZCTA row count
- Row order alignment between W and feature table
- No accidental target lags
- No unintended self-loops
- Row-standardization if expected
- Track islands / disconnected ZCTAs
- Validate neighbor examples manually
- Compare lag feature distribution to base feature distribution

Risk: Bad spatial lag creates fake signal or leakage. The solver is given the
wrong neighborhood.

RSCT note: Spatial lag is not a neutral feature. It is a graph morphism from
local geography into neighborhood context. If W is wrong, the solver is being
given the wrong neighborhood.

## Priority 3: Spatial Leakage / Split QA

Applies to: split_imputation, split_extrap, split_superres.

Checks:
- Random split vs spatial block split comparison
- Neighbor overlap between train/test
- State/county leakage checks
- Moran's I or residual spatial autocorrelation
- Performance drop under blocked splits
- Duplicate or near-duplicate ZCTAs across folds

Risk: A random split can make geography look easier than it is. The solver may
be borrowing nearby geography rather than generalizing.

## Additional QA Areas

### CDC PLACES / ACS Joins
- ZCTA key formatting (leading zeros)
- One row per ZCTA
- Join coverage rate
- Duplicate keys
- Vintage mismatch
- Suppression / missing value handling
- Target leakage from closely related variables

### N-Ceiling Construction
- All solver families use comparable train/test folds
- max_k computed only on valid runs
- Bootstrap confidence intervals
- One unstable solver dominating max_k
- Separate target difficulty from failed solver coverage

### Certificate / R-S-N Decomposition
- Confirm R/S/N labels mean what we think they mean
- Check simplex distribution
- Check class imbalance
- Check whether "balanced" is becoming a garbage bin
- Check alpha/kappa/sigma ranges
- Check degenerate near-uniform outputs
- Check trace consistency

## Implementation Pattern

The FEMA flood zone validation established the pattern:
1. Validation runs before the feature is trusted
2. Critical checks are fatal (abort run)
3. Non-fatal checks are logged as warnings
4. All results saved to validation_status.json artifact
5. Artifact uploaded to S3 for audit trail

Apply this same pattern to each QA area above.
