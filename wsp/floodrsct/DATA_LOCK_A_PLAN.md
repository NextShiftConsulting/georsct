# Data Lock A Remediation Plan

**Data Lock A target**: June 1, 2026 (Houston scenario)
**Data Lock B target**: June 2, 2026 (all 5 scenarios)
**SIGSPATIAL abstract**: June 5, 2026

## Status as of May 29, 2026

### Bucket: `s3://swarm-floodrsct-data` (43.6 GB, 503 objects)

| Dataset | Status | Files | Size | Problem |
|---------|--------|-------|------|---------|
| HRRR QPF (6 storms) | REAL | 175 | 21.4 GB | None |
| USGS 3DEP DEM | REAL | 63 | ~1 GB | None |
| Prithvi-EO-2.0 | REAL | 14 | 1.2 GB | None (smoke PASS) |
| TerraMind-base-Flood | REAL | 6 | 642 MB | None (smoke PASS) |
| FloodSimBench | REAL | 81 | varies | None (78/78 tiles) |
| ImpactMesh-Flood | REAL | 11 | varies | Masks only (by design) |
| USGS STN HWMs | REAL | 2 | ~58 KB | None |
| Houston 311 | REAL | 1 | varies | None |
| MTA stations | REAL | 1 | varies | None |
| MTBS burn perimeters | REAL | 1 | varies | None |
| NHDPlus catchments | REAL | 2 | varies | None |
| **MRMS Stage IV** | **BROKEN** | 162 | 114 KB | **720-byte stubs** |
| **NLCD impervious** | **EMPTY** | 0 | 0 | Fetcher failed silently |
| **Sen1Floods11** | METADATA | 4 | ~20 KB | By design (35 GB imagery skipped) |
| **NOAA tidal** | **MISSING** | 0 | 0 | No fetcher exists |
| **NHC SLOSH surge** | **MISSING** | 0 | 0 | No fetcher exists |
| **HURDAT2 tracks** | **MISSING** | 0 | 0 | No fetcher exists |

## Three-Day Plan

### Day 1: May 29 -- Write and Launch Fetchers

**All scripts written locally, all processing on AWS SageMaker.**

#### 1a. Fix MRMS Stage IV fetcher (CRITICAL)
- **Root cause**: `fetch_noaa_mrms.py` tries NOMADS then EMC URLs. Both return
  small error/redirect pages with HTTP 200 for historical data. No file-size
  validation, so 720-byte responses saved as-is.
- **Fix**: Switch primary source to Iowa State Mesonet archive
  (`mtarchive.geol.iastate.edu/YYYY/MM/DD/mrms/ncep/Stage4/`). Add minimum
  file size check (GRIB2 must be > 10 KB). Keep EMC as fallback.
- **Launch**: 6 SageMaker jobs (one per storm: harvey2017, beryl2024, hilary2023,
  ian2022, ida2021_nyc, imelda2019). Must first delete the 720-byte stubs.
- **Instance**: ml.m5.xlarge (I/O bound)
- **Expected**: ~162 real GRIB2 files, ~500 MB total

#### 1b. Write NOAA tidal predictions fetcher (CRITICAL)
- **Source**: CO-OPS API (`api.tidesandcurrents.noaa.gov/api/datagetter`)
- **Product**: Hourly verified water levels + predictions
- **Stations**: 
  - Houston: 8771450 (Galveston), 8770570 (Sabine Pass), 8771013 (Eagle Point)
  - NYC: 8518750 (The Battery), 8531680 (Sandy Hook)
  - SW FL: 8725520 (Fort Myers), 8726520 (Naples)
  - NO: 8761724 (Grand Isle), 8761927 (New Canal)
- **Format**: JSON -> parquet (one file per station-event)
- **Instance**: ml.m5.large (small downloads)

#### 1c. Write NHC SLOSH/surge fetcher (CRITICAL for Houston)
- **Approach**: Derive storm surge from tidal residuals (observed - predicted).
  SLOSH MOM grids are not freely downloadable in bulk.
- **Alternative**: Use USGS STN high-water marks + DEM to compute inundation
  depth per ZCTA. This is ground-truth surge, not modeled.
- **Output**: Per-ZCTA surge estimates as parquet
- **Instance**: ml.m5.large

#### 1d. Fix NLCD impervious surface fetcher (MEDIUM)
- **Source**: MRLC NLCD 2021 impervious surface
- **Fix**: Debug URL pattern for CONUS NLCD tiled download
- **Instance**: ml.m5.xlarge (raster data)

#### 1e. Write HURDAT2 track fetcher (LOW priority)
- **Source**: NHC HURDAT2 CSV (single file, ~3 MB)
- **URL**: `https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt`
- **Instance**: ml.m5.large (trivial)

### Day 2: May 30 -- Validate and Relaunch Failures

- Monitor all Day 1 SageMaker jobs via CloudWatch logs
- Verify MRMS files: size > 10 KB, openable with cfgrib
- Verify tidal data: complete hourly coverage for each event window
- Relaunch any failed jobs with fixes
- Launch MRMS for missing storms (Ian, Beryl, Ida if not covered)

### Day 3: May 31 -- Data Lock A Prep

- Run manifest reconciliation script
- Build `houston_event_dataset.parquet` from raw layers
- Sanity checks:
  - Spatial coverage: all Houston ZCTAs have data
  - Temporal alignment: HRRR, MRMS, tidal cover same event windows
  - No NaN gaps in critical columns
  - File integrity: all parquets readable, all GRIB2 valid
- Tag Data Lock A in S3 (copy manifests to `locks/data_lock_a/`)

## File Structure

```
data/floodrsct/s035/
  _manifest_writer.py       # Shared (copied from existing)
  _s3_stream.py             # Shared (copied from existing)
  fetch_noaa_mrms_v2.py     # Fixed MRMS fetcher
  fetch_noaa_tides.py       # New tidal fetcher
  fetch_surge_hwm.py        # Surge from STN HWMs
  fetch_nlcd_impervious.py  # Fixed NLCD fetcher
  fetch_hurdat2.py          # HURDAT2 track data
  entrypoint_mrms.sh        # SageMaker entrypoints
  entrypoint_tides.sh
  entrypoint_surge.sh
  entrypoint_nlcd.sh
  entrypoint_hurdat2.sh
  sagemaker_launch_all.py   # Launcher for all jobs
  validate_data_lock_a.py   # Data Lock A reconciliation
```

## SageMaker Constants

| Parameter | Value |
|-----------|-------|
| Bucket | `swarm-floodrsct-data` |
| Code prefix | `code/s035/` |
| IAM Role | `arn:aws:iam::865679935554:role/SageMakerExecutionRole` |
| Profile | `nsc-swarm` |
| Region | `us-east-1` |
| Image | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5-cpu-py311` |
| Instance (fetch) | `ml.m5.xlarge` |
| Volume | 30 GB |
| Max runtime | 43200 s |
