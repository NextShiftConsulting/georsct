# Series 035 Data Manifest

**Bucket:** `s3://swarm-floodrsct-data/`
**Status key:** `PENDING` | `LAUNCHED` | `READY` | `LOCKED`

---

## Reused from geocertdb2026 (no re-download)

| Dataset | S3 target | Status |
|---------|-----------|--------|
| ZCTA features + labels (ACS, SVI, HIFLD, flood zones, NOAA events, NFIP, TWI) | `raw/geocertdb2026/zcta_features_labels.parquet` | PENDING |
| SVI by ZCTA | `raw/geocertdb2026/svi_zcta.parquet` | PENDING |
| FEMA flood zones by ZCTA | `raw/geocertdb2026/flood_zones_zcta.parquet` | PENDING |
| NOAA storm events by ZCTA | `raw/geocertdb2026/noaa_storm_events_zcta.parquet` | PENDING |
| NFIP claims by ZCTA | `raw/geocertdb2026/nfip_claims_zcta.parquet` | PENDING |
| TWI / watershed features | `raw/geocertdb2026/twi_features_zcta.parquet` | PENDING |

---

## New pulls

### USGS NWIS — gauge timeseries

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Harris County gauges, Harvey 2017 (Aug 17 – Sep 15) | `raw/usgs_nwis/houston_harvey2017.parquet` | PENDING |
| Harris County gauges, Imelda 2019 (Sep 17-21) | `raw/usgs_nwis/houston_imelda2019.parquet` | PENDING |
| Harris County gauges, Beryl 2024 (Jul 8-12) | `raw/usgs_nwis/houston_beryl2024.parquet` | PENDING |
| Lake Pontchartrain / Industrial Canal gauges, Ida 2021 (Aug 29 – Sep 5) | `raw/usgs_nwis/no_ida2021.parquet` | PENDING |
| NYC coastal gauges, Ida 2021 (Sep 1-3) | `raw/usgs_nwis/nyc_ida2021.parquet` | PENDING |
| NYC coastal gauges, Henri 2021 (Aug 21-23) | `raw/usgs_nwis/nyc_henri2021.parquet` | PENDING |

Fields: `site_no, datetime_utc, flow_cfs, stage_ft, quality_code`

### NOAA MRMS Stage IV — gridded precipitation

| Dataset | S3 target | Notes | Status |
|---------|-----------|-------|--------|
| Harvey 2017 (Aug 17 – Sep 3, hourly) | `raw/noaa_mrms/harvey2017/` | ~500 MB, 1-km grid | PENDING |
| Imelda 2019 (Sep 17-21, hourly) | `raw/noaa_mrms/imelda2019/` | ~100 MB | PENDING |
| Beryl 2024 (Jul 8-12, hourly) | `raw/noaa_mrms/beryl2024/` | ~100 MB | PENDING |
| Ida 2021 NYC (Sep 1-3, hourly) | `raw/noaa_mrms/ida2021_nyc/` | ~100 MB | PENDING |

Format: grib2 files preserved as-is; ZCTA aggregation in build_event_dataset job.

### NLCD 2021 — Impervious Surface (Houston S1, NYC S5)

| Dataset | S3 target | Notes | Status |
|---------|-----------|-------|--------|
| TX state GeoTIFF | `raw/nlcd/impervious/v2021/48.tif` | 30m | PENDING |
| LA state GeoTIFF | `raw/nlcd/impervious/v2021/22.tif` | 30m | PENDING |
| NY state GeoTIFF | `raw/nlcd/impervious/v2021/36.tif` | 30m | PENDING |
| CA state GeoTIFF | `raw/nlcd/impervious/v2021/06.tif` | 30m | PENDING |
| FL state GeoTIFF | `raw/nlcd/impervious/v2021/12.tif` | 30m | PENDING |
| Manifest | `manifests/nlcd_impervious/v2021/manifest.json` | | PENDING |

Source: MRLC WCS `www.mrlc.gov/geoserver/mrlc_display/wcs`. ZCTA aggregation via `build_impervious_features()`.

### USGS 3DEP DEM — Elevation (SW Florida S4, New Orleans S2)

| Dataset | S3 target | Notes | Status |
|---------|-----------|-------|--------|
| SW Florida tiles | `raw/dem/3dep/v1/southwest_florida/*.tif` | 1/3 arc-second | PENDING |
| New Orleans tiles | `raw/dem/3dep/v1/new_orleans/*.tif` | 1/3 arc-second | PENDING |
| Manifest | `manifests/usgs_3dep_dem/v1/manifest.json` | | PENDING |

Source: USGS TNM API. ZCTA mean elevation via `build_elevation_features()`.

### USGS MTBS — Burn Scar Perimeters (Riverside S3)

| Dataset | S3 target | Notes | Status |
|---------|-----------|-------|--------|
| CA 2015-2023 perimeters | `raw/mtbs/perimeters/v2023/burn_perims_ca_2015_2023.parquet` | ~5 MB | PENDING |
| Manifest | `manifests/mtbs_burn_perimeters/v2023/manifest.json` | | PENDING |

Source: USGS MTBS `edcintl.cr.usgs.gov`. ZCTA binary overlap via `build_burn_scar_features()`.

### NHDPlus V2 — Catchments (Riverside S3, Houston S1)

| Dataset | S3 target | Notes | Status |
|---------|-----------|-------|--------|
| VPU 18 (California) catchments | `raw/nhdplus/catchments/v2/catchments_vpu18.parquet` | ~30 MB | PENDING |
| VPU 12 (Texas Gulf) catchments | `raw/nhdplus/catchments/v2/catchments_vpu12.parquet` | ~30 MB | PENDING |
| Manifest | `manifests/nhdplus_v2_catchments/v2/manifest.json` | | PENDING |

Source: EPA NHDPlus V2. ZCTA joins for `upstream_catchment_km2`, `wash_segment_id`, `bayou_segment_id` via `build_catchment_features()`.

### MTA Subway Stations (NYC S5)

| Dataset | S3 target | Notes | Status |
|---------|-----------|-------|--------|
| Subway stations | `raw/mta/subway_stations/v1/subway_stations.parquet` | ~500 rows | PENDING |
| Manifest | `manifests/mta_subway_stations/v1/manifest.json` | | PENDING |

Source: NYC Open Data `kk4q-3rt2`. ZCTA station counts and distances via `build_subway_features()`.

---

### NOAA HRRR — 3-km hourly QPF (surrogate training input)

| Dataset | S3 target | Notes | Status |
|---------|-----------|-------|--------|
| Harvey 2017 (Aug 17 – Sep 3, 6-hrly init, fhr=01) | `raw/noaa_hrrr/harvey2017/` | ~200 MB | PENDING |
| Imelda 2019 (Sep 17-21) | `raw/noaa_hrrr/imelda2019/` | ~30 MB | PENDING |
| Beryl 2024 (Jul 8-12) | `raw/noaa_hrrr/beryl2024/` | ~30 MB | PENDING |
| Ida 2021 NYC (Sep 1-3) | `raw/noaa_hrrr/ida2021_nyc/` | ~20 MB | PENDING |
| Ian 2022 (Sep 23 – Oct 1) | `raw/noaa_hrrr/ian2022/` | ~50 MB | PENDING |
| Helene 2024 (Sep 24 – Oct 1) | `raw/noaa_hrrr/helene2024/` | ~50 MB | PENDING |
| Milton 2024 (Oct 7-12) | `raw/noaa_hrrr/milton2024/` | ~30 MB | PENDING |
| Hilary 2023 (Aug 19-23) | `raw/noaa_hrrr/hilary2023/` | ~25 MB | PENDING |
| AR Flood 2023 (Mar 1-23, 6-hrly init) | `raw/noaa_hrrr/ar_flood_2023/` | ~150 MB | PENDING |

Source: University of Utah HRRR archive (pando-rgw01.chpc.utah.edu); AWS Open Data fallback.
Format: grib2 (APCP field); 3-km CONUS grid.

---

### NHC HURDAT2 — storm tracks

| Dataset | S3 target | Status |
|---------|-----------|--------|
| All Atlantic basin tracks 2015-2024 (covers Harvey/Imelda/Beryl/Ida/Henri) | `raw/hurdat2/hurdat2_1851_2024.parquet` | PENDING |

Fields: `storm_id, name, datetime_utc, lat, lon, max_wind_kt, min_pressure_mb, category`

### USGS STN — high-water marks

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Harvey 2017 high-water marks (~2000) | `raw/usgs_stn/harvey2017_hwm.parquet` | PENDING |
| Imelda 2019 high-water marks | `raw/usgs_stn/imelda2019_hwm.parquet` | PENDING |

Fields: `hwm_id, latitude, longitude, elev_ft, datum, uncertainty_ft, event_name`

### Houston 311

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Flood service requests, Harvey window (Aug 25 – Sep 10 2017) | `raw/houston_311/harvey2017_311.parquet` | PENDING |
| Flood service requests, Imelda window (Sep 17-25 2019) | `raw/houston_311/imelda2019_311.parquet` | PENDING |
| Flood service requests, Beryl window (Jul 7-15 2024) | `raw/houston_311/beryl2024_311.parquet` | PENDING |

Fields: `sr_number, create_date, sr_type, latitude, longitude, zcta_id`
Source: `data.houstontx.gov`

### NOAA Tides and Currents — New Orleans

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Station 8761724 (Pilots Station East), Ida 2021 | `raw/noaa_tides/no_8761724_ida2021.parquet` | PENDING |
| Station 8761927 (New Canal Station), Ida 2021 | `raw/noaa_tides/no_8761927_ida2021.parquet` | PENDING |
| Station 8760922 (Michoud Canal), Ida 2021 | `raw/noaa_tides/no_8760922_ida2021.parquet` | PENDING |

Fields: `station_id, datetime_utc, water_level_m, predicted_m, surge_m`

### USACE National Levee Database

| Dataset | S3 target | Status |
|---------|-----------|--------|
| New Orleans levee system records | `raw/usace_levees/no_levees.parquet` | PENDING |
| NYC / NJ levee records | `raw/usace_levees/nyc_levees.parquet` | PENDING |

Fields: `levee_id, system_name, district_name, condition_rating, inspection_date, geometry_wkt`
Source: `levees.sec.usace.army.mil/api/`

### NYC 311

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Flooding complaints, Ida 2021 (Sep 1-3, type: Sewer/Street Flooding/Catch Basin) | `raw/nyc_311/ida2021_flooding_311.parquet` | PENDING |
| Flooding complaints, Henri 2021 (Aug 21-23) | `raw/nyc_311/henri2021_flooding_311.parquet` | PENDING |

Fields: `unique_key, created_date, complaint_type, latitude, longitude, bbl, zip_code`
Source: NYC Open Data Socrata API (`data.cityofnewyork.us`)

### NYC DEP Sewer-sheds

| Dataset | S3 target | Status |
|---------|-----------|--------|
| NYC DEP drainage area polygons | `raw/nyc_sewersheds/nyc_sewersheds.gpkg` | PENDING |

### FEMA OpenFEMA — event-specific

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Disaster declarations (all DRs) | `raw/openfema/disaster_declarations.parquet` | PENDING |
| NFIP claims, DR-4332-TX (Harvey) | `raw/openfema/nfip_claims_dr4332.parquet` | PENDING |
| NFIP claims, DR-4466-TX (Imelda) | `raw/openfema/nfip_claims_dr4466.parquet` | PENDING |
| NFIP claims, DR-4781-TX (Beryl) | `raw/openfema/nfip_claims_dr4781.parquet` | PENDING |
| NFIP claims, DR-4611-LA (Ida) | `raw/openfema/nfip_claims_dr4611.parquet` | PENDING |
| NFIP claims, DR-4615-NY (Ida NY) | `raw/openfema/nfip_claims_dr4615.parquet` | PENDING |
| NFIP claims, DR-4673-FL (Ian) | `raw/openfema/nfip_claims_dr4673.parquet` | PENDING |
| NFIP claims, DR-4828-FL (Helene) | `raw/openfema/nfip_claims_dr4828.parquet` | PENDING |
| NFIP claims, DR-4834-FL (Milton) | `raw/openfema/nfip_claims_dr4834.parquet` | PENDING |
| NFIP claims, DR-4699-CA (Hilary) | `raw/openfema/nfip_claims_dr4699.parquet` | PENDING |

---

### NOAA MRMS Stage IV — SW Florida + Riverside-Coachella

| Dataset | S3 target | Notes | Status |
|---------|-----------|-------|--------|
| Ian 2022 (Sep 23 – Oct 1, hourly) | `raw/noaa_mrms/ian2022/` | ~200 MB | PENDING |
| Helene 2024 (Sep 24 – Oct 1, hourly) | `raw/noaa_mrms/helene2024/` | ~200 MB | PENDING |
| Milton 2024 (Oct 7-12, hourly) | `raw/noaa_mrms/milton2024/` | ~120 MB | PENDING |
| Hilary 2023 (Aug 19-23, hourly) | `raw/noaa_mrms/hilary2023/` | ~100 MB | PENDING |
| AR Flood 2023 (Mar 1-23, hourly) | `raw/noaa_mrms/ar_flood_2023/` | ~550 MB | PENDING |

### NOAA Tides — SW Florida (Ian/Helene/Milton)

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Fort Myers 8725520, Ian 2022 | `raw/noaa_tides/swfl_8725520_ian2022.parquet` | PENDING |
| Fort Myers 8725520, Helene 2024 | `raw/noaa_tides/swfl_8725520_helene2024.parquet` | PENDING |
| Fort Myers 8725520, Milton 2024 | `raw/noaa_tides/swfl_8725520_milton2024.parquet` | PENDING |
| St. Petersburg 8726520, all events | `raw/noaa_tides/swfl_8726520_{event}.parquet` | PENDING |
| Naples 8725110, all events | `raw/noaa_tides/swfl_8725110_{event}.parquet` | PENDING |
| Clearwater Beach 8726724, all events | `raw/noaa_tides/swfl_8726724_{event}.parquet` | PENDING |

### NOAA SLOSH — SW Florida surge grids

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Ian 2022 SLOSH MOM/MEOW | `raw/noaa_slosh/ian2022/` | PENDING |
| Helene 2024 SLOSH MOM/MEOW | `raw/noaa_slosh/helene2024/` | PENDING |
| Milton 2024 SLOSH MOM/MEOW | `raw/noaa_slosh/milton2024/` | PENDING |

Note: SLOSH auto-download may fail for some events; check for `MANUAL_DOWNLOAD_REQUIRED.txt`
in the prefix and download manually from https://www.nhc.noaa.gov/surge/slosh.php if needed.

### USGS NWIS — Riverside-Coachella + SW Florida

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Riverside-Coachella gauges, Hilary 2023 (Aug 19-23) | `raw/usgs_nwis/riverside_coachella_hilary2023.parquet` | PENDING |
| Riverside-Coachella gauges, AR Flood 2023 (Mar 1-23) | `raw/usgs_nwis/riverside_coachella_ar_flood_2023.parquet` | PENDING |
| SW Florida gauges, Ian 2022 (Sep 23 – Oct 1) | `raw/usgs_nwis/southwest_florida_ian2022.parquet` | PENDING |
| SW Florida gauges, Helene 2024 (Sep 24 – Oct 1) | `raw/usgs_nwis/southwest_florida_helene2024.parquet` | PENDING |
| SW Florida gauges, Milton 2024 (Oct 7-12) | `raw/usgs_nwis/southwest_florida_milton2024.parquet` | PENDING |

---

## Hand-coded (checked into evidence/)

| File | Content | Due |
|------|---------|-----|
| `evidence/no_pump_stations_ida2021.csv` | ~30 NO drainage districts, pump status during Ida | May 31 |
| `evidence/nyc_subway_flooding_ida2021.csv` | ~30 MTA stations, flooding Y/N, depth estimate | May 31 |

---

## Processed outputs (built by build_event_dataset job)

| Dataset | S3 target | Status |
|---------|-----------|--------|
| Houston (zcta, event) feature table | `processed/houston/houston_event_features.parquet` | PENDING |
| New Orleans (district, event) feature table | `processed/new_orleans/no_event_features.parquet` | PENDING |
| NYC (sewershed, event) feature table | `processed/nyc/nyc_event_features.parquet` | PENDING |
| Riverside-Coachella (zcta, event) variogram inputs | `processed/riverside_coachella/rc_event_features.parquet` | PENDING |
| SW Florida (zcta, event) variogram inputs | `processed/southwest_florida/swfl_event_features.parquet` | PENDING |

---

## Upstream model

| Path | Status |
|------|--------|
| `model/maxfloodcast/` | PENDING — awaiting Lee et al. response |
| `model/surrogate/{scenario}/xgboost/model.json` | PENDING — fallback if no response by May 30 |
| `model/surrogate/{scenario}/lstm/model.pt` | PENDING — fallback if no response by May 30 |
| `model/surrogate/{scenario}/eval_metrics.json` | PENDING — fallback if no response by May 30 |
