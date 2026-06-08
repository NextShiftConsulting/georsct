# Series 035 Data Acquisition Checklist

**Bucket:** `s3://swarm-floodrsct-data/`
**All jobs:** SageMaker Processing — commit + push before every launch.
**Scenarios:** Houston (full), New Orleans (partial), NYC (partial),
               Riverside-Coachella (variogram only), SW Florida (variogram only)

---

## Day 1 — May 27 (Today)

- [ ] Send MaxFloodCast request to `lipai.huang@tamu.edu`
- [ ] `git commit + push` this scaffold
- [ ] `launch_copy_geocertdb2026.py` — copy reusable ZCTA features (all 5 scenarios)
- [ ] `launch_fetch_usgs_nwis.py --scenario houston` — Harris County gauges (Harvey/Imelda/Beryl)
- [ ] `launch_fetch_hurdat2.py` — all 8 storm tracks (Harvey/Imelda/Beryl/Ida/Henri/Ian/Helene/Milton)
- [ ] `launch_fetch_openfema_event.py` — all 8 DRs across 5 scenarios

## Day 2 — May 28

- [ ] `python scripts/launch_fetch_nlcd_impervious.py` — NLCD 2021 impervious (S1, S5)
- [ ] `python scripts/launch_fetch_dem_elevation.py` — USGS 3DEP DEM (S2, S4)
- [ ] `python scripts/launch_fetch_mtbs_burn_scars.py` — MTBS burn perimeters (S3)
- [ ] `python scripts/launch_fetch_nhdplus_catchments.py` — NHDPlus catchments (S1, S3)
- [ ] `python scripts/launch_fetch_mta_stations.py` — NYC MTA stations (S5)
- [ ] `launch_fetch_noaa_mrms.py --event harvey2017` — Stage IV hourly (~500 MB)
- [ ] `launch_fetch_noaa_mrms.py --event imelda2019`
- [ ] `launch_fetch_noaa_mrms.py --event beryl2024`
- [ ] `launch_fetch_noaa_mrms.py --event ida2021_nyc`
- [ ] `launch_fetch_noaa_mrms.py --event ian2022`
- [ ] `launch_fetch_noaa_mrms.py --event hilary2023`
- [ ] `launch_fetch_noaa_hrrr.py --event harvey2017` — HRRR 3-km QPF
- [ ] `launch_fetch_noaa_hrrr.py --event imelda2019`
- [ ] `launch_fetch_noaa_hrrr.py --event beryl2024`
- [ ] `launch_fetch_noaa_hrrr.py --event ida2021_nyc`
- [ ] `launch_fetch_noaa_hrrr.py --event ian2022`
- [ ] `launch_fetch_noaa_hrrr.py --event hilary2023`
- [ ] `launch_fetch_usgs_stn.py --event harvey2017` — ~2000 high-water marks
- [ ] `launch_fetch_usgs_stn.py --event imelda2019`
- [ ] `launch_fetch_houston_311.py` — flood service requests, all Houston event windows

## Day 3 — May 29

- [ ] `launch_fetch_usgs_nwis.py --scenario new_orleans` — Ida 2021 gauges
- [ ] `launch_fetch_usgs_nwis.py --scenario nyc` — Ida/Henri 2021 coastal gauges
- [ ] `launch_fetch_usgs_nwis.py --scenario riverside_coachella` — Tahquitz/Whitewater/Mojave gauges
- [ ] `launch_fetch_usgs_nwis.py --scenario southwest_florida` — Ian/Helene/Milton gauges
- [ ] `launch_fetch_noaa_tides.py` — New Orleans tidal stations (Ida 2021)
- [ ] `launch_fetch_noaa_tides_swfl.py` — SW Florida stations (Ian/Helene/Milton)
- [ ] Verify Census/SVI/EJScreen columns in geocertdb2026 copy (no re-download needed)

## Day 4 — May 30

- [ ] `launch_fetch_noaa_mrms.py --event helene2024`
- [ ] `launch_fetch_noaa_mrms.py --event milton2024`
- [ ] `launch_fetch_noaa_mrms.py --event ar_flood_2023` — Riverside-Coachella AR event (~22 days)
- [ ] `launch_fetch_noaa_hrrr.py --event helene2024`
- [ ] `launch_fetch_noaa_hrrr.py --event milton2024`
- [ ] `launch_fetch_noaa_hrrr.py --event ar_flood_2023`
- [ ] `launch_fetch_noaa_slosh.py` — NHC SLOSH surge grids for Ian/Helene/Milton
- [ ] `launch_fetch_usace_levees.py --scenario new_orleans`
- [ ] `launch_fetch_usace_levees.py --scenario nyc`
- [ ] `launch_fetch_nyc_311.py` — Ida 2021 (Sept 1-2 window) + Henri 2021
- [ ] `launch_fetch_nyc_sewersheds.py` — NYC DEP drainage area polygons
- [ ] **MaxFloodCast decision deadline** — no response → activate surrogate path

## MOE Flood Stack (run in parallel with Day 2-3 data pulls)

- [ ] `python scripts/launch_fetch_prithvi_eo2.py --dry-run` — validate config (ml.g5.2xlarge)
- [ ] `python scripts/launch_fetch_prithvi_eo2.py` — download ibm-nasa-geospatial/Prithvi-EO-2.0 + smoke test → `s3://swarm-floodrsct-data/model/prithvi_eo2/`
- [ ] Verify smoke test: `aws s3 cp s3://swarm-floodrsct-data/model/prithvi_eo2/smoke_test/smoke_test_result.json -` — status must be `PASS`
- [ ] `python scripts/launch_fetch_floodsimbench.py --dry-run` — validate config (ml.m5.xlarge)
- [ ] `python scripts/launch_fetch_floodsimbench.py` — download ECMWF/FloodSimBench + clip to 5 scenarios → `s3://swarm-floodrsct-data/raw/floodsimbench/`
- [ ] Verify clip stats: `aws s3 cp s3://swarm-floodrsct-data/raw/floodsimbench/clip_stats.json -` — all 5 scenarios present
- [ ] `python scripts/launch_fetch_sen1floods11.py --dry-run` — validate config (ml.m5.large)
- [ ] `python scripts/launch_fetch_sen1floods11.py` — download Sen1Floods11 benchmark (metadata only) → `s3://swarm-floodrsct-data/raw/sen1floods11/`
- [ ] Verify summary: `aws s3 cp s3://swarm-floodrsct-data/raw/sen1floods11/summary.json -` — resolved_hf_id logged

## Day 5 — May 31

- [ ] Hand-code `evidence/no_pump_stations_ida2021.csv` from Ida 2021 post-event reports
- [ ] Hand-code `evidence/nyc_subway_flooding_ida2021.csv` from MTA post-Ida reports (~30 stations)
- [ ] If surrogate path: `launch_train_surrogate.py --scenario houston` (ml.g5.2xlarge, ~2-3 hrs each)
- [ ] If surrogate path: `launch_train_surrogate.py --scenario new_orleans`
- [ ] If surrogate path: `launch_train_surrogate.py --scenario nyc`
- [ ] If surrogate path: `launch_train_surrogate.py --scenario riverside_coachella`
- [ ] If surrogate path: `launch_train_surrogate.py --scenario southwest_florida`
- [ ] Verify SLOSH results: if `MANUAL_DOWNLOAD_REQUIRED.txt` present in any event prefix,
      download manually from https://www.nhc.noaa.gov/surge/slosh.php and upload to
      `s3://swarm-floodrsct-data/raw/noaa_slosh/{event}/`
- [ ] Verify all `raw/` prefixes have expected file counts (see DATA_MANIFEST.md)

## Day 6 — June 1 -- DATA LOCK A

- [ ] `python scripts/launch_build_event_dataset.py --scenario houston`
- [ ] Verify Houston MVD: 100+ ZCTAs, 2+ events each, all required columns present
- [ ] **DATA LOCK A: Houston minimum viable dataset frozen.** Begin Experiments 1-6 on Houston.

## Day 7 — June 2 -- DATA LOCK B

- [ ] `python scripts/launch_build_event_dataset.py --scenario new_orleans`
- [ ] `python scripts/launch_build_event_dataset.py --scenario nyc`
- [ ] `python scripts/launch_build_event_dataset.py --scenario riverside_coachella`
- [ ] `python scripts/launch_build_event_dataset.py --scenario southwest_florida`
- [ ] Verify NO MVD: 30+ districts, NFIP claims, levee ratings, pump-station ground truth
- [ ] Verify NYC MVD: 100+ sewer-sheds, 311 complaints, ACS, NFIP claims
- [ ] Verify RC MVD: 30+ ZCTAs, gauge peaks, rainfall totals (variogram inputs only)
- [ ] Verify SWFL MVD: 50+ ZCTAs, NFIP claims, SLOSH surge, NOAA tides (variogram inputs only)
- [ ] **DATA LOCK B: all 5 scenarios frozen.**

## Day 8-9 — June 3-4

- [ ] Experiment execution per DOE_LOCKED.md
- [ ] **RESULTS LOCK: June 4** — no experiment re-runs after this point
- [ ] §8 drafting

## Day 10 — June 5

- [ ] Paper submission
