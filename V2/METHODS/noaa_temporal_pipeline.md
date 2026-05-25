# NOAA Storm Events Temporal Pipeline

**Source:** NOAA NCEI Storm Events Database, 1996–2024
**Script:** `rsct-geocert/data/geocert/v24/run_noaa.py`
**Launcher:** `rsct-geocert/data/geocert/v24/sagemaker_noaa.py --temporal`
**Output prefix:** `s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/`

---

## Why Temporal Structure Matters

The existing `noaa_storm_events_zcta.parquet` collapses 29 years of flood history into
aggregate counts. This is sufficient for static risk features but blocks three planned
experiments that require temporal variation:

- **Experiment 1** — Pre-flood vs post-flood prediction: does a model trained on pre-2005
  ACS features predict 2005–2024 flood outcomes? Requires epoch-level aggregates.
- **Experiment 2** — Trend analysis: is flood frequency increasing per ZCTA? Requires
  year-by-year long format.
- **Experiment 3** — Anomaly detection: ZCTAs with high FEMA zone coverage but zero
  historical claims. Requires the long format joined against NFIP claims.

---

## Epoch Design

Epochs are keyed to named major flood events rather than arbitrary decades. This preserves
physical meaning when comparing across experiments.

| Epoch | Years     | Anchor event              |
|-------|-----------|---------------------------|
| e1    | 1996–2004 | Pre-Katrina baseline      |
| e2    | 2005–2011 | Post-Katrina / pre-Sandy  |
| e3    | 2012–2024 | Sandy onward              |

Epoch boundaries are defined in `run_noaa.py::EPOCHS` and applied consistently in both
the wide output and any downstream joins.

---

## Data Source

**NOAA NCEI Storm Events Database**
- URL: `https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/`
- Format: annual gzip CSV (`StormEvents_details-ftp_v1.0_dYYYY_cDATE.csv.gz`)
- Coverage: 1996–2024, ~29 files, ~200–400 MB compressed
- Flood event types retained: `Flash Flood`, `Flood`, `Coastal Flood`, `Lakeshore Flood`
- Only county-type entries (`CZ_TYPE == "C"`) are used; zone-type entries are dropped

**County → ZCTA join:** majority-county assignment from `zcta_county_crosswalk.parquet`.
Each ZCTA inherits the flood history of its majority county by land-area overlap. This is a
conservative join: ZCTAs split across counties with different flood histories will be
assigned to the county covering the largest fraction of their land area.

---

## Pipeline Stages

### Stage 1 — Fetch (per year)
`fetch_year(year, url)` downloads and parses one annual gzip file:
- Filters to flood event types
- Retains county-type entries only
- Parses NOAA damage strings (`10K`, `1.5M`, `2B`) to `$1000s`
- Returns columns: `county_fips`, `year`, `prop_dmg_k`, `crop_dmg_k`, `deaths`, `injuries`

### Stage 2A — Standard aggregation (existing behavior, unchanged)
`aggregate_to_county()` → `county_to_zcta()`:
- Groups by `county_fips` only (collapses all years)
- Produces: `flood_event_count`, `flood_event_count_5y` (2019–2024), `flood_events_per_year`,
  `flood_deaths`, `flood_injuries`, `flood_property_damage_k`, `flood_crop_damage_k`
- Output: `noaa_storm_events_zcta.parquet` (31,789 rows × 8 columns)

### Stage 2B — Temporal aggregation (`--temporal` flag)
`aggregate_to_county_by_year()`:
- Groups by `(county_fips, year)` — preserves the year dimension
- Same metrics as Stage 2A but at county × year grain

### Stage 3 — Long format
`make_long_zcta()`:
- Joins county × year to ZCTA via crosswalk
- Builds complete `zcta_id × year` index covering all ZCTAs and all years 1996–2024
- Zero-fills missing combinations (ZCTA had no flood events in that year)
- Output: `noaa_storm_events_long.parquet` (~920K rows × 7 columns)

### Stage 4 — Wide epoch format
`make_wide_epochs()`:
- Pivots long format to one row per ZCTA
- For each epoch and each metric: `{metric}_{epoch}` (e.g., `flood_events_e2`)
- Additional rolling features:
  - `flood_events_total`, `deaths_total`, `property_damage_k_total`, `crop_damage_k_total`
  - `property_damage_k_peak_yr` — maximum single-year property damage (captures extreme events)
- Output: `noaa_storm_events_wide.parquet` (31,789 rows × 25+ columns)

---

## Output Schemas

### `noaa_storm_events_long.parquet`
| Column | Type | Description |
|---|---|---|
| `zcta_id` | str | 5-digit ZCTA, zero-padded |
| `year` | int | Calendar year (1996–2024) |
| `flood_events` | int | Count of flood events in this county this year |
| `deaths` | int | Direct + indirect deaths |
| `injuries` | int | Direct + indirect injuries |
| `property_damage_k` | float | Property damage in $1,000s |
| `crop_damage_k` | float | Crop damage in $1,000s |

**Grain:** one row per (zcta_id, year). All 31,789 ZCTAs × 29 years = 922,181 rows.
Zero-filled: ZCTAs with no events in a given year appear with all metrics = 0.

### `noaa_storm_events_wide.parquet`
| Column pattern | Description |
|---|---|
| `zcta_id` | 5-digit ZCTA |
| `flood_events_{e1,e2,e3}` | Event count per epoch |
| `deaths_{e1,e2,e3}` | Deaths per epoch |
| `property_damage_k_{e1,e2,e3}` | Property damage per epoch ($1000s) |
| `crop_damage_k_{e1,e2,e3}` | Crop damage per epoch ($1000s) |
| `flood_events_total` | Total events 1996–2024 |
| `deaths_total` | Total deaths 1996–2024 |
| `property_damage_k_total` | Total property damage 1996–2024 |
| `crop_damage_k_total` | Total crop damage 1996–2024 |
| `property_damage_k_peak_yr` | Max single-year property damage |

**Grain:** one row per zcta_id. 31,789 rows.

---

## Running the Job

```bash
# From rsct-geocert/data/geocert/v24/

# Validate config (no launch)
python sagemaker_noaa.py --temporal --dry-run

# Launch
python sagemaker_noaa.py --temporal
```

**Instance:** `ml.m5.large` (2 vCPU, 8 GB RAM)
**Estimated runtime:** 20–35 min with temporal, 15–25 min without
**Estimated cost:** ~$0.06

Monitor:
```bash
MSYS_NO_PATHCONV=1 aws sagemaker describe-processing-job \
  --processing-job-name <JOB_NAME> \
  --region us-east-1 --profile nsc-swarm \
  --query 'ProcessingJobStatus'

MSYS_NO_PATHCONV=1 aws logs tail /aws/sagemaker/ProcessingJobs \
  --log-stream-name-prefix <JOB_NAME> --follow \
  --region us-east-1 --profile nsc-swarm
```

---

## Validation Checks

The pipeline includes a Houston/Harris County spot check on the standard output:
ZCTAs starting with `770xx` must have nonzero total flood events. Zero indicates a
broken county FIPS join.

For the temporal outputs, per-epoch active ZCTA counts are logged:
```
e1 (1996-2004): N ZCTAs with events
e2 (2005-2011): N ZCTAs with events
e3 (2012-2024): N ZCTAs with events
Peak damage ZCTA: XXXXX ($Y.YM damage in worst year)
```
e3 should show materially more active ZCTAs than e1, reflecting increased reporting
coverage and actual flood frequency growth.

---

## Known Limitations

- **County majority assignment** — ZCTAs at county boundaries inherit one county's history.
  A ZCTA split 51/49 across a flood-prone and flood-free county is assigned to the
  flood-prone one, which may overstate risk for the 49% portion.
- **Reporting coverage** — NOAA storm event reporting improved significantly after 2007.
  e1 counts are understated relative to e2/e3 for the same physical risk level.
- **Event type scope** — Only four flood types are retained. Tidal flooding, dam failures,
  and storm surge captured under other event types are excluded.
- **Damage string parsing** — NOAA damage values are reporter estimates, not verified
  insurance losses. Use NFIP claims (`nfip_claims_zcta.parquet`) for verified financial
  outcomes.
- **No population normalization** — Event counts and damage totals are raw aggregates.
  Divide by `acs_total_pop` or housing units before comparing high- vs low-density ZCTAs.
