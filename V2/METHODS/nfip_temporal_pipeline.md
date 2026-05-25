# FEMA NFIP Claims Temporal Pipeline

**Source:** FEMA OpenFEMA FimaNfipClaims v2, ~2.4M records
**Script:** `rsct-geocert/data/geocert/v24/run_nfip.py`
**Launcher:** `rsct-geocert/data/geocert/v24/sagemaker_nfip.py --temporal`
**Output prefix:** `s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/`

---

## Why Temporal Structure Matters

The existing `nfip_claims_zcta.parquet` collapses the full NFIP history into
aggregate totals. This is sufficient for static risk features but blocks three
planned experiments:

- **Experiment 1** — Pre-flood vs post-flood prediction: does a model trained on
  pre-2005 ACS features predict 2005–2024 NFIP claim rates? Requires epoch-level
  aggregates.
- **Experiment 2** — Trend analysis: is claims frequency increasing per ZCTA?
  Requires year-by-year long format.
- **Experiment 3** — Anomaly detection: ZCTAs with high FEMA flood zone coverage
  but zero historical NFIP claims. Requires long format joined against NOAA storm
  events.

---

## Epoch Design

Epochs are identical to the NOAA Storm Events temporal pipeline so that NFIP
claims and storm event counts can be joined on (zcta_id, epoch) without
re-alignment.

| Epoch | Years     | Anchor event              |
|-------|-----------|---------------------------|
| e1    | 1996–2004 | Pre-Katrina baseline      |
| e2    | 2005–2011 | Post-Katrina / pre-Sandy  |
| e3    | 2012–2024 | Sandy onward              |

Note: NFIP records extend back to ~1978 but epochs start at 1996 to match NOAA.
Pre-1996 claims are included in `_total` columns but excluded from epoch columns.
Epoch boundaries are defined in `run_nfip.py::EPOCHS`.

---

## Data Source

**FEMA OpenFEMA FimaNfipClaims**
- URL: `https://www.fema.gov/api/open/v2/FimaNfipClaims`
- Format: paginated JSON/CSV API, 10K records per page, ~240 pages total
- Coverage: ~2.4M paid claims, ~1978–2024
- Fields used: `reportedZipCode`, `amountPaidOnBuildingClaim`,
  `amountPaidOnContentsClaim`, `yearOfLoss`, `primaryResidenceIndicator`
- Filter: only records with `total_loss > 0` (building + contents) are accumulated

**ZIP → ZCTA mapping:** NFIP uses 5-digit ZIP codes rather than ZCTAs. ZIP codes
are normalized (strip non-digits, zero-pad to 5) and matched directly to ZCTAs.
For residential ZCTAs this is a near-1:1 match. ZCTAs with no matching ZIP claims
are zero-filled.

---

## Pipeline Stages

### Stage 1 — Streaming accumulation
`fetch_page(offset)` + `accumulate_page(page, agg, temporal_agg)`:
- Pages through the OpenFEMA API 10K records at a time
- Standard `agg` dict: `{zip5: {count, building, contents}}`
- Temporal `temporal_agg` dict when `--temporal`: `{zip5: {year: {count, building, contents}}}`
- `yearOfLoss` parsed per-row; values outside 1970–2030 stored under sentinel key 0 and
  excluded from long/wide outputs
- Checkpoint every 50 pages to S3 for crash recovery; temporal_agg included in checkpoint

### Stage 2 — Standard aggregation (existing behavior, unchanged)
Flattens `agg` dict → DataFrame → merges against crosswalk → zero-fills 31K ZCTAs.
Output: `nfip_claims_zcta.parquet` (31,789 rows × 6 columns)

### Stage 3 — Long format (`--temporal` flag)
`make_long_zcta_nfip(temporal_agg, xwalk)`:
- Flattens nested dict to flat rows: `(zip5, year) → {count, building, contents, total}`
- Builds complete `zcta_id × year` index from min to max year observed in data
- Zero-fills missing combinations
- Output: `nfip_claims_long.parquet` (~1.5M rows)

### Stage 4 — Wide epoch format (`--temporal` flag)
`make_wide_epochs_nfip(long)`:
- Pivots long format to one row per ZCTA
- For each epoch and each metric: `{metric}_{epoch}` (e.g., `nfip_claim_count_e2`)
- Rolling totals across all years: `{metric}_total`
- Peak single-year loss: `nfip_total_loss_peak_yr`
- Output: `nfip_claims_wide.parquet` (31,789 rows × 17+ columns)

---

## Output Schemas

### `nfip_claims_zcta.parquet` (standard, unchanged)
| Column | Type | Description |
|---|---|---|
| `zcta_id` | str | 5-digit ZCTA |
| `nfip_claim_count` | int | Total paid claims 1978–2024 |
| `nfip_total_building_loss` | float | Total building loss ($) |
| `nfip_total_contents_loss` | float | Total contents loss ($) |
| `nfip_total_loss` | float | Total loss ($) |
| `nfip_mean_loss_per_claim` | float | Mean loss per paid claim ($) |
| `nfip_has_claims` | bool | True if any paid claims |

### `nfip_claims_long.parquet`
| Column | Type | Description |
|---|---|---|
| `zcta_id` | str | 5-digit ZCTA, zero-padded |
| `year` | int | Calendar year (min–max in data) |
| `nfip_claim_count` | int | Paid claims in this ZCTA this year |
| `nfip_building_loss` | float | Building loss ($) |
| `nfip_contents_loss` | float | Contents loss ($) |
| `nfip_total_loss` | float | Total loss ($) |

**Grain:** one row per (zcta_id, year). Zero-filled: ZCTAs with no claims in a
given year appear with all metrics = 0.

### `nfip_claims_wide.parquet`
| Column pattern | Description |
|---|---|
| `zcta_id` | 5-digit ZCTA |
| `nfip_claim_count_{e1,e2,e3}` | Paid claim count per epoch |
| `nfip_building_loss_{e1,e2,e3}` | Building loss per epoch ($) |
| `nfip_contents_loss_{e1,e2,e3}` | Contents loss per epoch ($) |
| `nfip_total_loss_{e1,e2,e3}` | Total loss per epoch ($) |
| `nfip_claim_count_total` | Total paid claims all years |
| `nfip_building_loss_total` | Total building loss all years ($) |
| `nfip_contents_loss_total` | Total contents loss all years ($) |
| `nfip_total_loss_total` | Total loss all years ($) |
| `nfip_total_loss_peak_yr` | Max single-year total loss ($) |

**Grain:** one row per zcta_id. 31,789 rows.

---

## Running the Job

```bash
# From rsct-geocert/data/geocert/v24/

# Validate config (no launch)
python sagemaker_nfip.py --temporal --dry-run

# Launch
python sagemaker_nfip.py --temporal

# Quick smoke test (50K records)
python sagemaker_nfip.py --temporal --max-pages 5
```

**Instance:** `ml.m5.xlarge` (4 vCPU, 16 GB RAM)
**Estimated runtime:** 30–50 min with temporal, 25–40 min without
**Estimated cost:** ~$0.20

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

The pipeline logs per-epoch active ZCTA counts:
```
e1 (1996-2004): N ZCTAs with claims
e2 (2005-2011): N ZCTAs with claims
e3 (2012-2024): N ZCTAs with claims
Peak single-year loss ZCTA: XXXXX ($Y.YM loss in worst year)
```

e2 should spike (Katrina 2005), e3 should be highest overall (Harvey 2017,
Florence 2018, Ida 2021). The Harris County spot check on the standard output
verifies that at least 1 test ZCTA in ZIP 770xx carries claims.

---

## Known Limitations

- **ZIP ≠ ZCTA** — NFIP uses USPS ZIP codes. ZIP-to-ZCTA overlap is high for
  residential ZCTAs but imperfect for PO-box ZIPs and rural multi-county ZIPs.
  No spatial crosswalk is applied; the join is direct 5-digit code match.
- **Reporting date vs. loss year** — `yearOfLoss` is the year of the flood event,
  not the year the claim was filed. Delayed filings (common after major disasters)
  are attributed to the event year, which is the desired behavior.
- **No population normalization** — Claim counts and loss totals are raw aggregates.
  Divide by `acs_total_pop` or housing units before comparing dense vs sparse ZCTAs.
- **Pre-1996 claims** — Included in `_total` columns; excluded from epoch columns.
  About 5–10% of all claims pre-date 1996.
- **Loss amounts in raw dollars** — Unlike NOAA damage (stored in $1000s), NFIP
  losses are stored in full dollar amounts. Be careful when joining the two sources.
- **Primary residence filter not applied** — `primaryResidenceIndicator` is retained
  in `SELECT_COLS` for downstream use but not used as a pipeline filter. Downstream
  experiments may wish to filter to primary residence only for demographic analyses.
