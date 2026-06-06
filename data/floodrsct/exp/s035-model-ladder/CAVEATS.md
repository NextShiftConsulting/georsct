# S035 Model Ladder — Caveats & Known Limitations

**Experiment:** s035-model-ladder
**Created:** 2026-06-02
**Status:** Living document — update as caveats are discovered or resolved.

---

## C-001: ZIP Code ≠ ZCTA in NFIP Claims Join

**Severity:** Moderate
**Affects:** All phases (R0–R4) that use NFIP data
**Discovered by:** Human review (not caught by any of 6 MMAR reviewer LLMs)

### The Problem

`fetch_openfema_event.py` line 130 renames the OpenFEMA field
`reportedZipCode` to `zcta_id`:

```python
df = df.rename(columns={"reportedZipCode": "zcta_id"})
```

This silently treats USPS ZIP codes as Census ZCTAs. They are not the
same thing:

| Concept | Source | Definition |
|---------|--------|------------|
| ZIP code | USPS | Mail delivery route — point or polygon, updated monthly, no stable geometry |
| ZCTA | Census Bureau | Area approximation of ZIP delivery areas — TIGER/Line 2020/2022, stable geometry |

The mapping is approximately 1:1 for residential delivery ZIPs in metro
areas (~95% overlap), but the mismatch cases are real:

- **Many-to-one:** Multiple ZIPs can map to the same ZCTA (e.g., a large
  apartment complex with its own ZIP shares a ZCTA with surrounding
  residential ZIPs)
- **One-to-many:** A ZIP can span two ZCTAs (boundary splits)
- **Null match:** P.O. Box ZIPs, unique (single-building) ZIPs, and
  military ZIPs have no corresponding ZCTA
- **Retired ZIPs:** USPS retires/reassigns ZIPs; Census ZCTAs lag by
  up to a decade

### Impact on This Experiment

1. **NFIP historical features (build_nfip_historical.py):**
   Claims whose `reportedZipCode` doesn't match any ZCTA in the
   assembled parquet are silently dropped from the historical
   frequency/severity aggregation. This biases `nfip_historical_frequency`
   downward for ZCTAs near ZIP/ZCTA boundary mismatches.

2. **Target variable (obs_nfip_event_claims):**
   If the target was built from the same raw OpenFEMA data with the same
   ZIP→ZCTA rename, the bias is symmetric (both features and targets
   undercount equally), which limits the damage to overall signal strength
   but does not introduce directional bias.

3. **R4 VLM arm:**
   VLMs receive text evidence built from the assembled parquet, which
   inherits any ZIP/ZCTA mismatch in the NFIP claim counts shown to
   the model.

### Magnitude Estimate

HUD USPS-ZCTA crosswalk (2024 Q4) shows:
- ~97% of residential delivery ZIPs map 1:1 to a ZCTA
- ~2% split across 2 ZCTAs (partial match)
- ~1% are P.O. Box / unique / military (no ZCTA match)

For the 5 s035 metro scenarios (all dense urban), the effective mismatch
rate is likely <3% of claims. This is below the noise floor for most
analyses, but it is not zero.

### Correct Fix (Not Applied)

Use the HUD USPS-ZCTA crosswalk to probabilistically allocate claims:

```python
# Correct approach (not yet implemented)
hud_xwalk = load_hud_usps_zcta_crosswalk()  # ZIP → ZCTA + allocation ratio
claims = claims.merge(hud_xwalk, left_on="reportedZipCode", right_on="zip")
claims["weighted_claim"] = claims["amountPaidOnBuildingClaim"] * claims["alloc_ratio"]
# Then aggregate weighted claims per zcta_id
```

### Current Mitigation

None. The experiment proceeds with ZIP ≈ ZCTA. Results should be
interpreted with this caveat. If any analysis shows anomalous NFIP
signal near ZCTA boundaries, this join mismatch is the first suspect.

### Resolution Path

- Download HUD USPS-ZCTA crosswalk (quarterly release)
- Add `fetch_hud_zip_zcta_crosswalk.py` to fetch pipeline
- Refactor `fetch_openfema_event.py` to keep `reportedZipCode` as-is
- Add probabilistic allocation in `build_nfip_historical.py`
- Re-run from build_nfip_historical forward
- Compare before/after to quantify actual impact

---

## C-002: Observation Unit Documentation Gap

**Severity:** Low (code is correct, documentation is incomplete)
**Affects:** Experiment reproducibility and reviewer comprehension

### The Problem

The observation unit `(zcta_id, event)` is implemented correctly in code
but never formally defined in EXPERIMENT_CONTRACT.yaml or any DOE
document. The term "unit" appears throughout without being defined as
ZCTA. A reviewer unfamiliar with the codebase cannot determine from
documentation alone:

- What a row in the assembled parquet represents
- That Houston has 396 rows (132 ZCTAs × 3 events), not 132
- That temporally-gated features vary per event for the same ZCTA
- That the primary key is `[zcta_id, event]`

### Resolution

Added `observation_unit` block to EXPERIMENT_CONTRACT.yaml v1.1.

---

## C-003: Fold Files Not Archived

**Severity:** High (operational risk, not analytical)
**Affects:** Reproducibility if R0 is accidentally re-run

### The Problem

`train_r0_baseline.py` writes `folds/{scenario}_folds.parquet` via raw
`s3.put_object` — not through `upload_json_result`, so there is **no
archive-before-overwrite** protection. If R0 is re-run after R1/R2 have
already consumed the folds, the fold assignments change silently and
the controlled comparison (same folds across arms) is broken.

### Mitigation

Operational discipline: do not re-run R0 after R1/R2 have started.
Document in CHECKLIST.md.

### Resolution Path

Route fold uploads through an archive-safe mechanism, or add an
"exists" guard that refuses to overwrite without `--force`.

---

## C-004: Henri DR=None Produces Structurally Zero NFIP Target

**Severity:** High
**Affects:** NYC scenario NFIP target (obs_nfip_event_claims), all phases R0-R3
**Discovered by:** Root-cause analysis of degenerate NYC R2 results (2026-06-06)
**GeoRSCT taxonomy:** GEO-C.3 (Spatial Missingness Bias) — non-random geographic
missingness shifts N_ceiling under imputation

### The Problem

In `build_event_dataset.py`, the NYC event_map entry for Henri 2021 has
`"dr": None`:

```python
"henri2021": {"dr": None, "storm_id": "AL082021",
              "peak_window": ("2021-08-21", "2021-08-22")}
```

When `dr` is None, `fetch_openfema_event.py` fetches zero NFIP claims for
that event. All 211 Henri ZCTAs get `obs_nfip_event_claims = 0` by
construction — not because there was no flooding, but because no federal
disaster was declared for Henri in NY.

### Why This Is Worse Than Missing Data

Unlike a random missing-at-random gap, DR=None produces **structurally zero
target** for half the dataset (211 of 422 rows). The model trains on:
- 211 rows where NFIP claims reflect real flood damage (Ida)
- 211 rows where NFIP claims are zero by bureaucratic construction (Henri)

This is contradictory training signal: ZCTAs that flooded during Henri
(confirmed by 311 reports and MRMS rainfall) show zero claims, teaching
the model that flooding features predict zero claims. The result is
degenerate R2 (-0.352 at R0 baseline) and complete R2 failure (n_folds=0)
at the temporal arm.

### Taxonomy Classification

This is a **data-pipeline collapse** (GEO-C family), specifically:

| Taxonomy Mode | Match | Signal |
|---------------|-------|--------|
| C.3 Spatial Missingness Bias | **Primary** | Non-random missingness (bureaucratic, not spatial) shifts N_ceiling for Henri rows |
| B.3 Crosswalk Gap | **Secondary** | DR=None is a crosswalk failure between FEMA disaster declarations and the event catalog |

The signature is distinctive: the missingness is **event-structured**, not
spatially random. All 211 Henri ZCTAs are affected identically. This makes
it invisible to per-ZCTA diagnostics (no spatial pattern) but visible in
leave-event-out cross-validation (Henri fold is structurally degenerate).

### Resolution

Added Sandy 2012 (DR-4085, 57,500 claims) and NYC Flood Sep 2023
(DR-4755, 1,698 claims) to the NYC scenario. NYC now has 4 events,
3 of which have real NFIP claims. Henri retains DR=None (correct — there
was no federal disaster declaration) but the model can now learn from
the other 3 events.

Post-fix R0 baseline: R2 = +0.025 (was -0.352). No longer degenerate,
but NFIP remains the hardest target. Henri fold excluded from NFIP
leave-event-out since there is no ground truth to predict.

### Diagnostic Checklist for Future Scenarios

Before adding a new event to any scenario, verify:

1. Does a FEMA disaster declaration exist for this event in this state?
2. If DR=None, is there an alternative target source (311, HWM)?
3. What fraction of the scenario's total rows will have DR=None?
4. If >25% of rows have structurally zero target, flag as GEO-C.3.

---

## C-005: MRMS Archive Temporal Coverage Boundary

**Severity:** Low (handled gracefully by build pipeline)
**Affects:** Sandy 2012 MRMS rainfall features
**Discovered by:** Fetch job failure (2026-06-06)

### The Problem

NOAA MRMS (Multi-Radar Multi-Sensor) archive does not reliably cover
events before late October 2012. Sandy 2012 (Oct 28-Nov 2) is right
at the operational boundary. The fetch job attempted 120 hourly files
and all 120 failed (0% success rate).

### Impact

`build_event_dataset.py` handles missing MRMS gracefully:
`obs_mrms_coverage_pct = 0.0` and `rainfall_total_mm = NaN` for Sandy
rows. These become valid feature values (zero coverage is informative),
not pipeline failures.

Updated `data_availability.json`: Sandy MRMS marked as `source_empty`
(was incorrectly `available`).

### Diagnostic Rule

For any event before 2014: MRMS may be unavailable. Check
`data_availability.json` temporal_coverage before assuming availability.
HRRR has the same issue (operational from Sep 2014). SMAP from Mar 2015.

---

## C-006: Pipeline Dependency Ordering for Regeneration

**Severity:** Medium (operational, produces stale artifacts if violated)
**Affects:** All downstream phases when a scenario's data changes
**Discovered by:** Control flow error during NYC event addition (2026-06-06)

### The Problem

When events are added to a scenario mid-experiment, the following
**must be regenerated in strict order**:

```
1. Fetch (parallel OK within fetch phase)
2. build_event_dataset (assembles base parquet with new events)
3. build_nfip_historical (reads assembled parquet for ZCTA lists)
4. R0 baseline (generates new folds for changed row count)
5. build_r1_features + build_r2_features (parallel OK)
6. R1 training + R2 training (sequential: R1 before R2)
7. diagnostics_r0 → diagnostics_r1 → diagnostics_r2 (sequential)
8. certificates_r0 → certificates_r1 → certificates_r2 (sequential,
   each requires corresponding diagnostics)
9. uplift_table (requires all diagnostics + certificates)
10. R3 pipeline (block tests → order robustness → admission →
    certified training → R3 money table)
```

**Violated in the initial NYC rebuild:** Certificates R1 were launched
before diagnostics R1 completed (consumed stale diagnostics from the
previous 2-event run). Uplift table was launched before certificates
were regenerated with fresh diagnostics.

### Rule

**Certificates must consume same-generation diagnostics.** Never launch
`certificates_r{k}` until `diagnostics_r{k}` from the current data
generation has completed. The readiness check will pass (file exists
on S3) but the file may be stale.

**Uplift table requires all 3 levels of diagnostics and certificates.**
The EXPERIMENT_CONTRACT.yaml `s3_artifacts` for `uplift_table` lists
all 6 files. The readiness checker validates existence, not freshness.

### Mitigation

Verification gate V4.11 (`event_cascade_regeneration`) was added to
EXPERIMENT_CONTRACT.yaml to enforce this ordering when events change.
Operational discipline: when regenerating after data changes, run the
full chain sequentially. Do not parallelize across dependency boundaries.

---

## MMAR Reviewer Blind Spot Log

Tracking cases where the multi-model adversarial review (6 LLMs) failed
to catch an issue that human review found.

| ID | Issue | Why LLMs Missed It |
|----|-------|--------------------|
| C-001 | ZIP ≠ ZCTA | All 6 models treated `reportedZipCode → zcta_id` rename as a cosmetic column rename. None had the domain knowledge that USPS ZIP codes and Census ZCTAs are different geographic systems with a many-to-many relationship. This is a geospatial domain fact, not a code pattern — LLMs pattern-match on code structure, not on the semantic validity of column renames. |

---

## Change Log

| Date | Caveat | Action |
|------|--------|--------|
| 2026-06-02 | C-001 | Documented ZIP ≠ ZCTA mismatch |
| 2026-06-02 | C-002 | Documented observation unit gap; added to contract |
| 2026-06-02 | C-003 | Documented fold archive risk |
| 2026-06-06 | C-004 | Documented Henri DR=None structural zero; mapped to GEO-C.3 |
| 2026-06-06 | C-005 | Documented MRMS temporal coverage boundary for Sandy 2012 |
| 2026-06-06 | C-006 | Documented pipeline dependency ordering for regeneration |
