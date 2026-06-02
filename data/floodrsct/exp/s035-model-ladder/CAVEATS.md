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
