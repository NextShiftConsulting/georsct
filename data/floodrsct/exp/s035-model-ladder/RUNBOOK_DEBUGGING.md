# S035 Debugging Runbook

**Purpose:** Systematic diagnosis of pipeline failures using the GeoRSCT
failure taxonomy (Table 1 of the GeoRSCT paper). Maps observable symptoms
to taxonomy modes, provides investigation steps, and documents the
regeneration protocol.

---

## 1. Failure Taxonomy Quick Reference

The GeoRSCT taxonomy organizes evaluation failures into 12 modes across
4 families, named by where the failure originates in the certificate
pipeline.

| Family | Name | Modes | Gate Response |
|--------|------|-------|---------------|
| GEO-A | Spatial-Dependence Collapse | A.1 Autocorrelation Leakage, A.2 Geographic Heterogeneity, A.3 Smooth-Map Illusion | REJECT, BLOCK, REPAIR |
| GEO-B | Aggregation Collapse | B.1 MAUP/Partition Drift, B.2 Scale Mismatch, B.3 Crosswalk Gap | RE_ENCODE, RE_ENCODE, REPAIR |
| GEO-C | Data-Pipeline Collapse | C.1 Vintage Drift, C.2 Projection/CRS Inconsistency, C.3 Spatial Missingness Bias | RE_ENCODE, REJECT, REPAIR |
| GEO-D | Substrate-Ceiling Collapse | D.1 Ceiling-Aggregation Drift, D.2 Ceiling-Induced Architecture Flattening, D.3 Interp/Extrap Mismatch | BLOCK, BLOCK, RE_ENCODE |

**Canonical source:** `rsct-governance/primary/theory/combine_georsct.pdf`,
Section 2, Table 1.

**Machine-readable source:** `specifications/georsct_taxonomy.json`
(uses older GCF naming; GCF-1.x maps to certificate-level failures,
not the geo-substrate taxonomy above).

---

## 2. Symptom-to-Taxonomy Decision Tree

Start here when a scenario or target produces unexpected results.

### Symptom: Negative R2 on a single target

```
Is R2 negative for ALL events in leave-event-out?
  YES -> Is the target structurally zero for any events?
    YES -> C.3 Spatial Missingness Bias (see C-004)
           Check: does event_map have dr=None?
           Check: is the target column all-zero for that event?
    NO  -> Is random-CV R2 >> blocked-CV R2?
      YES -> A.1 Autocorrelation Leakage
             Check: Moran's I on residuals
      NO  -> D.1 Ceiling-Aggregation Drift
             Check: N_ceiling for this target
  NO (negative for some events, positive for others) ->
    Is the negative fold an out-of-distribution event?
      YES -> D.3 Interp/Extrap Mismatch
             Check: signed-delta across protocols
      NO  -> A.2 Geographic Heterogeneity
             Check: residual region-variance structure
```

### Symptom: n_folds = 0 or degenerate training

```
Did the fold generator produce valid folds?
  NO  -> Check folds/{scenario}_folds.parquet row count
         Does it match the assembled parquet?
  YES -> Did training produce NaN/Inf metrics?
    YES -> Is the target constant within train or test split?
      YES -> C.3 Spatial Missingness Bias
             (structurally zero target in one fold)
      NO  -> Check for collinear features or all-NaN columns
    NO  -> Is n_folds < n_folds_requested?
      YES -> Check fold_metadata.spatial_blocked_folds
             Are any folds empty or near-empty?
             -> B.1 MAUP/Partition Drift (block merging removed folds)
```

### Symptom: Fetch job failure

```
Is the failure "ALL sources failed"?
  YES -> Is the event date before the source's operational start?
    YES -> Temporal coverage boundary (see C-005)
           Update data_availability.json: source_empty
    NO  -> Network/API issue. Retry once. Check CloudWatch.
  NO  -> Is it an authentication error?
    YES -> Check swarm_auth credentials. Check IAM role.
    NO  -> Read the full CloudWatch log. Check for OOM (exit 137).
```

### Symptom: Stale artifacts downstream

```
Did you add/remove events from a scenario?
  YES -> Full regeneration required (see Section 4)
         V4.11 event_cascade_regeneration applies
Did you change feature construction code?
  YES -> Regenerate from build_event_dataset forward
Did you change fold generation code?
  YES -> STOP. C-003 fold archive risk applies.
         Regeneration breaks controlled comparison.
```

---

## 3. Taxonomy-Guided Investigation Protocol

For each failure, follow this protocol before proposing a fix.

### Step 1: Classify the symptom

Map the observable symptom to a candidate taxonomy mode using the
decision tree above. Record the mode ID (e.g., C.3) and the
observable signal pattern.

### Step 2: Gather evidence

For each candidate mode, check the diagnostic tests listed in
Table 1 of the GeoRSCT paper:

| Mode | Diagnostic Tests |
|------|-----------------|
| A.1 | random-CV vs blocked-CV gap (both in R0/R1/R2 results) |
| A.2 | Moran's I on residuals (in diagnostics_r{k}.json) |
| A.3 | alpha vs spatial coherence (in certificates_r{k}.json) |
| B.1 | sigma inflation under re-aggregation |
| B.2 | feature-target resolution audit |
| B.3 | ZIP-ZCTA mismatch rate (see C-001) |
| C.1 | ACS/PLACES/TIGER vintage alignment |
| C.2 | CRS metadata check on all spatial inputs |
| C.3 | target column value_counts by event; DR declaration check |
| D.1 | per-task alpha vs N_ceiling correlation |
| D.2 | cross-family sigma within a task |
| D.3 | signed-delta across spatial_blocked vs leave_event_out |

### Step 3: Confirm or reject

The taxonomy mode is confirmed if:
- The diagnostic signal matches the expected pattern in Table 1
- The mode's gate disposition matches the observed certificate behavior
- Alternative modes are ruled out by contradictory evidence

### Step 4: Document

Add a caveat entry (C-00N) to CAVEATS.md with:
- Severity, affects, discovery method
- GeoRSCT taxonomy mode mapping
- The problem (what happened)
- The signal pattern (how it manifests in certificates)
- Resolution or mitigation

---

## 4. Regeneration Protocol (Event Addition/Removal)

When events are added to or removed from a scenario, the following
chain must execute in strict dependency order. **Do not parallelize
across dependency boundaries.**

```
Phase                           Depends On           Parallelizable?
-----------------------------------------------------------------------
1. Fetch (all sources)          nothing               YES (within fetch)
2. build_event_dataset          all fetches            NO
3. build_nfip_historical        build_event_dataset    NO
4. R0 baseline (generates       nfip_historical        NO
   new folds)
5a. build_r1_features           build_event_dataset    YES (5a || 5b)
5b. build_r2_features           build_event_dataset    YES (5a || 5b)
6a. R1 training                 R0 folds + r1_features NO (before R2)
6b. R2 training                 R0 folds + r2_features NO (after R1)
7a. diagnostics_r0              R0 results             YES (7a || 7b || 7c)
7b. diagnostics_r1              R1 results             YES (but see note)
7c. diagnostics_r2              R2 results             YES (but see note)
8a. certificates_r0             diagnostics_r0         NO (after 7a)
8b. certificates_r1             diagnostics_r1         NO (after 7b)
8c. certificates_r2             diagnostics_r2         NO (after 7c)
9.  gearbox_warmup              R0 (all scenarios)     NO
10. uplift_table                all diag + certs       NO
11. R3 feature registry         all diag + certs       NO
12a. R3 block tests (scenario)  registry               YES (per scenario)
12b. R3 order robustness        registry               YES (per scenario)
13. R3 block admission          12a + 12b (all)        NO
14. R3 certified training       admission              per scenario
15. R3 money table              R3 training (all)      NO
```

**Note on diagnostics parallelism:** Diagnostics at different levels
(r0, r1, r2) read independent inputs and write independent outputs.
They CAN run in parallel. But certificates_r{k} MUST wait for
diagnostics_r{k} from the same generation.

### Critical ordering violations to avoid

1. **Never launch certificates before its diagnostics completes.**
   The readiness checker validates file existence, not freshness.
   Stale diagnostics produce stale certificates.

2. **Never launch uplift_table before all 6 files (3 diagnostics +
   3 certificates) are from the current generation.** Mixing
   generations produces inconsistent quality signals.

3. **Never re-run R0 after R1/R2 have consumed folds (C-003).**
   Fold assignments change silently, breaking the controlled comparison.

4. **Never skip build_nfip_historical after build_event_dataset.**
   NFIP historical uses the assembled parquet for ZCTA lists.
   If the assembled parquet has new events, nfip_historical must
   regenerate to include those events' ZCTAs.

### Verification gate

V4.11 `event_cascade_regeneration` enforces this protocol:
- r2_supplement regenerated
- nfip_historical regenerated
- fold balance re-verified (<10% imbalance)
- all downstream certificates + models retrained
- change control entry in DOE_AMENDMENT

---

## 5. Incident Log

Record each debugging incident for pattern recognition.

### INC-001: NYC NFIP Degenerate (2026-06-06)

**Symptom:** NYC obs_nfip_event_claims R2 = -0.352 (R0), -0.080 (R1),
None/n_folds=0 (R2). All other scenarios healthy.

**Taxonomy mode:** GEO-C.3 (Spatial Missingness Bias)

**Root cause:** Henri 2021 event has `dr: None` in event_map. No FEMA
disaster declaration for Henri in NY. All 211 Henri ZCTAs get
obs_nfip_event_claims = 0 by construction. This is 50% of the dataset
with structurally zero target. The model learns contradictory signal:
features indicating flooding but target = 0.

**Why not caught earlier:** Henri has valid 311 and HWM data; only the
NFIP target is affected. The 311 target (AUC = 0.953) masked the
NFIP failure. Per-ZCTA diagnostics see no spatial pattern because the
missingness is event-structured, not spatially structured.

**Resolution:** Added Sandy 2012 (DR-4085) and NYC Flood Sep 2023
(DR-4755). NYC now has 4 events, 3 with real NFIP claims. Henri
retains DR=None (correct). Post-fix R0 R2 = +0.025.

**Control flow error during fix:** Certificates R1 launched before
fresh diagnostics R1 completed. Uplift table consumed stale
certificates. Required full regeneration of diagnostics -> certificates
-> uplift in correct order.

**Lessons:**
1. DR=None is a structural zero, not missing data. Flag at design time.
2. Leave-event-out is the only split that exposes event-structured
   missingness. Spatial-blocked CV will not catch it.
3. Pipeline regeneration after data changes must follow strict
   dependency order. Readiness checks validate existence, not freshness.
4. Adding events fixes the degenerate target but does not fix Henri.
   Henri NFIP fold is correctly excluded from evaluation, not patched.

**Caveats added:** C-004, C-005, C-006.

---

## 6. Interpreting Results After Event Addition

When events are added mid-experiment, results require careful framing.

### What to say

> Adding two NYC events restored evaluability for the temporal R2 arm,
> increasing valid folds from 0 to 3. However, R2 remains unstable
> under leave-event-out transfer: one event shows positive skill,
> while the mean remains negative, suggesting temporal features add
> signal only under event-specific conditions and may overfit under
> small-sample transfer.

### What not to say

- "R2 is fixed" (mean R2 is still negative)
- "R2 = 0.250 for Sandy" without noting it is 1 of 3 folds
- Treat LEO results as the primary benchmark (spatial-blocked is primary)
- Mix LEO and spatial-blocked language in the same table

### Fold count caveat

If the experiment specifies 5-fold spatial-blocked CV but only 3 folds
are valid for a target (because 1 event has DR=None and 1 event has
no HWM), report as "3 valid / 4 attempted" in LEO, and note that
spatial-blocked folds remain at 5 (all ZCTAs have features even if
some targets are zero). These are different splitting strategies
testing different things.

---

## 7. Pre-Flight Checklist for New Events

Before adding any event to a scenario:

- [ ] FEMA disaster declaration exists (DR number)?
- [ ] If DR=None: what fraction of scenario rows will have zero target?
- [ ] If >25%: flag as potential GEO-C.3; consider whether the event
      adds value or just dilutes signal
- [ ] MRMS archive covers the event date? (check data_availability.json)
- [ ] HURDAT2 track exists? (tropical events only)
- [ ] STN HWM deployment? (check USGS STN API)
- [ ] 311 data available? (NYC/Houston only)
- [ ] Tide gauge data available? (coastal events only)
- [ ] Update data_availability.json with actual availability
- [ ] Update event_map in build_event_dataset.py
- [ ] Update SCENARIO_EVENTS in build_nfip_historical.py
- [ ] Update ALL_EVENTS in launcher scripts
- [ ] Update paper event count and scenario table
- [ ] Run full regeneration protocol (Section 4)
