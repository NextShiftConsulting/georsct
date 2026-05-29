# 05 — NYC / NJ Urban Cloudburst

## Scenario definition

```text
scenario_id:      nyc_nj_urban_cloudburst
display_name:     NYC / NJ Urban Cloudburst
flood_archetype:  dense_urban_cloudburst_drainage
forecast_horizon: 6h or 24h
primary_decision: where to warn, inspect drainage/sewer stress, protect
                  basement residents, monitor subway-adjacent access, and
                  stage rapid-response crews
build_phase:      phase_2 / extension
```

The operator question here is different from all four prior scenarios:

> **Which neighborhoods may become dangerous quickly under short-duration
> extreme rainfall overwhelming dense urban drainage, basements, subway
> access, sewer sheds, and street-level mobility?**

This is not bayou overflow, levee/pump basin failure, desert wash flash
flooding, or coastal surge evacuation. The risk is structural: rainfall rates
exceed drainage capacity in a dense urban environment.

Recommended demo default: `snapshot_t = -24` or `snapshot_t = -6`. For a
cloudburst, 24h or 6h is more credible than 72h. The strongest operator story
is short-fuse: which neighborhoods may become dangerous quickly?

**F40 is a non-scalar crisis vector.** ADR-033 is explicit. F40 emits five
independent components and does not rank ZCTAs.

**`access_fragility` is `null` in v24.002.** The required subway/sewer-shed
mapping is not available until v24.003. The field must render as
`Missing source data`, not `N/A`.

---

## Function inventory

### F1 — Select scenario

```python
get_scenario_detail(
    scenario_id="nyc_nj_urban_cloudburst"
)
```

```sql
SELECT *
FROM scenarios
WHERE scenario_id = 'nyc_nj_urban_cloudburst';
```

Expected fields: `anchor_storm`, `region`, `flood_archetype`,
`primary_decision`, `build_phase`, `affected_zctas`, `peak_extent_km2`,
`total_loss_usd`, `fatality_count`.

---

### F2 — Select forecast snapshot

```python
get_scenario_snapshots(
    scenario_id="nyc_nj_urban_cloudburst"
)
```

```sql
SELECT scenario_id, snapshot_t
FROM scenario_snapshots
WHERE scenario_id = 'nyc_nj_urban_cloudburst'
ORDER BY snapshot_t DESC;
```

Supported snapshots: `snapshot_t = -24`, `-6`, `0`.

---

### F18 — Scenario readiness

```python
check_scenario_readiness(
    scenario_id="nyc_nj_urban_cloudburst",
    snapshot_t=-24
)
```

| Status | Behavior |
|--------|----------|
| `ready` | F40 executes |
| `limited` | F40 executes with `field_status` warnings |
| `unreliable` | F40 returns `409 SCENARIO_UNREADY` |

ADR-033 requires scenario-level unreliability to fail closed with
`409 SCENARIO_UNREADY` and structured recovery guidance — not a vector that
looks valid but is not.

---

### F3 — List ZCTAs

```python
list_zctas(
    scenario_id="nyc_nj_urban_cloudburst",
    snapshot_t=-24,
    embedding_arm="graphsage_v1"
)
```

```sql
SELECT
    c.zcta_id,
    c.R,
    c.S_sup,
    c.N,
    c.alpha,
    c.kappa_compat,
    c.sigma,
    c.N_ceiling,
    c.public_decision,
    c.gate_reached,
    c.gate_reason
FROM certificates c
WHERE c.scenario_id = 'nyc_nj_urban_cloudburst'
  AND c.snapshot_t = $1
  AND c.embedding_arm = 'graphsage_v1'
ORDER BY c.zcta_id;
```

**Do not default-sort by `kappa_compat`, `basement_apartment_count`, or
subway exposure.** All of those would create a hidden scalar ranking. Sorting
belongs in a declared `policy_view_queue`.

---

### F4 — Select ZCTA / certificate walkthrough

```python
select_zcta(
    scenario_id="nyc_nj_urban_cloudburst",
    snapshot_t=-24,
    zcta_id="11201"
)
```

```sql
SELECT
    c.*,
    ie.*,
    array_agg(rc.reason_code ORDER BY rc.sort_order) AS reason_codes
FROM certificates c
JOIN infrastructure_evidence ie
  ON c.certificate_id = ie.certificate_id
LEFT JOIN certificate_reason_codes rc
  ON c.certificate_id = rc.certificate_id
WHERE c.scenario_id = 'nyc_nj_urban_cloudburst'
  AND c.snapshot_t = $1
  AND c.zcta_id = $2
GROUP BY c.certificate_id, ie.certificate_id;
```

Certificate fields use **ADR-034 names**:

| Use | Do not use |
|-----|------------|
| `kappa_compat` | `kappa` |
| `kappa_modal_min` | `kappa_gate` |
| `public_decision` | `decision` |
| `public_decision_source` | — |
| `gate_reason` | `reason` |
| `reason_codes` | — |

ADR-034 is a hard cutover — no aliases, no compatibility shims.

---

### F37 — Vulnerability / population context

```python
get_vulnerability_context(
    scenario_id="nyc_nj_urban_cloudburst",
    zcta_id="11201"
)
```

```sql
SELECT
    zcta_id,
    acs_population_density,
    svi_overall,
    svi_theme_socioeconomic,
    svi_theme_household,
    svi_theme_transportation,
    acs_vehicle_access,
    acs_age_65_plus
FROM read_parquet('s3://swarm-yrsn-datasets/georsct_table/v24.parquet')
WHERE zcta_id = '11201';
```

The most important vulnerability themes for S5:

| Field | Why it matters |
|-------|----------------|
| `population_density` | Dense areas have more basement residents and transit-dependent people exposed simultaneously |
| `svi_theme_household` | Crowded housing and below-grade residences |
| `svi_theme_transportation` | Transit dependence; subway flooding cuts mobility |
| `acs_vehicle_access` | Low vehicle access in NYC means subway disruption is full isolation |
| `acs_age_65_plus` | Mobility and basement-evacuation context |

Output:

```json
{
  "population_context": "high",
  "equity_context": "q4"
}
```

---

### F12 — Reliability action queue

```python
get_action_queue(
    scenario_id="nyc_nj_urban_cloudburst",
    snapshot_t=-24
)
```

```json
{
  "zcta_id": "11201",
  "reliability_action": "review",
  "reason_codes": ["WARN_INFORMATIONAL"]
}
```

Possible values: `trust` | `review` | `escalate` | `suppress`.

For S5, `review` or `escalate` is especially appropriate when subway/sewer-shed
data is missing or stale. Missing `access_fragility` does not suppress the
other four components.

---

### F40 — Build resource priority vector

F40 returns **five independent components** — not a scalar score, not a rank.

```python
build_resource_priority_vector(
    scenario_id="nyc_nj_urban_cloudburst",
    snapshot_t=-24,
    zcta_id="11201"
)
```

In v24.002, `access_fragility` is null with structured field status:

```json
{
  "scenario_id": "nyc_nj_urban_cloudburst",
  "snapshot_t": -24,
  "zcta_id": "11201",
  "vector": {
    "risk_level": "high",
    "reliability_action": "review",
    "population_context": "high",
    "access_fragility": null,
    "equity_context": "q4"
  },
  "field_status": {
    "access_fragility": {
      "status": "missing_source_data",
      "missing_reason": "S5 requires subway station and sewer-shed load mapping not available in v24.002.",
      "expected_available_version": "v24.003"
    }
  }
}
```

Dashboard must render `Missing source data`, not `N/A`. Per ADR-033 D5.

F40 must **not** return:

```json
{
  "resource_priority_score": 0.89,
  "priority_rank": 3
}
```

ADR-033 forbids this.

#### S5 F40 source map

| Component | Source function | Source field | Transform |
|-----------|----------------|--------------|-----------|
| `risk_level` | F4 | `certificate.public_decision` + `gate_result.active_metric` + `gate_result.active_value` | enum_remap |
| `reliability_action` | F12 | `action_queue[location_id].reliability_action` | passthrough |
| `population_context` | F37 | `vulnerability_context.population_density` | enum_remap |
| `access_fragility` | F4 | `infrastructure_evidence.subway_station_ids` + `sewer_shed_id` + `sewer_shed_load` + `basement_apartment_count` | scenario_discriminated_enum_remap |
| `equity_context` | F37 | `vulnerability_context.svi_quartile` | passthrough |

The source map exists in v24.002. The value is `null` because `sewer_shed_load`
is not yet available. When v24.003 lands, the same map produces a real value
with no schema change.

#### Future v24.003 remap

```text
sewer_shed_load = exceeded + basement_apartment_count high
  → access_fragility = high

sewer_shed_load = stressed + subway_station_ids present
  → access_fragility = medium or high

sewer_shed_load = nominal + low basement exposure
  → access_fragility = low

sewer_shed_load = unknown
  → access_fragility = null + field_status warning
```

Keep the mapping in scenario config. Do not hardcode it inside F40.

---

### F41 — Policy-view queue (optional)

Natural policy views for S5:

```text
life_safety_first
basement_flooding_first
subway_access_monitoring
sewer_shed_stress_first
rapid_response_staging
equity_stabilization
```

```python
build_policy_view_queue(
    scenario_id="nyc_nj_urban_cloudburst",
    snapshot_t=-24,
    policy_view="basement_flooding_first"
)
```

Required persisted queue metadata:

```json
{
  "policy_view": "basement_flooding_first",
  "policy_view_version": "v1",
  "ordering_rule": "population_context DESC, equity_context DESC, risk_level DESC, reliability_action DESC",
  "source": "F40_VECTOR",
  "note": "Policy-conditioned projection. Not a universal priority score."
}
```

In v24.002, do not include `access_fragility` in the ordering rule for S5 —
it is missing-source and ordering by a null column would be undefined. Once
v24.003 lands, `sewer_shed_stress_first` can safely include it.

Example queue item:

```json
{
  "zcta_id": "11201",
  "ordinal_position": 1,
  "recommended_action": "send basement-flood warning and inspect known drainage trouble spots",
  "action_rationale": [
    "risk_level=high",
    "population_context=high",
    "equity_context=q4",
    "access_fragility=missing_source_data"
  ]
}
```

`access_fragility=missing_source_data` appears in `action_rationale` as an
explicit data-quality signal, not silently dropped.

**F40 does not rank. F41 / Crisis Orchestrator may queue under a declared
policy view.**

---

## Infrastructure evidence

The key S5 infrastructure fields:

```sql
SELECT
    sewer_shed_id,
    basement_apartment_count,
    subway_station_ids,
    impervious_surface_pct
FROM infrastructure_evidence
WHERE scenario_id = 'nyc_nj_urban_cloudburst'
  AND certificate_id = $1;
```

| Field | Interpretation |
|-------|----------------|
| `sewer_shed_id` | Drainage / sewer-service exposure |
| `basement_apartment_count` | Below-grade residential vulnerability |
| `subway_station_ids` | Transit-access / underground-infrastructure exposure |
| `impervious_surface_pct` | Runoff amplification (shared with S1 Houston) |

`sewer_shed_id` and `subway_station_ids` are S5-specific. `impervious_surface_pct`
is the one field shared with Houston; the interpretation is the same but the
downstream consequence is different (sewer overflow vs. bayou backflow).

---

## Minimum viable field list

```text
zcta_id, scenario_id, snapshot_t
R, S_sup, N, alpha
kappa_compat, sigma, N_ceiling
public_decision, gate_reached, gate_reason, reason_codes, policy_id

population_density
svi_overall, svi_quartile
vehicle_access
age_65_plus

sewer_shed_id
basement_apartment_count
subway_station_ids
impervious_surface_pct

nfip historical loss / event features
noaa storm-event history
```

---

## Demo flow

```text
1.  Operator selects NYC / NJ Urban Cloudburst.
2.  App loads the 24h or 6h pre-event snapshot.
3.  F18 checks readiness.
4.  F3 lists ZCTAs (sorted by zcta_id; no default ranking).
5.  Operator selects a dense urban ZCTA.
6.  F4 shows certificate + sewer/subway/basement evidence.
7.  F37 shows population / equity / mobility context.
8.  F40 shows five-component crisis vector.
9.  access_fragility renders as "Missing source data" until v24.003.
10. F41 optionally shows basement_flooding_first queue, with
    access_fragility excluded from ordering in v24.002.
```

S5 proves that Floodcaster handles **dense urban compound infrastructure risk**
and can honestly expose missing source data instead of fabricating a
confidence-looking value. The `access_fragility = null` case is not a
deficiency to hide — it is the correct behavior, and the dashboard should
surface it as such.
