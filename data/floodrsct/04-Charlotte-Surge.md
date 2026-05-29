# 04 — Lee / Charlotte Coastal Surge Evacuation

## Scenario definition

```text
scenario_id:      lee_charlotte_surge_evacuation
display_name:     Lee / Charlotte Coastal Surge Evacuation
flood_archetype:  coastal_surge_evacuation_access
forecast_horizon: 72h or 48h
primary_decision: where to support evacuation routes, shelter staging,
                  transportation assistance, and coastal access monitoring
build_phase:      extension / phase 2
```

The operator question here is different from the first three scenarios:

> **Which ZCTAs may become evacuation- or access-fragile under coastal surge,
> low elevation, and route stress?**

This is not primarily about where water will collect. It is about whether
people and resources can move when the surge arrives.

**F40 is a non-scalar crisis vector.** ADR-033 is explicit. F40 emits five
independent components and does not rank ZCTAs. Downstream policy views may
order ZCTAs only if the ordering rule is explicit and labeled as
policy-conditioned.

---

## Function inventory

### F1 — Select scenario

```python
get_scenario_detail(
    scenario_id="lee_charlotte_surge_evacuation"
)
```

```sql
SELECT *
FROM scenarios
WHERE scenario_id = 'lee_charlotte_surge_evacuation';
```

Expected fields: `anchor_storm`, `region`, `flood_archetype`,
`primary_decision`, `build_phase`, `affected_zctas`, `peak_extent_km2`,
`fatality_count`, `total_loss_usd`.

---

### F2 — Select forecast snapshot

```python
get_scenario_snapshots(
    scenario_id="lee_charlotte_surge_evacuation"
)
```

```sql
SELECT scenario_id, snapshot_t
FROM scenario_snapshots
WHERE scenario_id = 'lee_charlotte_surge_evacuation'
ORDER BY snapshot_t DESC;
```

Recommended demo default: `snapshot_t = -72` (`forecast_horizon =
"72h_pre_landfall"`). Evacuation planning has to happen before roads and
shelters are stressed — the 72h story is stronger for this archetype.

---

### F18 — Scenario readiness

```python
check_scenario_readiness(
    scenario_id="lee_charlotte_surge_evacuation",
    snapshot_t=-72
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
    scenario_id="lee_charlotte_surge_evacuation",
    snapshot_t=-72,
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
WHERE c.scenario_id = 'lee_charlotte_surge_evacuation'
  AND c.snapshot_t = $1
  AND c.embedding_arm = 'graphsage_v1'
ORDER BY c.zcta_id;
```

**Do not default-sort by `kappa_compat`, `slosh_category`, or
`coastal_distance_m`.** Any of those would create a hidden scalar ranking.
Sorting belongs in a declared `policy_view_queue`.

---

### F4 — Select ZCTA / certificate walkthrough

```python
select_zcta(
    scenario_id="lee_charlotte_surge_evacuation",
    snapshot_t=-72,
    zcta_id="33931"
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
WHERE c.scenario_id = 'lee_charlotte_surge_evacuation'
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

ADR-034 is a hard cutover — no aliases.

---

### F37 — Vulnerability / population context

```python
get_vulnerability_context(
    scenario_id="lee_charlotte_surge_evacuation",
    zcta_id="33931"
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
    acs_age_65_plus,
    acs_vehicle_access
FROM read_parquet('s3://swarm-yrsn-datasets/georsct_table/v24.parquet')
WHERE zcta_id = '33931';
```

Two vulnerability fields are especially important for S4:

| Field | Why it matters |
|-------|----------------|
| `acs_vehicle_access` | Evacuation dependency — low access = route-assistance need |
| `acs_age_65_plus` | Mobility / shelter-support context under surge evacuation |

Output:

```json
{
  "population_context": "medium",
  "equity_context": "q3"
}
```

---

### F12 — Reliability action queue

```python
get_action_queue(
    scenario_id="lee_charlotte_surge_evacuation",
    snapshot_t=-72
)
```

```json
{
  "zcta_id": "33931",
  "reliability_action": "review",
  "reason_codes": ["WARN_INFORMATIONAL"]
}
```

Possible values: `trust` | `review` | `escalate` | `suppress`.

For S4, `review` or `escalate` should appear when evacuation-route status is
stale, surge category is high, or certificate confidence is borderline.

---

### F40 — Build resource priority vector

F40 returns **five independent components** — not a scalar score, not a rank.

```python
build_resource_priority_vector(
    scenario_id="lee_charlotte_surge_evacuation",
    snapshot_t=-72,
    zcta_id="33931"
)
```

```json
{
  "scenario_id": "lee_charlotte_surge_evacuation",
  "snapshot_t": -72,
  "zcta_id": "33931",
  "vector": {
    "risk_level": "high",
    "reliability_action": "review",
    "population_context": "medium",
    "access_fragility": "high",
    "equity_context": "q3"
  }
}
```

F40 must **not** return:

```json
{
  "resource_priority_score": 0.92,
  "priority_rank": 1
}
```

ADR-033 explicitly prohibits computing `resource_priority_score` in F40.

#### S4 F40 source map

| Component | Source function | Source field | Transform |
|-----------|----------------|--------------|-----------|
| `risk_level` | F4 | `certificate.public_decision` + `gate_result.active_metric` + `gate_result.active_value` | enum_remap |
| `reliability_action` | F12 | `action_queue[location_id].reliability_action` | passthrough |
| `population_context` | F37 | `vulnerability_context.population_density` | enum_remap |
| `access_fragility` | F4 | `infrastructure_evidence.slosh_category` + `evacuation_route_status` + `elevation_m_msl` + `coastal_distance_m` | scenario_discriminated_enum_remap |
| `equity_context` | F37 | `vulnerability_context.svi_quartile` | passthrough |

`access_fragility` represents route stress + exposure + geography, not surge
depth alone. Thresholds are scenario-configured, not hardcoded in F40.

| `evacuation_route_status` | `slosh_category` / elevation | `access_fragility` |
|---------------------------|------------------------------|--------------------|
| `closed` | any | `high` |
| `congested` | category >= 3 or low elevation | `high` |
| `congested` | low surge exposure | `medium` |
| `open` | category >= 3 + low elevation | `medium` or `high` |
| `open` | low surge exposure | `low` |
| `unknown` | any | `null` + `field_status` warning |

---

### F41 — Policy-view queue (optional)

Natural policy views for S4:

```text
life_safety_first
evacuation_route_first
shelter_staging_first
coastal_access_monitoring
mobility_support_first
equity_stabilization
```

```python
build_policy_view_queue(
    scenario_id="lee_charlotte_surge_evacuation",
    snapshot_t=-72,
    policy_view="evacuation_route_first"
)
```

Required persisted queue metadata:

```json
{
  "policy_view": "evacuation_route_first",
  "policy_view_version": "v1",
  "ordering_rule": "access_fragility DESC, reliability_action DESC, risk_level DESC",
  "source": "F40_VECTOR",
  "note": "Policy-conditioned projection. Not a universal priority score."
}
```

Example queue item:

```json
{
  "zcta_id": "33931",
  "ordinal_position": 1,
  "recommended_action": "verify evacuation-route status and stage traffic-support resources",
  "action_rationale": [
    "access_fragility=high",
    "risk_level=high",
    "reliability_action=review",
    "evacuation_route_status=congested"
  ]
}
```

**F40 does not rank. F41 / Crisis Orchestrator may queue under a declared
policy view.**

---

## Infrastructure evidence

The key S4 infrastructure fields:

```sql
SELECT
    slosh_category,
    elevation_m_msl,
    evacuation_route_status,
    coastal_distance_m
FROM infrastructure_evidence
WHERE scenario_id = 'lee_charlotte_surge_evacuation'
  AND certificate_id = $1;
```

| Field | Interpretation |
|-------|----------------|
| `slosh_category` | Surge exposure band |
| `elevation_m_msl` | Low-elevation physical exposure |
| `evacuation_route_status` | `open` \| `congested` \| `closed` \| `unknown` |
| `coastal_distance_m` | Coastal proximity / surge-relevance proxy |

These are scenario-discriminated in the `infrastructure_evidence` schema —
same canonical cert structure, different evidence columns per scenario.

---

## Field status for partial evidence

```json
{
  "access_fragility": null,
  "field_status": {
    "access_fragility": {
      "status": "missing_source_data",
      "missing_reason": "Evacuation route status unavailable for this surge evacuation snapshot.",
      "expected_available_version": "v24.003"
    }
  }
}
```

Dashboard renders `Missing source data`, not `N/A`. Per ADR-033 D5.

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

slosh_category
elevation_m_msl
evacuation_route_status
coastal_distance_m

nfip historical loss / event features
noaa storm-event history
```

---

## Demo flow

```text
1.  Operator selects Lee / Charlotte Coastal Surge Evacuation.
2.  App loads the 72h pre-landfall snapshot.
3.  F18 checks readiness.
4.  F3 lists ZCTAs (sorted by zcta_id; no default ranking).
5.  Operator selects a coastal or evacuation-sensitive ZCTA.
6.  F4 shows certificate + SLOSH / elevation / route evidence.
7.  F37 shows population / equity / mobility context.
8.  F40 shows five-component crisis vector.
9.  F41 optionally shows evacuation_route_first or shelter_staging_first queue.
10. Dashboard note: the queue is policy-conditioned, not a universal priority
    score.
```
