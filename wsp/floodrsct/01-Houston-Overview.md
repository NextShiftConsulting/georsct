# 01 — Harris County / Houston Urban Flood

## Scenario definition

```text
scenario_id:      harris_houston_urban
display_name:     Harris County / Houston Urban Flood
flood_archetype:  urban_bayou_drainage_flood
forecast_horizon: 72h
primary_decision: where to pre-position crews, high-water vehicles,
                  shelter support, and drainage monitoring
build_phase:      core
```

This is the **core demo scenario**. It supports the full 72-hour pre-flood story:
forecast risk, infrastructure fragility, vulnerable populations, certificate
trust, and downstream resource orchestration.

**F40 is a non-scalar crisis vector.** ADR-033 is explicit: F40 emits five
independent components and does not rank ZCTAs. Downstream policy-view
orchestration (F41) may rank only if the policy view and ordering rule are
declared.

---

## Function inventory

### F1 — Select scenario

Choose the Houston scenario and load metadata.

```python
get_scenario_catalog()
get_scenario_detail(scenario_id="harris_houston_urban")
```

```sql
SELECT *
FROM scenarios
WHERE scenario_id = 'harris_houston_urban';
```

Expected fields: `anchor_storm`, `region`, `flood_archetype`,
`primary_decision`, `build_phase`, `total_nfip_claims`, `total_loss_usd`,
`affected_zctas`.

---

### F2 — Select forecast snapshot

Choose the time slice: `72h`, `48h`, `24h`, or `0h`.

```python
get_scenario_snapshots(scenario_id)
select_snapshot(scenario_id, snapshot_t)
```

```sql
SELECT scenario_id, snapshot_t
FROM scenario_snapshots
WHERE scenario_id = 'harris_houston_urban'
ORDER BY snapshot_t DESC;
```

Demo default: `snapshot_t = -72` (`forecast_horizon = "72h_pre_flood"`).

---

### F18 — Scenario readiness check

Determines whether S1 can safely produce F40 vectors.

```python
check_scenario_readiness(
    scenario_id="harris_houston_urban",
    snapshot_t=-72
)
```

| Status | Behavior |
|--------|----------|
| `ready` | F40 executes |
| `limited` | F40 executes with `field_status` warnings |
| `unreliable` | F40 returns `409 SCENARIO_UNREADY` |

ADR-033 requires `409 SCENARIO_UNREADY` for scenario-level unreliability
(not a generic 400).

Checks:
- `FloodcasterETags` / `FloodcasterCerts` source_etags
- Live sensor freshness
- Required S3 parquet version

---

### F3 — List ZCTAs

List all Houston-area ZCTAs in scope.

```python
list_zctas(
    scenario_id="harris_houston_urban",
    snapshot_t=-72,
    embedding_arm="graphsage_v1"
)
```

**Do not sort by `kappa_compat` by default** — that recreates a scalar ranking.
Default sort is `zcta_id`.

```sql
SELECT
    c.zcta_id,
    c.R,
    c.S_sup,
    c.N,
    c.alpha,
    c.kappa_compat,
    c.sigma,
    c.task_residual_floor,
    c.public_decision,
    c.gate_reached,
    c.gate_reason
FROM certificates c
WHERE c.scenario_id = 'harris_houston_urban'
  AND c.snapshot_t = $1
  AND c.embedding_arm = 'graphsage_v1'
ORDER BY c.zcta_id;
```

If the operator selects a policy-view queue, sort by `q.ordinal_position`
instead — not by certificate metrics.

---

### F4 — Select ZCTA / certificate walkthrough

Inspect a single ZCTA.

```python
select_zcta(
    scenario_id="harris_houston_urban",
    snapshot_t=-72,
    zcta_id="77011"
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
WHERE c.scenario_id = 'harris_houston_urban'
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

ADR-034 is a hard cutover — the retired identifiers are forbidden.

---

### F37 — Vulnerability / population context

Convert ACS/SVI/population fields into F40 context.

```python
get_vulnerability_context(
    scenario_id="harris_houston_urban",
    zcta_id="77011"
)
```

DuckDB query against S3:

```sql
SELECT
    zcta_id,
    acs_population_density,
    svi_overall,
    svi_theme_socioeconomic,
    svi_theme_household,
    svi_theme_transportation
FROM read_parquet('s3://swarm-yrsn-datasets/georsct_table/v24.parquet')
WHERE zcta_id = '77011';
```

Output:

```json
{
  "population_context": "high",
  "equity_context": "q4"
}
```

---

### F12 — Reliability action queue

Transform certificate/gate state into the `reliability_action` used by F40.

```python
get_action_queue(
    scenario_id="harris_houston_urban",
    snapshot_t=-72
)
```

```json
{
  "zcta_id": "77011",
  "reliability_action": "review",
  "reason_codes": ["WARN_INFORMATIONAL"]
}
```

Possible values: `trust` | `review` | `escalate` | `suppress`.

---

### F40 — Build resource priority vector

F40 returns **five independent components** — not a scalar score, not a rank.

```json
{
  "scenario_id": "harris_houston_urban",
  "snapshot_t": -72,
  "zcta_id": "77011",
  "vector": {
    "risk_level": "high",
    "reliability_action": "review",
    "population_context": "high",
    "access_fragility": "high",
    "equity_context": "q4"
  }
}
```

F40 must **not** return:

```json
{
  "resource_priority_score": 0.83,
  "priority_rank": 4
}
```

#### Houston F40 source map

| Component | Source function | Source field | Transform |
|-----------|----------------|--------------|-----------|
| `risk_level` | F4 | `certificate.public_decision` + `certificate.kappa_compat` or `gate_result.active_value` | enum_remap |
| `reliability_action` | F12 | `action_queue[location_id].decision` | passthrough |
| `population_context` | F37 | `vulnerability_context.population_density` | enum_remap |
| `access_fragility` | F4 | `infrastructure_evidence.drainage_capacity_status` + `impervious_surface_pct` | scenario_discriminated_enum_remap |
| `equity_context` | F37 | `vulnerability_context.svi_quartile` | passthrough |

`access_fragility` is where Houston-specific logic lives:

| `drainage_capacity_status` | `access_fragility` |
|----------------------------|--------------------|
| `exceeded` | `high` |
| `stressed` | `medium` |
| `nominal` | `low` |
| `unknown` | `null` + `field_status` warning |

---

### F41 — Policy-view queue (optional)

Ranking is only allowed here, under a declared policy view.

```python
build_policy_view_queue(
    scenario_id="harris_houston_urban",
    snapshot_t=-72,
    policy_view="life_safety_first"
)
```

Required persisted fields:

```text
policy_view
policy_view_version
ordering_rule
source = F40_VECTOR
note = Policy-conditioned projection. Not a universal priority score.
```

Example queue item:

```json
{
  "policy_view": "life_safety_first",
  "zcta_id": "77011",
  "ordinal_position": 1,
  "recommended_action": "preposition high-water vehicle",
  "action_rationale": [
    "risk_level=high",
    "reliability_action=review",
    "access_fragility=high",
    "population_context=high"
  ]
}
```

**F40 does not rank. F41 / Crisis Orchestrator may queue under a declared
policy view.**

---

## Data inputs

### Static S3 / Parquet

```text
s3://swarm-yrsn-datasets/georsct_table/v24.parquet
s3://swarm-yrsn-datasets/noaa_storm_events_long/v1.parquet
s3://swarm-yrsn-datasets/oof/graphsage_v1.parquet
s3://swarm-yrsn-datasets/ceilings/{target}_{protocol}_v1.parquet
```

Relevant feature groups: `acs_*`, `svi_*`, `twi_*`, `nfip_*`, `noaa_*`,
`targets`.

### Houston infrastructure evidence

```sql
SELECT *
FROM infrastructure_evidence
WHERE scenario_id = 'harris_houston_urban'
  AND certificate_id = $certificate_id;
```

| Field | Interpretation |
|-------|----------------|
| `bayou_segment_id` | Nearby bayou / channel exposure |
| `drainage_district_id` | Drainage authority or service area |
| `impervious_surface_pct` | Urban runoff amplification |
| `drainage_capacity_status` | `nominal` \| `stressed` \| `exceeded` \| `unknown` |

These fields feed F40's `access_fragility` component.

---

## Minimum viable field list

```text
zcta_id, scenario_id, snapshot_t
R, S_sup, N, alpha
kappa_compat, sigma, task_residual_floor
public_decision, gate_reached, gate_reason, reason_codes, policy_id

population_density
svi_overall, svi_quartile
impervious_surface_pct
drainage_capacity_status
bayou_segment_id
drainage_district_id
nfip historical loss / event features
noaa storm-event history
```

---

## Demo flow

```text
1.  Operator selects Harris County / Houston Urban Flood.
2.  App loads 72h pre-flood snapshot.
3.  F18 confirms readiness.
4.  F3 lists ZCTAs (sorted by zcta_id; no default ranking).
5.  Operator selects a ZCTA.
6.  F4 shows certificate walkthrough (ADR-034 field names).
7.  F37 shows population / equity context.
8.  F40 shows five-component crisis vector.
9.  Optional F41 shows policy-view queue for life_safety_first.
10. Dashboard note: this is not a universal scalar priority score.
```
