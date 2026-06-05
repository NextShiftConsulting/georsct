# 02 — New Orleans / Jefferson Protected Basin

## Scenario definition

```text
scenario_id:      orleans_jefferson_protected_basin
display_name:     New Orleans / Jefferson Protected Basin Flood
flood_archetype:  protected_basin_pump_levee_flood
forecast_horizon: 72h
primary_decision: where to monitor pump/levee stress, stage crews,
                  shelter support, and evacuation support
build_phase:      core_or_extension_candidate
```

This is the right second scenario because it is not Houston again. Houston
is urban bayou/drainage flooding. New Orleans adds **protected-basin failure
logic**: levees, pumps, canals, subsidence, and protected low-lying
neighborhoods.

Same architecture as Houston: S3 Parquet for static features, DynamoDB for
operational cert/cache/audit, DuckDB for session queries, Redis for hot reads.

**F40 is a non-scalar crisis vector.** ADR-033 is explicit. F40 emits five
independent components and does not rank ZCTAs. Downstream policy-view
orchestration (F41) may rank only if the policy view and ordering rule are
declared.

---

## Function inventory

### F1 — Select scenario

```python
get_scenario_detail(
    scenario_id="orleans_jefferson_protected_basin"
)
```

```sql
SELECT *
FROM scenarios
WHERE scenario_id = 'orleans_jefferson_protected_basin';
```

Expected fields: `anchor_storm`, `region`, `flood_archetype`,
`primary_decision`, `build_phase`, `affected_zctas`, `total_loss_usd`,
`fatality_count`.

---

### F2 — Select forecast snapshot

```python
get_scenario_snapshots(
    scenario_id="orleans_jefferson_protected_basin"
)
```

```sql
SELECT scenario_id, snapshot_t
FROM scenario_snapshots
WHERE scenario_id = 'orleans_jefferson_protected_basin'
ORDER BY snapshot_t DESC;
```

Demo default: `snapshot_t = -72` (`forecast_horizon = "72h_pre_flood"`).

---

### F18 — Scenario readiness

```python
check_scenario_readiness(
    scenario_id="orleans_jefferson_protected_basin",
    snapshot_t=-72
)
```

| Status | Behavior |
|--------|----------|
| `ready` | F40 executes |
| `limited` | F40 executes with `field_status` warnings |
| `unreliable` | F40 returns `409 SCENARIO_UNREADY` |

ADR-033 requires `409 SCENARIO_UNREADY` with structured recovery guidance
when scenario-level readiness is unreliable (not a generic 400).

---

### F3 — List ZCTAs

```python
list_zctas(
    scenario_id="orleans_jefferson_protected_basin",
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
    c.task_residual_floor,
    c.public_decision,
    c.gate_reached,
    c.gate_reason
FROM certificates c
WHERE c.scenario_id = 'orleans_jefferson_protected_basin'
  AND c.snapshot_t = $1
  AND c.embedding_arm = 'graphsage_v1'
ORDER BY c.zcta_id;
```

**Do not sort by `kappa_compat` by default** — that recreates a scalar
ranking. Ranking belongs only in a named downstream `policy_view_queue`.

---

### F4 — Select ZCTA / certificate walkthrough

```python
select_zcta(
    scenario_id="orleans_jefferson_protected_basin",
    snapshot_t=-72,
    zcta_id="70117"
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
WHERE c.scenario_id = 'orleans_jefferson_protected_basin'
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

```python
get_vulnerability_context(
    scenario_id="orleans_jefferson_protected_basin",
    zcta_id="70117"
)
```

```sql
SELECT
    zcta_id,
    acs_population_density,
    svi_overall,
    svi_theme_socioeconomic,
    svi_theme_household,
    svi_theme_transportation
FROM read_parquet('s3://swarm-yrsn-datasets/georsct_table/v24.parquet')
WHERE zcta_id = '70117';
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

```python
get_action_queue(
    scenario_id="orleans_jefferson_protected_basin",
    snapshot_t=-72
)
```

```json
{
  "zcta_id": "70117",
  "reliability_action": "review",
  "reason_codes": ["WARN_INFORMATIONAL"]
}
```

Possible values: `trust` | `review` | `escalate` | `suppress`.

For S2, `review` or `escalate` is especially important when pump/levee
evidence is incomplete or when certificate confidence is borderline.

---

### F40 — Build resource priority vector

F40 returns **five independent components** — not a scalar score, not a rank.

```python
build_resource_priority_vector(
    scenario_id="orleans_jefferson_protected_basin",
    snapshot_t=-72,
    zcta_id="70117"
)
```

```json
{
  "scenario_id": "orleans_jefferson_protected_basin",
  "snapshot_t": -72,
  "zcta_id": "70117",
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
  "resource_priority_score": 0.91,
  "priority_rank": 1
}
```

#### S2 F40 source map

| Component | Source function | Source field | Transform |
|-----------|----------------|--------------|-----------|
| `risk_level` | F4 | `certificate.public_decision` + `certificate.kappa_compat` or `gate_result.active_value` | enum_remap |
| `reliability_action` | F12 | `action_queue[location_id].decision` | passthrough |
| `population_context` | F37 | `vulnerability_context.population_density` | enum_remap |
| `access_fragility` | F4 | `infrastructure_evidence.pump_station_status` + `canal_proximity_m` + `subsidence_rate_mm_yr` | scenario_discriminated_enum_remap |
| `equity_context` | F37 | `vulnerability_context.svi_quartile` | passthrough |

`access_fragility` is driven by protected-basin infrastructure evidence, not
generic road access. Thresholds are scenario-configured, not hardcoded in F40.

| `pump_station_status` | `access_fragility` |
|-----------------------|--------------------|
| contains `failed` | `high` |
| contains `degraded` | `medium` |
| all `operational` | `low` |
| `unknown` / missing | `null` + `field_status` warning |

---

### F41 — Policy-view queue (optional)

Natural policy views for S2:

```text
life_safety_first
pump_failure_containment
protected_basin_monitoring
shelter_and_transport_support
equity_stabilization
```

```python
build_policy_view_queue(
    scenario_id="orleans_jefferson_protected_basin",
    snapshot_t=-72,
    policy_view="pump_failure_containment"
)
```

Required persisted queue metadata:

```json
{
  "policy_view": "pump_failure_containment",
  "policy_view_version": "v1",
  "ordering_rule": "access_fragility DESC, reliability_action DESC, risk_level DESC",
  "source": "F40_VECTOR",
  "note": "Policy-conditioned projection. Not a universal priority score."
}
```

Example queue item:

```json
{
  "zcta_id": "70117",
  "ordinal_position": 1,
  "recommended_action": "verify pump station status and stage drainage crew",
  "action_rationale": [
    "access_fragility=high",
    "risk_level=high",
    "reliability_action=review",
    "equity_context=q4"
  ]
}
```

**F40 does not rank. F41 / Crisis Orchestrator may queue under a declared
policy view.**

---

## Infrastructure evidence

The key S2 infrastructure fields:

```sql
SELECT
    levee_segment_id,
    pump_station_ids,
    pump_station_status,
    subsidence_rate_mm_yr,
    canal_proximity_m
FROM infrastructure_evidence
WHERE scenario_id = 'orleans_jefferson_protected_basin'
  AND certificate_id = $1;
```

| Field | Interpretation |
|-------|----------------|
| `levee_segment_id` | Protection boundary exposure |
| `pump_station_ids` | Drainage dependency |
| `pump_station_status` | Operational fragility |
| `subsidence_rate_mm_yr` | Long-term basin vulnerability |
| `canal_proximity_m` | Local canal / drainage exposure |

These are scenario-discriminated in the `infrastructure_evidence` schema.
They feed F40's `access_fragility` component.

---

## Field status for partial evidence

S2 is likely to have incomplete infrastructure evidence. Use structured
`field_status`, not silent nulls.

```json
{
  "access_fragility": null,
  "field_status": {
    "access_fragility": {
      "status": "missing_source_data",
      "missing_reason": "Pump station status unavailable for this protected-basin snapshot.",
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
kappa_compat, sigma, task_residual_floor
public_decision, gate_reached, gate_reason, reason_codes, policy_id

population_density
svi_overall, svi_quartile

levee_segment_id
pump_station_ids
pump_station_status
subsidence_rate_mm_yr
canal_proximity_m

nfip historical loss / event features
noaa storm-event history
```

---

## Demo flow

```text
1.  Operator selects New Orleans / Jefferson Protected Basin.
2.  App loads the 72h pre-flood snapshot.
3.  F18 checks readiness.
4.  F3 lists protected-basin ZCTAs (sorted by zcta_id; no default ranking).
5.  Operator selects a ZCTA.
6.  F4 shows certificate + pump/levee infrastructure evidence.
7.  F37 shows population / equity context.
8.  F40 shows five-component crisis vector.
9.  F41 optionally shows pump_failure_containment queue.
10. Dashboard note: this is policy-conditioned support, not a universal
    priority score.
```
