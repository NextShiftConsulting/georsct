# 03 — Riverside / Coachella Desert Flash Flood

## Scenario definition

```text
scenario_id:      riverside_coachella_desert_flash
display_name:     Riverside / Coachella Valley Desert Flash Flood
flood_archetype:  desert_wash_flash_flood
forecast_horizon: 24h or 72h
primary_decision: where to warn, close roads, pre-position swift-water rescue,
                  and protect isolated communities
build_phase:      core
```

This is a strong third scenario because it is very different from Houston and
New Orleans. It is not bayou drainage or protected-basin pump/levee risk. It is
**fast-onset desert flash flooding** driven by canyon/wash flow, burn-scar
overlap, upstream catchment, and road-access fragility.

**F40 is a non-scalar crisis vector.** ADR-033 is explicit. F40 emits five
independent components and does not rank ZCTAs. Any ranking belongs downstream
in a named policy-view queue.

---

## Function inventory

### F1 — Select scenario

```python
get_scenario_detail(
    scenario_id="riverside_coachella_desert_flash"
)
```

```sql
SELECT *
FROM scenarios
WHERE scenario_id = 'riverside_coachella_desert_flash';
```

Expected fields: `anchor_storm`, `region`, `flood_archetype`,
`primary_decision`, `build_phase`, `affected_zctas`, `peak_extent_km2`,
`fatality_count`.

---

### F2 — Select forecast snapshot

```python
get_scenario_snapshots(
    scenario_id="riverside_coachella_desert_flash"
)
```

```sql
SELECT scenario_id, snapshot_t
FROM scenario_snapshots
WHERE scenario_id = 'riverside_coachella_desert_flash'
ORDER BY snapshot_t DESC;
```

Recommended demo default: `snapshot_t = -24` (`forecast_horizon =
"24h_pre_flood"`). For flash flooding, 24h is more convincing than 72h —
flash-flood response is time-compressed. 72h is also supported.

---

### F18 — Scenario readiness

```python
check_scenario_readiness(
    scenario_id="riverside_coachella_desert_flash",
    snapshot_t=-24
)
```

| Status | Behavior |
|--------|----------|
| `ready` | F40 executes |
| `limited` | F40 executes with `field_status` warnings |
| `unreliable` | F40 returns `409 SCENARIO_UNREADY` |

ADR-033 requires scenario-level unreliability to fail closed with
`409 SCENARIO_UNREADY`, not a degraded vector (not a generic 400).

---

### F3 — List ZCTAs

```python
list_zctas(
    scenario_id="riverside_coachella_desert_flash",
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
    c.task_residual_floor,
    c.public_decision,
    c.gate_reached,
    c.gate_reason
FROM certificates c
WHERE c.scenario_id = 'riverside_coachella_desert_flash'
  AND c.snapshot_t = $1
  AND c.embedding_arm = 'graphsage_v1'
ORDER BY c.zcta_id;
```

**Do not default-sort by `kappa_compat`** — that quietly creates a scalar
ranking. Use policy-view queues for any ordering.

---

### F4 — Select ZCTA / certificate walkthrough

```python
select_zcta(
    scenario_id="riverside_coachella_desert_flash",
    snapshot_t=-24,
    zcta_id="92262"
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
WHERE c.scenario_id = 'riverside_coachella_desert_flash'
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
    scenario_id="riverside_coachella_desert_flash",
    zcta_id="92262"
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
WHERE zcta_id = '92262';
```

Vehicle access and older-population indicators matter here — flash floods can
isolate roads quickly and strand residents without transportation.

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
    scenario_id="riverside_coachella_desert_flash",
    snapshot_t=-24
)
```

```json
{
  "zcta_id": "92262",
  "reliability_action": "escalate",
  "reason_codes": ["WARN_INFORMATIONAL"]
}
```

Possible values: `trust` | `review` | `escalate` | `suppress`.

For S3, `escalate` should appear when evidence suggests rapid-onset risk but
confidence is borderline — especially near washes, burn scars, or degraded
road access.

---

### F40 — Build resource priority vector

F40 returns **five independent components** — not a scalar score, not a rank.

```python
build_resource_priority_vector(
    scenario_id="riverside_coachella_desert_flash",
    snapshot_t=-24,
    zcta_id="92262"
)
```

```json
{
  "scenario_id": "riverside_coachella_desert_flash",
  "snapshot_t": -24,
  "zcta_id": "92262",
  "vector": {
    "risk_level": "high",
    "reliability_action": "escalate",
    "population_context": "medium",
    "access_fragility": "high",
    "equity_context": "q3"
  }
}
```

F40 must **not** return:

```json
{
  "resource_priority_score": 0.88,
  "priority_rank": 2
}
```

ADR-033 forbids this.

#### S3 F40 source map

| Component | Source function | Source field | Transform |
|-----------|----------------|--------------|-----------|
| `risk_level` | F4 | `certificate.public_decision` + `gate_result.active_metric` + `gate_result.active_value` | enum_remap |
| `reliability_action` | F12 | `action_queue[location_id].reliability_action` | passthrough |
| `population_context` | F37 | `vulnerability_context.population_density` | enum_remap |
| `access_fragility` | F4 | `infrastructure_evidence.road_access_status` + `burn_scar_overlap` + `upstream_catchment_km2` | scenario_discriminated_enum_remap |
| `equity_context` | F37 | `vulnerability_context.svi_quartile` | passthrough |

`access_fragility` is where S3-specific logic lives. Thresholds are
scenario-configured, not hardcoded in F40.

| `road_access_status` | `burn_scar_overlap` | `access_fragility` |
|----------------------|---------------------|--------------------|
| `closed` | any | `high` |
| `at_risk` | any | `medium` |
| `open` | `true` | `medium` |
| `open` | `false` | `low` |
| `unknown` | any | `null` + `field_status` warning |

---

### F41 — Policy-view queue (optional)

Natural policy views for S3:

```text
life_safety_first
road_closure_first
swift_water_rescue_staging
burn_scar_flash_response
isolated_community_support
```

```python
build_policy_view_queue(
    scenario_id="riverside_coachella_desert_flash",
    snapshot_t=-24,
    policy_view="road_closure_first"
)
```

Required persisted queue metadata:

```json
{
  "policy_view": "road_closure_first",
  "policy_view_version": "v1",
  "ordering_rule": "access_fragility DESC, risk_level DESC, reliability_action DESC",
  "source": "F40_VECTOR",
  "note": "Policy-conditioned projection. Not a universal priority score."
}
```

Example queue item:

```json
{
  "zcta_id": "92262",
  "ordinal_position": 1,
  "recommended_action": "verify road closure and pre-stage swift-water rescue",
  "action_rationale": [
    "access_fragility=high",
    "risk_level=high",
    "reliability_action=escalate",
    "burn_scar_overlap=true"
  ]
}
```

**F40 does not rank. F41 / Crisis Orchestrator may queue under a declared
policy view.**

---

## Infrastructure evidence

The key S3 infrastructure fields:

```sql
SELECT
    canyon_id,
    wash_segment_id,
    burn_scar_overlap,
    upstream_catchment_km2,
    road_access_status
FROM infrastructure_evidence
WHERE scenario_id = 'riverside_coachella_desert_flash'
  AND certificate_id = $1;
```

| Field | Interpretation |
|-------|----------------|
| `canyon_id` | Flash-flood channel / terrain funnel |
| `wash_segment_id` | Desert wash exposure |
| `burn_scar_overlap` | Post-fire runoff amplification |
| `upstream_catchment_km2` | Upstream contributing area |
| `road_access_status` | `open` \| `at_risk` \| `closed` \| `unknown` |

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
      "missing_reason": "Road access status unavailable for this desert flash-flood snapshot.",
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
vehicle_access
age_65_plus

canyon_id
wash_segment_id
burn_scar_overlap
upstream_catchment_km2
road_access_status

nfip historical loss / event features
noaa storm-event history
```

---

## Demo flow

```text
1.  Operator selects Riverside / Coachella Desert Flash Flood.
2.  App loads the 24h pre-flood snapshot.
3.  F18 checks readiness.
4.  F3 lists ZCTAs (sorted by zcta_id; no default ranking).
5.  Operator selects a ZCTA near a wash or canyon.
6.  F4 shows certificate + road/wash/burn-scar evidence.
7.  F37 shows population / equity context (vehicle access, age 65+).
8.  F40 shows five-component crisis vector.
9.  F41 optionally shows road_closure_first or swift_water_rescue_staging queue.
10. Dashboard note: this is policy-conditioned decision support, not a
    universal priority score.
```
