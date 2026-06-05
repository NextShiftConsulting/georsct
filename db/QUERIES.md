# Floodcaster DB — Queries and Schema Reference

> **ADR compliance**: All field names follow the ADR-034 hard cutover.
> No `kappa`, `kappa_gate`, bare `decision`, or bare `reason` anywhere in this layer.

---

## Storage architecture

| Store | What | Why |
|-------|------|-----|
| S3 parquet | Static features, OOF embeddings, NOAA/NFIP/SVI/ACS, ceilings | Large, immutable; DuckDB reads directly |
| DynamoDB | Certificate cache, ceiling jobs, audit log, ETag tracking | Fast key-value, TTL, write-once audit |
| DuckDB (embedded) | Session/portfolio query layer (F1–F7) | SQL over local tables + S3 parquet; no server |
| Redis | Hot F3 list views, live sensor values (5-min TTL) | Sub-10ms dashboard read path |

---

## Schema files (`db/schema/`)

Apply in numeric order at startup via `db/scripts/db_init.py`.

| File | Tables |
|------|--------|
| `001_scenarios.sql` | `scenarios`, `scenario_snapshots`, `scenario_zctas` + seed data |
| `002_certificate.sql` | `certificate`, `certificate_reason_code`, `gate_result`, `alternative_arm` |
| `003_zcta_observation.sql` | `zcta_observation`, `infrastructure_evidence` |
| `004_f40_and_policy_queue.sql` | `f40_vector`, `f40_field_status`, `f40_source_map`, `f40_audit`, `policy_view_queue`, `policy_view_queue_item` |
| `005_sessions.sql` | `sessions`, `session_decision`, `modal_source_etag` |

---

## Field name rules (ADR-034)

| Canonical name | Retired name(s) | Where |
|----------------|----------------|-------|
| `kappa_compat` | `kappa`, `kappa_gate` | `certificate`, `gate_result`, `alternative_arm` |
| `kappa_modal_min` | (new) | `certificate` |
| `public_decision` | `decision` | `certificate`, `session_decision` |
| `gate_reason` | `reason` | `certificate`, `gate_result` |
| `reason_codes` (table) | bare `reason` JSON | `certificate_reason_code` |

The CI grep check (`db_init.py::verify_adr034_compliance`) fails if `kappa_gate`, `kappa`, `decision`, or `reason` appear as column names in any table.

---

## Queries (`db/queries/`)

### F1 — select_event

**File**: `f01_select_event.sql`  
**Params**: `$scenario_id`  
**Returns**: scenario metadata + available snapshot hours  
**Errors**: 0 rows → `UNKNOWN_SCENARIO`; phase-2 build → `SCENARIO_NOT_DEPLOYED`

```sql
SELECT s.scenario_id, s.display_name, ...,
       array_agg(ss.snapshot_t ORDER BY ss.snapshot_t DESC) AS available_snapshots
FROM scenarios s
JOIN scenario_snapshots ss USING (scenario_id)
WHERE s.scenario_id = $scenario_id
GROUP BY ...;
```

---

### F2 — select_snapshot

**File**: `f02_select_snapshot.sql`  
**Params**: `$scenario_id`, `$snapshot_t`  
**Returns**: `precomputed_cert_count` (0 = fetch from DynamoDB)  
**Errors**: 0 rows → `SNAPSHOT_NOT_AVAILABLE`

---

### F3 — list_zctas

**File**: `f03_list_zctas.sql`  
**Params**: `$scenario_id`, `$snapshot_t`, `$embedding_arm`  
**Default arm**: `graphsage_v1`  
**Returns**: all in-scope ZCTAs with cert fields; cert-less ZCTAs get `unavailable_reason = 'NOT_COMPUTED'`

**Order**: `ORDER BY sz.zcta_id` — alphabetic, **not** by `kappa_compat`.

> **ADR-033 D1**: Ordering ZCTAs by `kappa_compat` constitutes an implicit scalar ranking.
> Operator-facing prioritisation lives only in `policy_view_queue`.

**Side-effect**: Python wrapper writes each certificated ZCTA to `session_decision` for F5/F7.

---

### F4 — select_zcta

**File**: `f04_select_zcta.sql`  
**Params**: `$scenario_id`, `$snapshot_t`, `$zcta_id`, `$embedding_arm`  
**Returns**: six sub-queries run sequentially:

| Block | Returns |
|-------|---------|
| 1 | Core cert fields (all ADR-034 canonical names) |
| 2 | `reason_codes` (normalised, ordered by `sort_order`) |
| 3 | `gate_result` rows (decision walkthrough; `gate_decision` is internal vocab) |
| 4 | `infrastructure_evidence` (scenario-discriminated union, §4 of five-scenario delta) |
| 5 | `alternative_arm` rows (for `decision_walkthrough.alternative_arms`) |
| 6 | `modal_source_etag` rows (audit block per ADR-030 D4) |

**Errors**: `ZCTA_NOT_IN_SCOPE` checked in Python wrapper before SQL.

---

### F5 — summarize_session

**File**: `f05_summarize_session.sql`  
**Params**: `$session_id`  
**Returns**: `total_zctas`, `zctas_targeted`, `zctas_that_flooded`, `zctas_that_didnt`, `flooded_zctas_missed`, `precision_at_execute`, `recall_at_execute`, `f1_at_execute`  
**Errors**: `NO_DECISIONS_YET` if no `session_decision` rows

---

### F6 — list_scenarios

**File**: `f06_list_scenarios.sql`  
**Params**: none  
**Returns**: all supported scenarios with metadata, snapshot hours, in-scope ZCTA count  
**Errors**: `PORTFOLIO_UNAVAILABLE` if fewer than 3 core scenarios present

Used to bootstrap the dashboard before F1; independent of session state.

---

### F7 — summarize_portfolio

**File**: `f07_summarize_portfolio.sql`  
**Params**: `$session_id`  
**Returns**: `scenarios_evaluated`, `scenarios_available`, `per_scenario[]`, `schema_fields_identical`, `scenarios_with_certs`  
**Errors**: `INSUFFICIENT_SCENARIOS` if fewer than 3 scenarios evaluated

> This is the benchmark claim: gate logic behaves consistently across five distinct flood archetypes. Three archetypes is the minimum to make the claim falsifiable.

---

### Ablation

**File**: `ablation.sql`  
**Params**: `$scenario_id`, `$snapshot_t`, `$zcta_id` (+ `$perturbation_arm` for stress test)  
**Two blocks**:
1. All arms for one ZCTA with `kappa_compat_delta` vs default arm and `would_change_decision`
2. Portfolio-level instability rate for a perturbation arm

Implements the per-scenario stress tests from five-scenario delta §3.3:

| Scenario | Stress test |
|----------|-------------|
| S1 | `spatial_lag_v1` vs `graphsage_v1` (drainage adjacency) |
| S2 | Pump-failure perturbation arm |
| S3 | Canyon routing perturbation arm |
| S4 | `geo_v1` track-shift arm |
| S5 | Sewer-shed perturbation arm |

---

### Policy-view queue

**File**: `policy_view_queue.sql`  
**Three blocks**:
1. Build a new queue from `f40_vector` with a declared `ordering_rule`
2. Retrieve an existing queue with its items
3. Decision counts per queue item (F5-style summary per queue)

> Ordering lives **only** in this table. Certificates and F40 vectors must not be sorted by any scalar proxy to produce an operator queue.

---

## Python wrappers (`db/scripts/`)

| Script | Wraps | Key behaviour |
|--------|-------|---------------|
| `db_init.py` | Schema setup | Applies schema files in order; runs ADR-034 compliance grep |
| `session.py` | F1, F2 | Raises typed errors (`UnknownScenario`, `SnapshotNotAvailable`, etc.) |
| `f03_list_zctas.py` | F3 | Writes `session_decision` rows; looks up `y_true` from parquet |
| `f04_select_zcta.py` | F4 | Runs 6 SQL blocks; assembles `ZCTADetail` dataclass |
| `f05_f07_portfolio.py` | F5, F7 | Reads `session_decision`; enforces preconditions |
| `f06_list_scenarios.py` | F6 | Enforces `PORTFOLIO_UNAVAILABLE` guard |

---

## ADR-033: what F40 must NOT do

F40 (`f40_vector`) is composition-only. These columns must never appear:

```
resource_priority_score   ← banned
priority_rank             ← banned
risk_score                ← banned
weighted_priority         ← banned
overall_priority          ← banned
```

Ordering belongs in `policy_view_queue.ordering_rule`, declared with a named
`policy_view` and `source = 'F40_VECTOR'`.

---

## Invariants enforced by schema constraints

| Invariant | Enforcement |
|-----------|-------------|
| I4: every cert has `task_residual_floor` or `task_residual_floor_unavailable_reason` | `CHECK` constraint in `002_certificate.sql` |
| I7: schema invariance across scenarios | `verify_adr034_compliance()` in `db_init.py` |
| I8: gate ordering invariance | Gate names are fixed strings; no per-scenario gate columns |
| I9: decision determinism | Cert keyed by `(scenario_id, snapshot_t, zcta_id, embedding_arm, target)`; `UNIQUE` constraint |

---

## Error taxonomy

| Code | Raised by | Condition |
|------|-----------|-----------|
| `UNKNOWN_SCENARIO` | F1, F6 | `scenario_id` not in `scenarios` table |
| `SCENARIO_NOT_DEPLOYED` | F1 | Phase-2 scenario, phase-1-only deployment |
| `SNAPSHOT_NOT_AVAILABLE` | F2 | `snapshot_t` not in `scenario_snapshots` |
| `ZCTA_NOT_IN_SCOPE` | F4 | `zcta_id` not in `scenario_zctas` |
| `NO_DECISIONS_YET` | F5 | No `session_decision` rows for session |
| `INSUFFICIENT_SCENARIOS` | F7 | Fewer than 3 distinct scenarios in session |
| `PORTFOLIO_UNAVAILABLE` | F6 | Fewer than 3 core scenarios in `scenarios` table |
