# Floodcaster MVP1 — Functional Contract

> **Scope.** What the dashboard *does*, expressed as functions with
> inputs, outputs, preconditions, postconditions, and error modes. No
> framework. No UI. No deployment. A correct implementation of this
> contract behaves identically regardless of what renders it.

---

## 0. The session model

A user session has three pieces of state:

```
session.event         : one of {harvey_2017, florence_2018, imelda_2019}
session.snapshot_t    : hours-before-landfall ∈ {72, 48, 24, 0}
session.selected_zcta : ZCTA id, or null
```

Default on session start: `event = harvey_2017`, `snapshot_t = 72`,
`selected_zcta = null`.

All five functions below read this state. Three of them mutate it.
No other state exists in MVP1.

---

## 1. The five functions

### F1. `select_event(event)`

**Purpose.** Switch the active historical event.

**Input.**
- `event` ∈ `{harvey_2017, florence_2018, imelda_2019}`

**Preconditions.**
- Event must be in the supported set.

**Postconditions.**
- `session.event = event`
- `session.snapshot_t` reset to 72
- `session.selected_zcta = null`
- All cached certificate results from prior event are invalidated

**Outputs.**
- New event metadata: name, landfall timestamp, affected region bbox,
  list of available snapshot times, ground-truth outcome summary
  (total NFIP claims filed, peak flood extent, fatality count)

**Errors.**
- `UNKNOWN_EVENT` if `event` not in supported set

---

### F2. `select_snapshot(hours_before)`

**Purpose.** Pick how far before landfall to evaluate the certificate.

**Input.**
- `hours_before` ∈ `{72, 48, 24, 0}`

**Preconditions.**
- Snapshot must be available for `session.event` (all four are, for v1).

**Postconditions.**
- `session.snapshot_t = hours_before`

**Outputs.**
- Snapshot metadata: as-of timestamp, which data sources were frozen
  at this snapshot, advisory bulletin issued by NHC at that hour

**Errors.**
- `UNAVAILABLE_SNAPSHOT` if the requested hour has no data for the event

---

### F3. `list_zctas()`

**Purpose.** Return the ZCTAs covered by the active event with their
certificates evaluated at the active snapshot.

**Input.** None (reads `session.event`, `session.snapshot_t`).

**Preconditions.** None.

**Postconditions.** None (read-only).

**Outputs.** A list of records, one per in-scope ZCTA:

```
{
  zcta_id              : str        e.g. "77002"
  state_fips           : str
  county_fips          : str
  certificate : {
    R                  : float
    S_sup              : float
    N                  : float
    alpha              : float
    kappa              : float
    sigma              : float
    task_residual_floor          : float | null
    decision           : "EXECUTE" | "CAUTION" | "REFUSE"
    gate_reached       : str       (ADR-016 identifier or "NONE")
    reason             : str       (human-readable)
  }
  outcome : {
    actually_flooded   : bool
    nfip_claims_filed  : int
    total_loss_usd     : float
  } | null
}
```

`outcome` is null for ZCTAs without ground-truth claim data.

**Errors.**
- `CERTIFICATE_UNAVAILABLE` for individual ZCTAs is **not** an error —
  the record is returned with `certificate = null` and a per-record
  `unavailable_reason` field.
- `EVENT_DATA_INCOMPLETE` if the event/snapshot pair has fewer than
  100 covered ZCTAs (indicates a data pipeline problem).

**Performance contract.**
- Must return within 2 seconds for any (event, snapshot) pair.
- Implementations call swarm-it-api `/certify/geo` per ZCTA with the
  appropriate snapshot_year, ZCTA, target = `nfip_total_loss`.
  Per-call cache hit rate is >95% for the supported events (all are
  pre-warmed at deploy time).

---

### F4. `select_zcta(zcta_id)`

**Purpose.** Drill down to one ZCTA for detailed certificate inspection.

**Input.**
- `zcta_id` (string)

**Preconditions.**
- `zcta_id` must appear in the result of `list_zctas()` for the
  current (event, snapshot).

**Postconditions.**
- `session.selected_zcta = zcta_id`

**Outputs.** Full per-ZCTA detail:

```
{
  zcta_id                  : str
  certificate              : <as in F3>
  outcome                  : <as in F3>
  decision_walkthrough : {
    gate_1_integrity       : { passed: bool, value: float, threshold: float, why: str }
    gate_2_consensus       : { passed: bool, value: float, threshold: float, why: str }
    gate_3_admissibility   : { passed: bool, value: float, threshold: float, why: str }
    gate_4_grounding       : { passed: bool, value: float, threshold: float, why: str }
  }
  modal_evidence : {
    flood_modal_features   : { fema_zone, twi_value, nfip_density, ... }
    socioeconomic_features : { svi_quartile, median_income, ... }
    static_etag            : str
  }
  alternative_arms : [
    { embedding_arm: str, certificate: <as in F3>, would_change_decision: bool }
  ]
  ground_truth_comparison : {
    decision               : "EXECUTE" | "CAUTION" | "REFUSE"
    actually_flooded       : bool
    correct                : bool      (decision matches what was right)
  } | null
}
```

**Errors.**
- `ZCTA_NOT_IN_SCOPE` if `zcta_id` is not in the active event's coverage
- `ZCTA_CERTIFICATE_UNAVAILABLE` if the per-ZCTA certify call failed

---

### F5. `summarize_session()`

**Purpose.** Compute "what would have happened if you acted on this."

**Input.** None (reads full session state).

**Preconditions.**
- `list_zctas()` must have been called at least once for the current
  (event, snapshot) pair.

**Postconditions.** None (read-only).

**Outputs.**

```
{
  event                    : str
  snapshot_t               : int

  decision_counts : {
    EXECUTE                : int
    CAUTION                : int
    REFUSE                 : int
  }

  if_acted_on_execute : {
    zctas_targeted         : int
    zctas_that_flooded     : int     (true positives)
    zctas_that_didnt       : int     (false positives)
    flooded_zctas_missed   : int     (false negatives — flooded but not in EXECUTE)
    precision              : float
    recall                 : float
    estimated_loss_reduced : float   (sum of NFIP losses in true positives)
  }

  if_acted_on_baseline_rank_by_kappa_only : { <same shape> }
  if_acted_on_baseline_rank_by_R2_only    : { <same shape> }

  comparison_narrative     : str   (auto-generated, 2-3 sentences)
}
```

The three "if acted on" blocks let a user see what the certificate
adds over scalar-only ranking. This is the operator-facing form of
the manuscript's headline experimental finding.

**Errors.**
- `NO_DECISIONS_YET` if `list_zctas()` has not been called

---

## 2. Read-only auxiliary functions

These do not mutate session state. They exist because the dashboard
will need them; whether they are exposed as user-callable verbs or
hidden implementation details is a downstream call.

### A1. `available_events()`
Returns the list of supported events with metadata.

### A2. `available_snapshots(event)`
Returns the list of available `hours_before` values for an event.

### A3. `event_outcome_truth(event)`
Returns the final ground-truth flood map and claims summary for an
event. Independent of any snapshot — this is the answer key.

### A4. `gate_reference()`
Returns the canonical list of gate identifiers (per ADR-016) with
plain-language descriptions. Used by the decision walkthrough.

### A5. `disclosure()`
Returns the patent-pending notice, dataset version, API version,
and ADK version strings. Must appear in every response payload as
an `audit` block.

---

## 3. State transitions

The five mutating functions induce a small state machine:

```
                  ┌─────────────────┐
                  │   start session │
                  └────────┬────────┘
                           │ default: harvey_2017, t=72, zcta=null
                           ▼
                  ┌─────────────────┐
       F1 ───────►│  event chosen   │◄────────── F1
                  └────────┬────────┘
                           │
                           │ list_zctas() may be called any time;
                           │ does not change state
                           │
                           ▼
                  ┌─────────────────┐
       F2 ───────►│ snapshot chosen │◄────────── F2
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  ZCTA selected  │◄────────── F4
                  └────────┬────────┘
                           │
                           │ summarize_session() may be called any time
                           │ after list_zctas() at least once
                           ▼
                  ┌─────────────────┐
                  │  session ready  │
                  └─────────────────┘
```

The state is small enough to fit in a URL query string. This is
deliberate — a session can be linked, bookmarked, and reproduced
without server-side state.

---

## 4. Invariants

Properties that must hold at all times.

**I1. Read-only against ground truth.** No function mutates the
event outcome data. `event_outcome_truth(event)` is the answer key
and is immutable across the session.

**I2. Snapshot causality.** No function ever returns an outcome
field that "leaks" from a snapshot later than `session.snapshot_t`.
A certificate computed at t=72 must not be informed by NFIP claims
filed during the actual event. Pre-event modal sources only.

**I3. Decision determinism.** Calling `list_zctas()` twice in a row
with the same (event, snapshot) pair must return certificates whose
`decision` fields are bitwise identical. The compute path is
deterministic; any seed-dependent step (bootstrap, perturbation
ensembles) uses fixed seeds keyed to (event, snapshot, zcta_id).

**I4. Certificate completeness.** Every returned certificate carries
all six core fields (R, S_sup, N, alpha, kappa, sigma) and either a
non-null task_residual_floor or an explicit `task_residual_floor_unavailable_reason`. No
silent omissions.

**I5. Disclosure propagation.** Every response from every function
carries the `disclosure()` audit block. No exceptions.

**I6. No mutation of upstream state.** The dashboard never writes
to swarm-it-api, never modifies the v24.002 dataset, never creates
or updates ADRs. It is strictly a read-side consumer.

---

## 5. Error taxonomy

```
UNKNOWN_EVENT                  — F1 input out of supported set
UNAVAILABLE_SNAPSHOT           — F2 input has no data for current event
EVENT_DATA_INCOMPLETE          — F3 finds < 100 ZCTAs covered
ZCTA_NOT_IN_SCOPE              — F4 input not in event coverage
ZCTA_CERTIFICATE_UNAVAILABLE   — F4 underlying API call failed
NO_DECISIONS_YET               — F5 called before F3
API_UNAVAILABLE                — any function when swarm-it-api times out
CACHE_STALE                    — ETag changed mid-session (informational)
```

`CACHE_STALE` is not a hard failure — the dashboard surfaces a banner
that the underlying data version has changed and offers to reload.
This is the operator-facing form of the architecture's high-σ modal
incoherence detection from a different layer.

---

## 6. Function-call dependency graph

```
F1 (select_event)        ──┐
                           ├──► F3 (list_zctas) ──► F5 (summarize_session)
F2 (select_snapshot)     ──┤            │
                           │            ▼
                           │      F4 (select_zcta)
                           │
                           └─ A1, A2, A3 are independent reads
```

F4 depends on F3 having been called (because F4 needs the list of
in-scope ZCTAs to validate its input). F5 depends on F3 (no decisions
to summarize otherwise). F1 and F2 are independent triggers that
both invalidate F3/F4/F5 results.

---

## 7. What this contract does not specify

- How any function renders (text? table? map? list?)
- Whether functions are synchronous or async at the implementation layer
- What language/framework wraps them
- Caching strategy beyond what's required for the 2-second F3 contract
- Authentication (assumed: same public read-only API key as the
  tutorial notebooks)
- Layout, color, typography, interaction patterns

All seven are deliberate omissions. They are implementation choices
downstream of this contract.

---

## 8. What this contract assumes from swarm-it-api

- `/certify/geo` accepts snapshot_year and returns the full
  certificate including R, S_sup, N, alpha, kappa, sigma, decision,
  gate_reached, reason.
- `/certify/geo/ablation` accepts an embedding_arm parameter for
  the F4 alternative_arms block.
- `/ceiling` results for the supported event targets are pre-warmed
  and return from cache; no async polling needed at runtime.
- The audit block on every response includes static ETag, API
  version, ADK version, and `ip_status: patent_pending_19/575,615`.

If any of these change, the contract changes. Worth a one-line check
against the OpenAPI spec when it ships.
