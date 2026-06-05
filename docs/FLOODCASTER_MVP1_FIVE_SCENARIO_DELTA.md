# Floodcaster MVP1 — Five-Scenario Functional Requirements

> **Scope.** Functional delta against `FLOODCASTER_MVP1_FUNCTIONAL_CONTRACT.md`
> to support five 72-hour pre-flood crisis scenarios spanning five distinct
> flood archetypes. No UI, no framework, no deployment. The contract from
> the prior document holds; this document specifies what changes and what
> is added.

---

## 0. The five scenarios

| ID | Scenario | Anchor event | Crisis geometry | Build phase |
|----|----------|--------------|-----------------|-------------|
| S1 | `harris_houston_urban` | Harvey 2017 | urban drainage, bayou overflow, drainage-channel stress | Core (phase 1) |
| S2 | `orleans_jefferson_protected_basin` | Ida 2021 (Gulf landfall, not the Northeast remnants) | levee/pump/canal-dependent protected basin | Core (phase 1) |
| S3 | `riverside_coachella_desert_flash` | Hilary 2023 | mountain-canyon → desert-wash flash flooding | Core (phase 1) |
| S4 | `lee_charlotte_surge_evacuation` | Ian 2022 | storm surge, evacuation timing, coastal catastrophic loss | Extension (phase 2) |
| S5 | `nyc_nj_urban_cloudburst` | Ida 2021 Northeast remnants | dense urban cloudburst, basement/subway/sewer overload | Extension (phase 2) |

Each scenario produces the same four-decision output: `EXECUTE`,
`CAUTION`, `REFUSE`, plus the new `ESCALATE` value introduced in §3.
The crisis geometry differs; the certificate schema does not.

The portfolio is the contribution. A single scenario is a product
demo; five scenarios with one schema across five archetypes is a
benchmark.

---

## 1. Changes to the existing contract

### 1.1 `session.event` enum expands

```
session.event ∈ {
  harris_houston_urban,
  orleans_jefferson_protected_basin,
  riverside_coachella_desert_flash,
  lee_charlotte_surge_evacuation,
  nyc_nj_urban_cloudburst,
}
```

Default on session start moves from `harvey_2017` to
`harris_houston_urban` — the same event, renamed to the
scenario-identifier convention. Storm-name strings are surfaced in
metadata, not used as enum values, because two scenarios share a
storm (S2 and S5 both anchor on Ida).

### 1.2 `session.snapshot_t` per-scenario availability

The `{72, 48, 24, 0}` default holds for S1, S2, S4, S5. S3 (Hilary)
adds two extra snapshots because the storm's track and intensity
shifted late in the warning window: `{96, 72, 48, 36, 24, 12, 0}`. The
F2 contract is unchanged — `available_snapshots(event)` returns the
per-scenario list, and the dashboard picks from it.

### 1.3 `decision` enum expands from three values to four

```
decision ∈ {EXECUTE, CAUTION, REFUSE, ESCALATE}
```

`ESCALATE` is the new operator-facing terminal verdict, distinct from
`REFUSE` because it carries an action: route the case to human review
with priority. Mapping to the internal ADR-016 gate vocabulary:

| Internal (ADR-016) | Public API (ADR-029) | Floodcaster operator verdict |
|--------------------|----------------------|------------------------------|
| EXECUTE            | EXECUTE              | TRUST (proceed)              |
| REJECT (Gate 1)    | REFUSE               | SUPPRESS (do not act, do not escalate — task too noisy) |
| BLOCK (Gate 2)     | REFUSE               | ESCALATE (no consensus across arms — needs human)       |
| RE_ENCODE (Gate 3) | CAUTION              | REVIEW (try different evidence)                         |
| REPAIR (Gate 4)    | CAUTION              | ESCALATE (Landauer breach — needs human)                |

The mapping splits REFUSE into two operator-facing terminal verdicts
(SUPPRESS and ESCALATE) and CAUTION into two non-terminal verdicts
(REVIEW and ESCALATE). Both ESCALATE entries are the same final verb;
they differ in `gate_reached` only.

**`ESCALATE` is a Floodcaster application-layer verdict, not an API
field.** `swarm-it-api` never returns `ESCALATE`. Floodcaster computes
it from `(certificate.decision, certificate.gate_reached)` client-side.
`GeoCertificate.decision` remains `EXECUTE | CAUTION | REFUSE` per
ADR-029 D1. The `ESCALATE` value lives in F3/F4 output only.

This is necessary because the original three-value collapse hides the
operator's most important question: *who do I call?* SUPPRESS means
"nobody, this is statistical noise"; ESCALATE means "human reviewer,
now." Conflating them in a UI is exactly the failure mode prior
operator-voice work has stumbled on.

### 1.4 New required field on F4 output: `infrastructure_evidence`

The detail view returned by `select_zcta` (F4) now includes a per-
scenario `infrastructure_evidence` block that captures the
crisis-geometry-specific signals. The block exists for every scenario;
its *fields* are scenario-specific (§4 below).

---

## 2. The portfolio-level functions

Two new functions complete the five-scenario contract.

### F6. `list_scenarios()`

**Purpose.** Return the supported scenarios with metadata.

**Input.** None.

**Postconditions.** None (read-only).

**Outputs.**

```
[
  {
    scenario_id        : str
    display_name       : str   e.g. "Houston 72-Hour Urban Flood Forecast"
    anchor_storm       : str   e.g. "Harvey 2017"
    region             : str   e.g. "Harris County, TX"
    flood_archetype    : str   e.g. "urban drainage / bayou overflow"
    primary_decision   : str   e.g. "Which high-risk neighborhoods are ready for action?"
    in_scope_zcta_count: int
    snapshot_hours     : list[int]
    build_phase        : "core" | "extension"
    ground_truth_summary : {
      total_nfip_claims   : int
      total_loss_usd      : float
      affected_zctas      : int
      peak_extent_km2     : float | null
      fatality_count      : int | null
    }
  },
  ...
]
```

**Errors.**
- `PORTFOLIO_UNAVAILABLE` if fewer than three scenarios are deployed
  (indicates configuration error; MVP1 requires phase 1 minimum).

### F7. `summarize_portfolio()`

**Purpose.** Compute cross-scenario reliability statistics. This is
the "benchmark, not demo" function.

**Input.** None (reads completed scenario sessions from server-side
cache).

**Preconditions.**
- At least one snapshot of each phase-1 scenario must have been
  evaluated in this session.

**Outputs.**

```
{
  scenarios_evaluated     : int
  scenarios_available     : int

  per_scenario : [
    {
      scenario_id              : str
      decision_distribution    : { EXECUTE: int, CAUTION: int, REFUSE: int, ESCALATE: int }
      precision_at_execute     : float
      recall_at_execute        : float
      escalation_rate          : float
      ground_truth_alignment   : float   (fraction of decisions that matched outcome)
      dominant_failure_gate    : str | null  (most frequent gate_reached among non-EXECUTE)
    }
  ]

  cross_scenario : {
    archetype_invariance : {
      decision_consistency_across_archetypes : float
      same_gate_fires_same_way               : bool
      narrative                              : str
    }
    schema_invariance : {
      certificate_fields_identical_across_scenarios : bool
      gate_ordering_identical                       : bool
    }
    headline_finding : str   (auto-generated, 2-3 sentences)
  }
}
```

The `archetype_invariance` block is what turns the five-scenario set
from a product portfolio into a benchmark result. If the gate logic
behaves consistently across five distinct flood geometries — urban
drainage, protected basin, desert flash, surge, urban cloudburst —
that is itself the finding. If it doesn't, that's *also* a finding,
and a more informative one.

**Errors.**
- `INSUFFICIENT_SCENARIOS` if fewer than three scenarios have been
  evaluated in the current session.

---

## 3. Per-scenario evidence requirements

Each scenario must produce three things beyond the base certificate.

### 3.1 Scenario-specific modal sources

Beyond the static sources already in v24.002 (FEMA, NFIP, TWI, SVI,
ACS), each scenario requires modal sources that match its crisis
geometry. Sources marked `[PENDING]` need v24.003 or sidecar
inclusion before phase-2 ships.

| Scenario | Required static modal sources | Required live modal sources |
|----------|-------------------------------|----------------------------|
| S1 | Harris County drainage network, bayou stage history | USGS Buffalo Bayou + Brays Bayou + Greens Bayou gauges, NOAA NWM HUC8 12040104 |
| S2 | Levee segments + pump station network + subsidence rate [PENDING] | USGS Mississippi River + Lake Pontchartrain gauges, Sewerage & Water Board pump status [PENDING] |
| S3 | Mountain-canyon drainage paths, dry-wash centerlines, burn-scar polygons [PENDING] | USGS Whitewater River + canyon-mouth gauges, NWS flash-flood watches |
| S4 | Storm-surge inundation modeling (SLOSH) zones, evacuation route graph [PENDING] | NHC surge forecasts, NOAA tide gauges, evacuation order status [PENDING] |
| S5 | NYC/NJ sewer-shed boundaries, basement-apartment density layer [PENDING], subway entrance elevations [PENDING] | USGS Hudson + Passaic + Raritan gauges, NYC DEP green-infrastructure status [PENDING] |

The `[PENDING]` sources are not ship-blockers for phase 1 (only S1–S3
ship in phase 1, and S1/S3 have no pending sources). They are
ship-blockers for the specific phase-2 scenarios that need them.

### 3.2 Scenario-specific gate evidence

The decision walkthrough in F4 must surface the gate evidence that
matters most for the scenario's archetype. The four base gates run
identically across scenarios (per ADR-016 fixed ordering — that's the
invariance claim). What changes is the *narrative* the dashboard
surfaces at each gate.

| Gate | S1 narrative | S2 narrative | S3 narrative | S4 narrative | S5 narrative |
|------|--------------|--------------|--------------|--------------|--------------|
| Gate 1 (N) | "Is the bayou-stage signal too noisy for prediction?" | "Is the surge × subsidence × levee interaction too noisy?" | "Is canyon runoff too sparsely sensed for prediction?" | "Is the surge tail too extreme for the model class?" | "Is the cloudburst intensity outside the calibration range?" |
| Gate 2 (consensus) | "Do drainage-network and flood-zone evidence agree?" | "Do levee status and surge forecast agree on protected-basin risk?" | "Do upstream rain and downstream wash gauges agree?" | "Do SLOSH and rainfall-runoff models agree?" | "Do sewer-shed model and rainfall radar agree?" |
| Gate 3 (admissibility) | "Is the certificate stable under bayou-stage perturbation?" | "Is the certificate stable under pump-failure simulation?" | "Is the certificate stable under canyon-routing uncertainty?" | "Is the certificate stable under track-shift perturbation?" | "Is the certificate stable under cell-scale rainfall variance?" |
| Gate 4 (grounding) | "Is the prediction grounded in modal flood evidence, not SVI alone?" | "Is the prediction grounded in infrastructure data, not population alone?" | "Is the prediction grounded in terrain, not historical-claim density?" | "Is the prediction grounded in surge physics, not surrounding-storm history?" | "Is the prediction grounded in sewer/cloudburst evidence, not just rainfall?" |

The underlying compute is identical. Only the natural-language
rendering of `reason` and `gate_reached` differs per scenario. This is
a localization layer over a single compute path — not five
implementations.

### 3.3 Scenario-specific stress tests

Each scenario must include at least one stress test specific to its
crisis geometry, surfaced as part of F4's `alternative_arms` block:

| Scenario | Stress test |
|----------|-------------|
| S1 | Re-rank with drainage-network adjacency vs queen contiguity |
| S2 | Simulate one pump-station failure; certificate must not collapse silently |
| S3 | Perturb canyon routing by ±2 routing-graph edges; ranking stability |
| S4 | Shift forecast track by 15 nm; coastal ZCTA recall must not collapse |
| S5 | Perturb rainfall cell assignment by one block-group; basement-flood ZCTAs must remain stable |

Each stress test is implementable as a `/certify/geo/ablation` call
with a scenario-specific perturbation parameter. The contract
requirement is the existence of the test and a documented pass/fail
threshold; the specific perturbation magnitudes are tuning parameters
not locked here.

---

## 4. `infrastructure_evidence` block — per-scenario shape

The new F4 field is a union type, scenario-discriminated.
`infrastructure_evidence` is a **Floodcaster application-layer field**
assembled from API response metadata and scenario-specific data
sources. It does not appear in `GeoCertifyResponse` (ADR-030).

```
infrastructure_evidence: union {
  S1_urban_drainage {
    bayou_segment_id          : str | null
    drainage_district_id      : str
    impervious_surface_pct    : float
    drainage_capacity_status  : "nominal" | "stressed" | "exceeded" | "unknown"
  },
  S2_protected_basin {
    levee_segment_id          : str | null
    pump_station_ids          : list[str]
    pump_station_status       : list["operational" | "degraded" | "failed" | "unknown"]
    subsidence_rate_mm_yr     : float
    canal_proximity_m         : float
  },
  S3_desert_flash {
    canyon_id                 : str | null
    wash_segment_id           : str | null
    burn_scar_overlap         : bool
    upstream_catchment_km2    : float
    road_access_status        : "open" | "at_risk" | "closed" | "unknown"
  },
  S4_surge_evacuation {
    slosh_category            : int   (1-5)
    elevation_m_msl           : float
    evacuation_route_status   : "open" | "congested" | "closed" | "unknown"
    coastal_distance_m        : float
  },
  S5_urban_cloudburst {
    sewer_shed_id             : str
    basement_apartment_count  : int | null
    subway_station_ids        : list[str]
    impervious_surface_pct    : float
  }
}
```

A scenario-discriminating tag lets a single F4 implementation return
the right shape per scenario. Consumers branch on the tag.

---

## 5. Invariants — additions to the prior contract

The six invariants from `FLOODCASTER_MVP1_FUNCTIONAL_CONTRACT.md` (I1–I6)
all hold. Three new invariants are added.

**I7. Schema invariance across archetypes.** Every scenario must
return the same certificate field set. R, S_sup, N, alpha, kappa,
sigma, task_residual_floor, decision, gate_reached, reason are present and
non-null for every supported scenario. A scenario-specific
`infrastructure_evidence` block is present per §4; its discriminator
tag is mandatory.

**I8. Gate ordering invariance.** The four ADR-016 gates fire in the
same fixed order in every scenario. A scenario cannot reorder gates,
skip gates, or add scenario-local gates. (Per ADR-004 single gate
authority. The scenarios differ in *narrative*, not in *enforcement*.)

**I9. Cross-scenario decision determinism.** Calling `list_zctas()`
twice in the same session with the same (scenario, snapshot) pair
must return identical decisions. The seed-keying scheme must include
`scenario_id` so the same ZCTA appearing in multiple scenarios (none
currently do, but the invariant protects against future overlap)
returns deterministically per scenario.

---

## 6. Error taxonomy — additions

```
UNKNOWN_SCENARIO               — F1/F6 input not in supported set
SCENARIO_NOT_DEPLOYED          — phase-2 scenario requested but not shipped
INSUFFICIENT_SCENARIOS         — F7 called with fewer than 3 scenarios completed
PENDING_MODAL_SOURCE           — F4 needs a [PENDING] source not yet ingested
INFRASTRUCTURE_EVIDENCE_INCOMPLETE  — F4's infrastructure_evidence block has nulls in required fields
PORTFOLIO_UNAVAILABLE          — F6 finds fewer than 3 scenarios deployed
```

`PENDING_MODAL_SOURCE` is the soft-error counterpart for phase-2
scenarios whose modal source manifests aren't complete yet. F4 must
return a partial result with the missing block explicitly flagged,
not a hard failure.

---

## 7. Function-call dependency graph — extended

```
F1 (select_event) ──────────► F3 (list_zctas) ──► F5 (summarize_session)
F2 (select_snapshot) ──────►          │
                                       ▼
                                 F4 (select_zcta)

F6 (list_scenarios) — independent read, used to bootstrap the session

F7 (summarize_portfolio) — independent of single-session F3/F4/F5;
                            reads server-side completed-scenario cache
```

F6 is upstream of F1 (the dashboard needs the list before the user
can pick). F7 is fully independent — it operates over a server-side
log of which scenarios have been visited, not the current session.

---

## 8. Build-phase functional gates

| Phase | Required scenarios | Required functions | Optional |
|-------|--------------------|--------------------|----------|
| Phase 1 (core, ship to SIGSPATIAL 2026) | S1, S2, S3 | F1–F7 all operational | none |
| Phase 2 (extension) | S4, S5 added | F1–F7 unchanged | `[PENDING]` modal sources fulfilled |
| Phase 3 (post-conference) | All five with same output schema | F1–F7 unchanged | cross-scenario archetype-invariance claim, with statistical backing, becomes a publishable result |

Phase 1 must ship F7 even though F7 only operates over three
scenarios — because the cross-archetype invariance claim is the
benchmark contribution, not a post-hoc add-on. Three archetypes is
the minimum that makes the claim falsifiable. Two would not.

---

## 9. What this delta does not specify

- How `infrastructure_evidence` is rendered (text, table, overlay, icon)
- How `archetype_invariance` is visualized
- Whether scenarios are presented as a top-nav, a card grid, a
  geographic chooser, or anything else
- Whether F7 runs on every session load or only on demand
- Per-scenario branding, color, or copy decisions
- Any localization beyond English

All deliberate. Implementation-side.

---

## 10. What this delta assumes from swarm-it-api

Beyond the four §8 assumptions in the prior contract:

- `/certify/geo` accepts `scenario_id` and uses it to select the
  correct embedding arm, modal source set, and seed.
- `/certify/geo/ablation` accepts scenario-specific perturbation
  parameters (drainage adjacency, pump failure, canyon routing, track
  shift, sewer-shed perturbation) — one parameter schema per scenario,
  discriminated by `scenario_id`.
- The audit block includes `scenario_id` and a list of modal-source
  ETags (one per active source) rather than a single static-table ETag.
- Pre-warmed certificates and ceilings exist for all phase-1
  scenarios at deploy time.

If any of these change, the delta changes.

---

## 11. Positioning sentence (locked)

> Floodcaster.com is a crisis-readiness diagnostic layer for flood
> forecasts, tested across five 72-hour pre-flood crisis scenarios
> spanning urban drainage flooding, protected-basin coastal
> infrastructure flooding, desert flash flooding, catastrophic
> storm-surge flooding, and dense urban cloudburst flooding.

This sentence appears in the dashboard's About page, the
FLOODCASTER_CLAIMS.md preamble, and the SIGSPATIAL 2026 demonstrator
abstract. Single source, three uses. No variants.
