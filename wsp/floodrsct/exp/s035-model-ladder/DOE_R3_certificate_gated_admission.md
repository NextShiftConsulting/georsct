# DOE Amendment: R3 Certificate-Gated Feature Admission

**Amends:** DOE_LOCKED.md v1.2
**Date:** 2026-06-05
**Status:** DRAFT
**Reason:** R0/R1/R2 results show heterogeneous, often negative uplift from
blind representation enrichment. Houston NFIP degrades monotonically
(R0=0.571 > R1=0.551 > R2=0.526). SW FL shows +146% at R2 but with a
leakage spike (0.606->0.828). The representation ladder diagnoses the
problem; R3 fixes it by gating feature admission on certificate diagnostics.

**Triggered by:** MMAR Round 2 consensus (5/5 reviewers, 2026-06-04):
- H2a INCONCLUSIVE (no cell reaches p<0.05)
- Houston primary cell contradicts the enrichment hypothesis
- kappa=0.0 pipeline bug renders DGM routing inoperable
- SW FL uplift provisional due to leakage concern
- Effective n=4 cells (classification targets fail spatial-blocked eval)

---

## Core Principle

R3 is not stepwise regression. It is **certificate-gated feature admission
implemented by DGM-controlled morph routing over candidate representation
blocks**. Candidate variables are not accepted because they add context; they
are admitted only if their RSCT certificates survive typed morph operations
that separate relevant signal from superfluous context, noise, and leakage.

**DGM is the admission controller.** Certificate-gated admission is not
separate from DGM — DGM consumes RSCT certificates and applies typed morph
operations to admit, repair, re-encode, or prune candidate feature blocks.
The admission gates (G1-G6) produce certificates; DGM reads those certificates
and routes each block to its admission outcome.

**Admission is two-dimensional: (Decision, Gear).** The enforcement pipeline
produces a typed decision (EXECUTE, RE_ENCODE, REPAIR, REJECT, BLOCK) AND a
gear state (FIRST through FOURTH, or REVERSE). The decision says WHAT to do;
the gear says HOW confidently. A block admitted at FIRST gear is qualitatively
different from one admitted at FOURTH gear with a sigma warning — the DOE
tracks both dimensions.

**Paper-safe claim:**
> GeoRSCT does not treat enrichment as inherently beneficial. DGM routes
> certificate outcomes into feature-admission actions: each candidate block
> receives an RSCT certificate, and the DGM morph controller admits, repairs,
> re-encodes, or prunes the block based on its certificate diagnostics. The
> tau-driven gearbox (P14/P16) modulates admission confidence: blocks admitted
> at lower gears carry higher certainty than those admitted under caution or
> restriction.

---

## Representation Ladder (Updated)

| Level | Name | Question |
|-------|------|----------|
| R0 | Static baseline | 30 ACS/census features. What can tabular data alone predict? |
| R1 | Spatial/hydrologic representation | +27 hydrology features. Does spatial context help? |
| R2 | Temporal event representation | +10 MRMS/HRRR features. Does temporal context help? |
| **R3_0** | **Candidate registry + graph** | What features exist, which pass boundary checks, and how do blocks interact? |
| **R3_1** | **DGM-controlled block testing** | Which blocks survive gates G1-G6? DGM morphs route each block to admit/repair/re-encode/prune. |
| **R3_2** | **Certified representation** | Retrain using only DGM-admitted blocks. Compare against R0/R1/R2. |

---

## DGM as Admission Controller

### Architecture: Gate Decision -> Morph Plan -> Operator Execution

The R3 admission pipeline reuses the production DGM stack across two repos:

```
yrsn-controlplane                          yrsn
(gate decisions)                           (morph execution)

SequentialGatekeeper.evaluate(cert)        DefaultPlanner.plan_for_decision(decision)
  -> EnforcementDecision                     -> List[PlanStep]
       |                                          |
Coordinator.coordinate(gate_result)        MorphDispatcher.execute(plan, G_R, G_S, phi)
  -> ExecutionPlan + morph_hint              -> PlanResult + OperatorWitness[]
```

**Existing code:** `compute_dgm_routing.py` (Phase 6) already imports
`MorphType` from `yrsn.core.dgm_unified` and uses controlplane thresholds.
R3_1 extends this to feature-block admission, not cell-level routing.

### Gate Decision -> Admission Mapping

The controlplane's `EnforcementDecision` maps to R3 admission actions.
Note: the controlplane uses REJECT/BLOCK, not "PRUNING" — PRUNING is a
yrsn graph operation (`MorphType.PRUNING`), not a gate decision.

| Gate Decision (`yrsn-controlplane`) | Gate Triggered | Morph Plan (`yrsn`) | R3 Admission Action |
|-------------------------------------|----------------|---------------------|---------------------|
| EXECUTE | All gates pass | `[]` (no action) | **Admit block** to R3_2 |
| WARN | Marginal quality | `[]` (proceed with elevated gear) | **Admit with flag** (diagnostic-stabilizer) |
| RE_ENCODE | Gate 3 (kappa < kappa_req) | `[ReEncode]` or `[AddVariants, AddAggregator]` | **Retest** with leakage-safe feature version |
| REPAIR | Gate 4 (kappa_L < threshold) | `[ReEncode, AddVerifier]` | **Stabilize** via ensemble/robustness test |
| REJECT | Gate 1 (N > N_thr) | `[PruneEvents]` | **Reject** block (noise-dominated) |
| BLOCK | Gate 2 (coherence < c_min) | `[AddVerifier]` | **Quarantine** block (structural incoherence) |

### Feasibility Pre-Check

Before executing a morph, DGM calls `KappaPolicyRegistry.feasibility()`:

```python
feasibility = registry.feasibility(
    current_cert=block_cert,
    candidate_policy_id=target_policy,
    estimated_kappa_improvement=0.0,  # conservative
)
if not feasibility.feasible:
    # Skip morph, go straight to REJECT
    # Log gap analysis: feasibility.gap["kappa_vs_oobleck"], etc.
```

This prevents wasting compute on morphs that cannot reach a passing state.
The gap analysis (alpha, sigma, kappa_vs_oobleck, kappa_vs_landauer) is
logged in the admission trace for post-hoc analysis.

### Morph Operator Contracts

Each morph primitive has a **directional contract** (from `yrsn.core.morphs`):

| Operator | Target Metric | Direction | Violation = |
|----------|--------------|-----------|-------------|
| RE_ENCODE | kappa_L | UP | Contract breach → abort plan |
| PRUNE_EVENTS | active_count | DOWN | — |
| ADD_VERIFIER | coherence | UP | — |
| ADD_VARIANTS | kappa_interface | UP | — |
| ADD_AGGREGATOR | coherence | UP | — |
| COVER | coverage | UP | — |

If a morph operator moves its target metric in the wrong direction, the
`MorphDispatcher` aborts the plan immediately. This is a hard safety
guarantee — no morph can make a block certificate worse on its declared axis.

### Thresholds

Thresholds come from `yrsn-controlplane` presets, not ad-hoc values:

| Threshold | Source | Value (GEOSPATIAL_CONUS27 preset) |
|-----------|--------|-----------------------------------|
| kappa_base | `ResolvedPolicy.kappa_base` | 0.22 |
| sigma_thr | `ResolvedPolicy.sigma_thr` | 0.50 |
| lambda_turbulence | `ResolvedPolicy.lambda_turbulence` | 0.0 |
| epsilon_L | `ResolvedPolicy.epsilon_L` | 0.01 |
| kappa_L_min | `ResolvedPolicy.kappa_L_min` | 0.15 |
| N_thr | `ResolvedPolicy.N_thr` | (from preset) |
| alpha_min | `ResolvedPolicy.alpha_min` | (from preset) |

The GEOSPATIAL_CONUS27 preset exists specifically for this domain. If R0/R1/R2
certificate distributions suggest tighter or looser thresholds, a new preset
is registered via `KappaPolicyRegistry` — thresholds are never hardcoded in
experiment scripts.

### Gearbox: Tau-Driven Admission Confidence (P14/P16)

Admission is not binary. The **gearbox** adds a confidence dimension to every
enforcement decision. Gear is computed from tau (temperature = 1/alpha_omega),
where alpha_omega is the reliability-weighted quality metric (P16):

```
alpha     = R / (R + N)                         # raw quality
alpha_omega = alpha * omega + prior * (1 - omega)  # blended (P16)
tau       = 1 / alpha_omega                      # temperature
gear      = tau_to_gear(tau)                     # P14 mapping
```

#### Gear States (P14)

| Gear | tau Range | Semantics | Engine | R3 Interpretation |
|------|-----------|-----------|--------|-------------------|
| FIRST | tau < 1.0 | Full autonomy | gradient | **Strong admit** — high quality, high confidence |
| SECOND | 1.0 <= tau < 1.43 | Normal operation | gradient | **Standard admit** — adequate quality |
| THIRD | 1.43 <= tau < 2.5 | Caution | eggroll | **Cautious admit** — degraded quality, flag as diagnostic-stabilizer |
| FOURTH | tau >= 2.5 | Maximum restriction | eggroll | **Marginal admit** — poor quality, report separately from headline |
| REVERSE | hash fail | Integrity violation | recovery | **Integrity reject** — certificate hash check failed (P16) |

#### Sigma Warning Gear Bump

When sigma > sigma_warning_threshold (0.4) AND the decision is EXECUTE, the
`Coordinator` bumps gear by one notch toward more restrictive:

```
FIRST  + sigma_warning → SECOND
SECOND + sigma_warning → THIRD
THIRD  + sigma_warning → FOURTH
FOURTH + sigma_warning → FOURTH  (already maximum)
```

This is the **only** mechanism that overrides tau-driven gear selection.
It prevents a block with high alpha but unstable folds from receiving
undeserved FIRST/SECOND gear.

#### Oobleck Dynamic Threshold (Gate 3)

Gate 3 does not use a flat kappa threshold. It uses the **Oobleck sigmoidal**:

```
kappa_req(sigma) = kappa_base + delta_kappa * sigmoid(steepness * (sigma - sigma_c))
```

Like shear-thickening fluid (oobleck), the threshold tightens as sigma
increases — blocks under high instability must demonstrate stronger
compatibility to pass. The Landauer tolerance (epsilon_L) creates a gray zone
where sigma breaks the tie:

| Zone | Condition | Action |
|------|-----------|--------|
| Clear pass | kappa >= kappa_req | EXECUTE |
| Landauer gray | kappa_req - epsilon_L <= kappa < kappa_req | sigma tiebreaker |
| Clear fail | kappa < kappa_req - epsilon_L | RE_ENCODE |

#### Decision x Gear Admission Matrix

Every block receives a **(Decision, Gear)** pair. The R3 admission action
depends on BOTH dimensions:

| Decision | FIRST/SECOND | THIRD | FOURTH | REVERSE |
|----------|-------------|-------|--------|---------|
| **EXECUTE** | Admit (headline) | Admit (diagnostic-stabilizer) | Admit (marginal, report separately) | Integrity reject |
| **EXECUTE + sigma_warning** | Admit (standard) | Admit (cautious) | Admit (marginal + instability flag) | Integrity reject |
| **WARN** | Admit (standard) | Admit (cautious, diagnostic-stabilizer) | Admit (marginal) | Integrity reject |
| **RE_ENCODE** | Morph cycle | Morph cycle | Morph cycle (pessimistic feasibility) | Integrity reject |
| **REPAIR** | Morph cycle | Morph cycle | Morph cycle (pessimistic feasibility) | Integrity reject |
| **REJECT** | Reject | Reject | Reject | Integrity reject |
| **BLOCK** | Quarantine | Quarantine | Quarantine | Integrity reject |

**Headline vs. diagnostic-stabilizer vs. marginal:** Only FIRST/SECOND gear
EXECUTE blocks contribute to the headline R3_2 claim. THIRD gear blocks are
reported as diagnostic-stabilizers. FOURTH gear blocks are reported separately
as marginal admits. This prevents a block with borderline certificate quality
from inflating the headline result.

#### Lyapunov Advisory Cross-Validation

The evidence layer computes an independent gear recommendation from the
Lyapunov function V (not tau):

| V Range | Lyapunov Gear Recommendation |
|---------|------------------------------|
| V < 0.2 | FIRST |
| 0.2 <= V < 0.5 | SECOND |
| 0.5 <= V < 1.0 | THIRD |
| 1.0 <= V < 1.5 | FOURTH |
| V >= 1.5 | REVERSE |

This is **advisory only** — it never blocks execution or overrides the
tau-driven gear. But divergence between tau-gear and Lyapunov-gear is logged
in the admission trace as a diagnostic signal:

- **Agreement:** tau-gear == lyapunov-gear → normal
- **Lyapunov more conservative:** lyapunov-gear > tau-gear → flag for review
  (stability verification says quality is worse than alpha suggests)
- **Lyapunov more optimistic:** lyapunov-gear < tau-gear → unusual but safe
  (system is more stable than raw quality indicates)

#### Three Stability Concepts (Orthogonal)

The enforcement pipeline distinguishes three stability concepts that are often
conflated. R3 uses all three:

| Concept | Layer | What It Checks | Blocking? | R3 Role |
|---------|-------|----------------|-----------|---------|
| **Admissibility** (E in E_adm) | Control (Gates 1-4) | RSN quality, kappa compatibility, sigma stability | **Yes** — typed decision | Primary admission gate |
| **Quality envelope** | Control (Grid A/B) | alpha x kappa or alpha x omega classification | **Yes** — informs decision | Degradation type diagnosis |
| **Lyapunov stability** (V_dot <= 0) | Evidence (Step 13) | Chordal distance derivative | **No** — advisory only | Cross-validation of gear |

#### Driving Mode

R3 uses **ECO driving mode** (from `yrsn.core.gearbox.driving_mode`):

| Parameter | ECO Value | Why |
|-----------|-----------|-----|
| Hysteresis | 0.2 | Prevents gear hunting during block iteration |
| Upshift delay | 2s | Conservative — requires sustained quality improvement before downshifting gear |
| Downshift aggression | 0.5 | Moderate — not as aggressive as SPORT (1.0), not as frozen as MANUAL (0.0) |

ECO is appropriate for scientific feature selection where stability matters
more than latency. SPORT mode (edge robotics, <5ms) and MANUAL mode
(debugging, no auto-shifting) are not appropriate for R3.

#### Dual Decision Grids

The enforcement engine selects degradation type and control action from two
grids, depending on whether kappa is available:

**Grid A: alpha x kappa (primary — RSCT solver provides kappa)**

| | kappa HIGH (>=0.6) | kappa MODERATE (0.3-0.6) | kappa LOW (<0.3) |
|---|---|---|---|
| **alpha HIGH (>=0.6)** | HEALTHY / PROCEED | HEALTHY / PROCEED_CAUTIOUS | INCOMPATIBLE / RE_ENCODE |
| **alpha MODERATE (0.3-0.6)** | DISTRACTION / FILTER_CONTEXT | DRIFT / PROCEED_CAUTIOUS | INCOMPATIBLE / RE_ENCODE |
| **alpha LOW (<0.3)** | DEGRADED_COMPATIBLE / FILTER_CONTEXT | COLLAPSE / DEFER_HUMAN | COLLAPSE / REJECT |

**Grid B: alpha x omega (fallback — kappa unavailable)**

| | omega HIGH (>=0.6) | omega MODERATE (0.3-0.6) | omega LOW (<0.3) |
|---|---|---|---|
| **alpha HIGH (>=0.6)** | HEALTHY / PROCEED | HALLUCINATION / DEFER_HUMAN | HALLUCINATION / SUPPRESS |
| **alpha MODERATE (0.3-0.6)** | DISTRACTION / FILTER_CONTEXT | DRIFT / RECALIBRATE | UNRELIABLE / GATE_ADAPTATION |
| **alpha LOW (<0.3)** | COLLAPSE / DEFER_HUMAN | COLLAPSE / REJECT | COLLAPSE / REJECT |

In R3, Grid A is always used (kappa is computed from G1-G6 gate results).
Grid B is fallback only if kappa computation fails for a block.

### Compute-Awareness: Encode/Decode Asymmetry

The representation ladder reveals a fundamental **encode/decode asymmetry**:
encode cost (feature construction) monotonically increases with representation
level, but decode cost (prediction) is nearly constant. The key insight is that
encode cost does not predict quality — Houston NFIP degrades monotonically
R0→R1→R2 despite tripling encode cost at each step.

#### Encode Cost per Block

| Level | Features | Encode Sources | Approx Cost/Block | Bottleneck |
|-------|----------|----------------|-------------------|------------|
| R0 | 30 ACS/census | Census API (cached) | ~0.1s | Network (cache hit) |
| R1 | +27 hydro | FEMA NFHL + NHDPlus + DEM (raster I/O) | ~2.5s | Disk I/O (raster reads) |
| R2 | +10 temporal | MRMS + HRRR (grib2 decode) | ~8.0s | CPU (grib2 decompression) |
| R3 | subset of R2 | Certificate lookup + gate eval | ~0.01s | Memory (certificate table) |

#### Decode Cost (Prediction)

Decode is memory-bound and nearly level-invariant: HistGBDT model size is
~identical across R0/R1/R2 (tree structure adapts to feature count, but
inference cost is dominated by tree traversal, not feature dimension).
Approximate decode cost: ~0.02s/block across all levels.

#### Instance Type Mapping

| Phase | Instance | Rationale |
|-------|----------|-----------|
| R3_0 (feature registry) | ml.m5.xlarge | S3 reads only, no compute |
| R3_1a (block tests) | ml.m5.xlarge | Model fitting, 4 vCPU sufficient |
| R3_1b (order robustness) | ml.m5.xlarge | Permutation tests, parallelizable across cores |
| R3_1c (block admission) | ml.m5.xlarge | Gate evaluation, lightweight |
| R3_2 (certified training) | ml.m5.4xlarge | Full training pipeline, 16 vCPU for parallelism |
| R3_3 (money table) | ml.m5.xlarge | Statistics computation only |

#### Paper-Safe Framing

The encode/decode asymmetry is a benchmark finding, not a machinery claim.
For the SIGSPATIAL paper: "The representation ladder reveals an encode/decode
cost asymmetry: feature construction cost increases monotonically with
representation level (0.1s → 2.5s → 8.0s per block), but prediction cost
remains constant (~0.02s). Crucially, higher encode cost does not guarantee
improved prediction — the primary cell (Houston NFIP) degrades monotonically
across levels despite tripling encode investment at each step."

No gearbox, certificate, DGM, or controlplane terminology appears in this
paragraph. It is a pure empirical observation about the cost-quality frontier.

### Two-Level DGM Operation

DGM operates at two levels in R3, both using the same gate pipeline
(`SequentialGatekeeper`) and morph stack (`DefaultPlanner` + `MorphDispatcher`):

| Level | Scope | Input | Gate Pipeline | Output |
|-------|-------|-------|---------------|--------|
| **Level 1: Cell Routing** | Per scenario-target cell | R0/R1/R2 certificates | `compute_dgm_routing.py` (Phase 6, exists) | `(EnforcementDecision, GearState)`: which representation level is certified, at what confidence |
| **Level 2: Block Admission** | Per candidate feature block | R3_1 block certificates from G1-G6 | New: extends Phase 6 to feature blocks | `(EnforcementDecision, GearState)` + morph plan + Lyapunov advisory gear |

Level 1 reuses `compute_dgm_routing.py` (currently inoperable due to
kappa=0.0 bug — prerequisite P1). Level 2 is the new R3 contribution: the
same `SequentialGatekeeper` → `Coordinator` → `DefaultPlanner` pipeline
applied to feature-block certificates, not just cell-level certificates.

Both levels use the GEOSPATIAL_CONUS27 preset from `yrsn-controlplane`
(`presets.py`). Both levels log to the DGM admission trace.

### DGM Admission Trace

Every enforcement decision is logged to `r3_dgm_admission_trace.json` with:

| Field | Description |
|-------|-------------|
| `cell` | scenario/target |
| `block` | Candidate block name |
| `certificate` | Full `YRSNCertificate` (R, S_sup, N, kappa, sigma, alpha, omega, tau) |
| `enforcement_decision` | `EnforcementDecision` enum: EXECUTE, REJECT, BLOCK, RE_ENCODE, REPAIR, WARN |
| `gate_reached` | Which gate triggered the decision (GATE_1, GATE_2, GATE_3, GATE_4, ALL_PASSED) |
| `gate_evidence` | Per-gate metrics from `GatekeeperResult` |
| `feasibility` | `MorphFeasibility` result (feasible, gap analysis) if morph attempted |
| `morph_plan` | `List[PlanStep]` from `DefaultPlanner` |
| `operator_witnesses` | `List[OperatorWitness]` — before/after values, direction satisfied |
| `terminal_decision` | Final decision after morph cycle (EXECUTE or REJECT) |
| `tau` | Temperature value (1/alpha_omega) driving gear selection |
| `gear` | `GearState` from tau_to_gear(): FIRST, SECOND, THIRD, FOURTH, REVERSE |
| `sigma_warning` | Boolean: did sigma > 0.4 trigger a gear bump? |
| `gear_after_bump` | `GearState` after sigma warning bump (if applicable) |
| `lyapunov_V` | Lyapunov function value (advisory) |
| `lyapunov_gear` | Advisory gear recommendation from V (for cross-validation) |
| `gear_agreement` | tau-gear == lyapunov-gear? (diagnostic flag) |
| `driving_mode` | ECO (for R3) |
| `grid_used` | Grid A (alpha x kappa) or Grid B (alpha x omega fallback) |
| `degradation_type` | From grid: HEALTHY, INCOMPATIBLE, DRIFT, COLLAPSE, etc. |
| `admission_action` | admit-headline, admit-stabilizer, admit-marginal, reject, quarantine |
| `admission_tier` | headline (FIRST/SECOND), diagnostic-stabilizer (THIRD), marginal (FOURTH) |
| `preset_id` | Policy preset used (e.g., GEOSPATIAL_CONUS27) |

This trace is the audit trail proving that admission decisions are
certificate-driven, gear-qualified, use the production enforcement stack,
and respect morph operator directional contracts. The (Decision, Gear) pair
is the atomic admission unit — neither dimension alone determines the outcome.

---

## R3_0: Candidate Inventory and Admissibility Screen

R3_0 produces a feature registry AND a candidate graph. **No predictive model
is trained.**

The candidate graph encodes block-level dependencies and interaction
hypotheses. Each node is a feature block; edges encode expected interactions
(e.g., hydrology <-> spatial relation, temporal <-> infrastructure). The graph
is consumed by Level 2 DGM to determine block testing order and to detect
interaction-dependent blocks that may be path-dependent (G6).

### Required Fields

| Field | Description |
|-------|-------------|
| `feature_name` | Candidate variable name |
| `source` | Dataset or derivation source |
| `scenario_coverage` | Which scenarios contain the feature |
| `temporal_status` | `pre-event`, `event-concurrent`, `post-event`, `historical-gated` |
| `spatial_grain` | ZCTA, county, catchment, pixel, structure, neighbor-lag |
| `expected_mechanism` | Why the feature should matter (causal story) |
| `missingness_rate` | Fraction missing by scenario |
| `leakage_risk` | `none`, `low`, `medium`, `high` |
| `candidate_block` | Block assignment (see R3_1) |
| `admissibility_verdict` | `ADMIT_TO_TEST`, `QUARANTINE_LEAKAGE`, `DROP_SOURCE_INVALID`, `ILLUSTRATIVE_ONLY` |

### Admissibility Rules

A feature is **not allowed** into R3_1 unless it passes temporal and causal
boundary checks:
- Post-event targets, cumulative ungated losses, and features computed using
  test-fold outcomes are quarantined.
- Quarantined features may be re-admitted if they can be recomputed per fold
  using training data only.

### Output

| Artifact | Description |
|----------|-------------|
| `r3_feature_registry.json` | Complete registry with all fields above |
| `r3_candidate_graph.json` | Block-level dependency graph (nodes=blocks, edges=interaction hypotheses) |
| `r3_admissibility_report.md` | Human-readable summary of admit/quarantine/drop decisions |

---

## R3_1: Certificate-Gated Block Testing (DGM-Controlled)

R3_1 evaluates **feature blocks**, not arbitrary one-variable sequences.

### Candidate Blocks

| Block | Examples |
|-------|---------|
| Hydrology | Catchment area, slope, TWI, drainage district, levee proximity |
| Spatial relation | W-lags, neighbor exposure, R0 residual lag, adjacency degree |
| Temporal | Rainfall peaks, storm duration, time to peak, surge-rain lag |
| Infrastructure | Hospital access, pharmacy access, road/drainage access |
| Socioeconomic | ACS, SVI, vehicle access, housing age |
| Engineering | FAST/Hazus-derived ZCTA damage estimates |

### Admission Gates

| Gate | Name | Question | Metric |
|------|------|----------|--------|
| **G1** | Causal boundary | Could this feature be known at prediction time? | Temporal status check |
| **G2** | Leakage | Does it inflate random split more than spatial-blocked? | `delta_leakage = S_sup(R2+B) - S_sup(R2)` |
| **G3** | Spatial skill | Does it improve or stabilize spatial-blocked performance? | `delta_spatial = metric(R2+B, spatial_blocked) - metric(R2, spatial_blocked)` |
| **G4** | Fold stability | Is the gain stable across folds and cells? | `fold_stability = std(delta_spatial across folds)` |
| **G5** | Solver compatibility | Does the signal survive across solvers? | `delta_solver = diag_solver(R2+B) - diag_solver(R2)` |
| **G6** | Order robustness | Does the block still matter under add/drop/reverse order? | Add-block, drop-block, block-only agreement |

### Test Battery

For each block B, run the following under **identical folds, solvers, targets,
and splits**:

| Test | Model | Purpose |
|------|-------|---------|
| Add-block | R2 + B | Marginal value over current strongest baseline |
| Drop-block | R2_all_candidates - B | Necessity once candidates interact |
| Block-only | R0 + B | Independent signal |
| Leakage-stress | Random vs spatial-blocked gap | Detects gains driven by spatial leakage |
| Transfer-stress | Leave-event-out vs spatial-blocked | Detects event-specific overfitting |
| Solver-compatibility | HistGBDT vs Ridge agreement | Detects solver-specific signal |

### DGM Enforcement Pipeline (After Gate Evaluation)

After G1-G6 are evaluated for a block, the pipeline computes an RSCT
certificate (`YRSNCertificate`) and runs it through the production
enforcement stack:

```
Block B: compute block certificate (R, S_sup, N, kappa, sigma, alpha, omega, tau)
  |
  +-- KappaPolicyRegistry.feasibility(cert, GEOSPATIAL_CONUS27)
  |     --> If infeasible: skip morph, REJECT directly
  |
  +-- SequentialGatekeeper.evaluate(cert, GEOSPATIAL_CONUS27)
  |     |
  |     +-- All gates pass --> EXECUTE
  |     +-- Gate 1 fail   --> REJECT
  |     +-- Gate 2 fail   --> BLOCK
  |     +-- Gate 3 fail   --> RE_ENCODE (Oobleck dynamic threshold)
  |     +-- Gate 4 fail   --> REPAIR
  |
  +-- Coordinator.coordinate(gate_result, tau=cert.tau, driving_mode=ECO)
  |     |
  |     +-- Compute gear from tau (P14):
  |     |     tau < 1.0   → FIRST
  |     |     tau < 1.43  → SECOND
  |     |     tau < 2.5   → THIRD
  |     |     tau >= 2.5  → FOURTH
  |     |
  |     +-- Sigma warning check:
  |     |     sigma > 0.4 AND decision == EXECUTE → bump gear +1
  |     |
  |     +-- Output: ExecutionPlan(decision, gear, sigma_warning, morph_hint)
  |
  +-- Evidence layer (advisory, non-blocking):
  |     +-- Compute Lyapunov V, V_dot from certificate
  |     +-- Lyapunov gear recommendation (independent of tau)
  |     +-- Log gear agreement/divergence
  |
  +-- Admission classification from (Decision, Gear):
  |     +-- EXECUTE @ FIRST/SECOND  → admit-headline
  |     +-- EXECUTE @ THIRD         → admit-stabilizer
  |     +-- EXECUTE @ FOURTH        → admit-marginal
  |     +-- RE_ENCODE/REPAIR        → enter morph cycle
  |     +-- REJECT/BLOCK            → reject/quarantine
  |     +-- REVERSE                 → integrity reject (any decision)
  |
  +-- MorphDispatcher.execute(plan) if RE_ENCODE or REPAIR
  |     --> Operator contracts enforced (directional guarantee)
  |     --> Re-measure certificate after morph → recompute tau → new gear
  |     --> Re-gate: if EXECUTE now, classify by new gear; if fails, reject
  |
  +-- Log full trace to r3_dgm_admission_trace.json
```

**Bounded retry:** `MorphDispatcher.execute_until_terminal()` owns the
morph→re-measure→re-gate loop. Exit conditions (priority order):
1. Contract violation (operator moved metric wrong direction) → abort
2. Gate passed (EXECUTE on re-measurement) → admit
3. Observer abort (divergence/budget exceeded) → reject
4. Plan complete without passing → reject

The trace records all attempts including intermediate certificates.

### Admission Rules

A block is **certified for R3_2** only if:

1. Spatial-blocked performance improves or remains within near-optimal tolerance
2. Leakage does not increase beyond pre-specified tolerance
3. Fold-level uplift is not dominated by a single fold
4. Solver compatibility does not collapse
5. The effect is not entirely explained by a quarantined leakage feature
6. The block remains useful under at least one order-robustness check

**Exclusion and classification labels:**
- `leakage-amplified`: Improves random split but degrades spatial-blocked
- `path-dependent`: Only helps in one block ordering
- `solver-specific`: Signal disappears under Ridge or vice versa
- `diagnostic-stabilizer`: EXECUTE @ THIRD gear — doesn't improve performance
  but reduces leakage or residual clustering (retained, reported separately)
- `marginal-admit`: EXECUTE @ FOURTH gear — borderline certificate quality,
  reported separately from headline claims
- `sigma-bumped`: Admission gear was bumped by sigma warning — instability
  flag even though enforcement decision was EXECUTE
- `lyapunov-divergent`: Lyapunov advisory gear is more conservative than
  tau-driven gear — flagged for review

### Order-Robustness Protocol

Required:
- Add-block test: R2 + B
- Drop-block test: full candidate model - B
- Block-only test: R0 + B

Recommended:
- Forward order: hydrology -> spatial -> temporal -> infrastructure -> socioeconomic -> engineering
- Reverse order: engineering -> socioeconomic -> infrastructure -> temporal -> spatial -> hydrology
- Random block-order permutations (20 seeds if compute allows)

A block is **order-robust** if its contribution remains positive, near-optimal,
or diagnostically stabilizing across the required tests. If a block only helps
in one order, it is labeled `path-dependent` and excluded from the headline
R3_2 claim.

### Output

| Artifact | Description |
|----------|-------------|
| `r3_block_admission_table.json` | Per-block: (EnforcementDecision, GearState, admission_tier, labels) |
| `r3_block_certificates.json` | Full `YRSNCertificate` per block (R, S_sup, N, kappa, sigma, alpha, omega, tau) |
| `r3_block_tests.json` | Full add/drop/block-only/leakage/transfer/solver results |
| `r3_order_robustness.json` | Forward/reverse/permutation sensitivity |
| `r3_dgm_admission_trace.json` | Complete DGM audit trail: certificate → gate → (decision, gear) → morph → outcome |
| `r3_gear_summary.json` | Per-block: tau, gear, sigma_warning, lyapunov_gear, gear_agreement, admission_tier |

---

## R3_2: Certified Representation

R3_2 is trained **from scratch** using only blocks that received DGM EXECUTE
decisions in R3_1. The feature set depends on the admission tier:

| Tier | Blocks Included | Reporting |
|------|-----------------|-----------|
| **R3_2 headline** | EXECUTE @ FIRST/SECOND gear only | Primary claim against R0/R1/R2 |
| **R3_2 full** | All EXECUTE blocks (any gear) + post-morph survivors | Secondary comparison |
| **R3_2 stabilized** | Headline + THIRD gear diagnostic-stabilizers | Stability-focused comparison |

All three variants are compared against R0, R1, and R2 using the **same folds,
targets, solvers, and split protocols**. The headline variant is the primary
claim; the other two are sensitivity analyses.

### Output

| Artifact | Description |
|----------|-------------|
| `r3_2_results.json` | Full solver metrics under spatial-blocked CV |
| `r3_2_predictions.parquet` | Spatial-blocked predictions per ZCTA |
| `r3_certificates.json` | RSN certificates for R3_2 |
| `r3_money_table.json` | R0/R1/R2/R3_2 comparison with uplift and Wilcoxon tests |
| `r3_spatial_diagnostics.json` | LISA, GWR, Geary for R3_2 |

---

## Hypotheses

### H5: Certificate-Gated Admission Improves or Stabilizes Over Blind Enrichment

**Statement:** R3_2 (certified blocks only) achieves spatial-blocked R^2 >= R2
for the primary cell (Houston NFIP) OR reduces sigma (fold volatility) while
maintaining R^2 within 5% of R2.

| Variable Type | Description |
|---------------|-------------|
| Independent | Feature set (R3_2 certified vs R2 full) |
| Dependent | Spatial-blocked R^2, sigma, S_sup |
| Control | Folds (fixed), solver (HistGBDT), target (obs_nfip_event_claims) |

**Test:** Fold-level Wilcoxon signed-rank on paired (R2_fold, R3_2_fold).
Supplemented by spatially-blocked paired loss analysis (see
Statistical-Considerations.md §14): per-ZCTA squared-error deltas
aggregated to county-level spatial blocks, tested by exact permutation
and block bootstrap. The fold-level test is primary; the spatially-blocked
test provides higher-resolution evidence with explicit independence control.
- PASS if p < 0.05 and d > 0.2 (improvement), OR
- PASS if p >= 0.05 (no degradation) AND sigma(R3_2) < sigma(R2) (stabilization)

### H6: Admitted Blocks Have Lower Leakage Than Rejected Blocks

**Statement:** Mean delta_leakage for admitted blocks < mean delta_leakage for
rejected blocks. This validates that the gate pipeline correctly separates
signal from leakage.

**Test:** Mann-Whitney U on delta_leakage(admitted) vs delta_leakage(rejected).

### H7: Block Admission Is Order-Robust

**Statement:** >= 80% of blocks receive the same admission verdict under
forward, reverse, and random orderings.

**Test:** Concordance rate across orderings. PASS if >= 80%.

### H8: Gear Discriminates Admission Quality

**Statement:** Blocks admitted at FIRST/SECOND gear (headline tier) contribute
higher spatial-blocked R^2 improvement per block than blocks admitted at
THIRD/FOURTH gear (stabilizer/marginal tier).

| Variable Type | Description |
|---------------|-------------|
| Independent | Admission tier (headline vs stabilizer/marginal) |
| Dependent | Per-block delta_spatial (spatial-blocked R^2 change when block included) |
| Control | Folds (fixed), solver (HistGBDT), target (obs_nfip_event_claims) |

**Test:** Mann-Whitney U on delta_spatial(headline_tier) vs
delta_spatial(stabilizer_marginal_tier).
- PASS if headline tier has significantly higher median delta_spatial (p < 0.05)
- PARTIAL if direction is correct but p >= 0.05 (insufficient power from small n)
- FAIL if marginal/stabilizer blocks contribute equal or higher delta_spatial

**Interpretation:** If H8 PASS, the gearbox correctly separates high-confidence
from low-confidence blocks — tau/alpha_omega is a valid admission confidence
signal. If H8 FAIL, gear adds no discriminative power and should be reported
as a diagnostic annotation, not an admission tier.

---

## Decision Tree

```
R3_1 enforcement results available: (Decision, Gear) per block, per cell
  |
  +-- All blocks → EXECUTE @ FIRST/SECOND
  |     --> R3_2 headline = R0 + all admitted blocks (high confidence)
  |     --> H5: does R3_2 improve/stabilize over R2?
  |
  +-- Mixed gears (some FIRST/SECOND, some THIRD/FOURTH)
  |     --> R3_2 headline = R0 + FIRST/SECOND blocks only
  |     --> R3_2 full = R0 + all EXECUTE blocks (any gear)
  |     --> Compare headline vs full → gear matters?
  |     --> H8: do FIRST/SECOND blocks outperform THIRD/FOURTH blocks?
  |
  +-- Mixed decisions (some EXECUTE, some REJECT/BLOCK)
  |     --> R3_2 headline = R0 + FIRST/SECOND EXECUTE blocks
  |     --> H5: does R3_2 improve/stabilize over R2?
  |     --> H6: do admitted blocks have lower leakage than rejected blocks?
  |
  +-- All blocks → REJECT or BLOCK
  |     --> R3_2 = R0 (baseline is the certified representation)
  |     --> Report: enrichment is entirely noise/leakage for this substrate
  |     --> DGM trace + gear log prove every block failed
  |
  +-- RE_ENCODE/REPAIR blocks enter morph cycle
  |     --> MorphDispatcher runs plan, re-measures, re-gates
  |     --> New (Decision, Gear) computed from post-morph certificate
  |     --> If terminal = EXECUTE: classify by post-morph gear tier
  |     --> If terminal = contract violation or budget: reject
  |
  +-- Lyapunov gear diverges from tau-gear
  |     --> Flag blocks where lyapunov_gear > tau_gear (Lyapunov more
  |         conservative than tau suggests)
  |     --> Report as diagnostic: "stability verification disagrees with
  |         quality signal for N blocks"
  |
  +-- Order-robustness fails (H7 < 80%) --> Flag path-dependence
        --> Report: block admission is sensitive to entry order;
            headline claims limited to order-robust subset
```

---

## Failure Interpretation

| Outcome | Interpretation | Paper Framing |
|---------|---------------|---------------|
| R3_2 headline > R2 | High-confidence blocks recover signal that blind enrichment dilutes | "Certificate-gated admission at FIRST/SECOND gear recovers signal lost to noise in R2" |
| R3_2 full > R3_2 headline | Marginal blocks add value beyond high-confidence set | "Lower-gear blocks contribute incremental signal; tiered admission captures a gradient" |
| R3_2 headline ~ R2, lower sigma | Diagnostic stabilization without predictive gain | "Admission gates remove unstable features without sacrificing skill" |
| R3_2 headline < R2 | Candidate context was noise; R2 or earlier is the certified level | "The correct representation is R2; additional features do not survive diagnostic gates" |
| R3_2 ~ R0 | Hydrology + temporal features are entirely superfluous at ZCTA grain | "Tabular features dominate; spatial/temporal enrichment is noise at this resolution" |
| H8 PASS | Gear discriminates: headline blocks > marginal blocks | "The tau-driven gearbox correctly identifies high-confidence feature blocks" |
| H8 FAIL | Gear adds no discriminative power | "Gear is diagnostic annotation, not a valid admission tier separator" |
| Lyapunov diverges from tau | Quality signal and stability signal disagree | "Further investigation needed: alpha-driven confidence does not match Lyapunov-verified stability" |

---

## Compute Requirements

### R3_0 (no compute)
- Manual/scripted feature inventory from S3 metadata
- Output: JSON registry

### R3_1 (primary compute)
- Per block: 6 tests x 5 scenarios x 3 targets x 5 folds = 450 model fits per block
- 6 candidate blocks = 2,700 model fits
- Order robustness (forward + reverse + 20 random): +22 orderings x 6 blocks x 5 scenarios = 660 additional
- **Total: ~3,360 model fits**
- HistGBDT + Ridge per fit = ~6,720 solver runs
- Estimated wall clock: 4-6 hours on ml.m5.xlarge (parallel across scenarios)

### R3_2 (single rerun)
- Same as R0/R1/R2 training: 5 scenarios x 3 targets x 5 folds x 2 solvers = 150 fits
- Plus spatial diagnostics sidecar: ~30 min
- Plus certificates + money table: ~10 min

---

## Constraints

1. **Folds are frozen.** R3 uses the same spatial-blocked fold assignments as R0/R1/R2.
2. **Solvers are frozen.** HistGBDT (primary) + Ridge (compatibility check).
3. **Targets are frozen.** obs_nfip_event_claims (regression), obs_has_311 (classification), obs_has_hwm (classification).
4. **No feature engineering in R3_1.** Blocks use existing features from R0/R1/R2; no new derived features.
5. **No hyperparameter tuning per block.** Same solver defaults across all tests.
6. **Order robustness is required, not optional.** A block that only helps in one order is path-dependent.
7. **Gear thresholds are frozen.** P14 tau boundaries (1.0, 1.43, 2.5) are production constants, not tuned per experiment.
8. **Driving mode is ECO.** No SPORT (too aggressive) or MANUAL (no auto-shifting) for scientific feature selection.
9. **GEOSPATIAL_CONUS27 preset is frozen.** Oobleck parameters (kappa_base=0.22, lambda=0.0, epsilon_L=0.01) come from the registered preset, not per-block tuning.
10. **Lyapunov is advisory only.** Lyapunov gear recommendations are logged for cross-validation but never override tau-driven gear or block admission decisions.

---

## Future Work: Cost-Aware Routing (Post-NeurIPS)

The current DGM stack has **zero cost awareness** — neither yrsn nor
yrsn-controlplane have compute-cost, budget, or latency fields in any modern
class (Navigator, GearSpec, MorphRecord, ExecutionPlan, ResolvedPolicy). The
encode/decode asymmetry documented above motivates five deferred extensions:

1. **`encode_cost` in NavigatorConfig**: Allow the navigator to factor feature
   construction cost into routing decisions. A block with 3x encode cost and
   marginal quality uplift should be deprioritized.

2. **`latency_budget` in ResolvedPolicy**: Gate evaluation already produces
   (Decision, Gear); adding a latency constraint would allow the coordinator
   to downgrade gear when the remaining time budget is tight.

3. **Cost-weighted morph selection in DefaultPlanner**: When multiple morph
   plans can reach a passing state, prefer the plan with lower encode cost.
   Currently DefaultPlanner selects by operator signature, not cost.

4. **Gear-aware batch sizing in MorphDispatcher**: FOURTH gear blocks could
   be batched more aggressively (larger batches, lower per-block overhead)
   since they are marginal admits anyway.

5. **Encode/decode split metrics in EvidenceTelemetry**: Log encode_time and
   decode_time separately in the Lyapunov advisory layer so that cost-quality
   Pareto frontiers can be computed post-hoc.

These are **deferred** — R3 uses the existing stack without modification. The
encode/decode table above provides the empirical foundation for prioritizing
these extensions after the NeurIPS 2026 submission.

---

## Relationship to MMAR Findings

| MMAR Finding | R3 Response |
|-------------|-------------|
| Houston degrades R0>R1>R2 | R3_1 will identify which R1/R2 blocks cause degradation (drop-block test) |
| SW FL leakage spike at R2 | G2 (leakage gate) will test whether temporal block inflates leakage |
| kappa=0.0 pipeline bug | **Critical prerequisite**: fix kappa wiring before R3_1. DGM morph routing depends entirely on valid kappa — Level 1 (cell routing) AND Level 2 (block admission) are inoperable without it. |
| Pooled Wilcoxon NaN | R3 prerequisites: filter NaN before pooling in R3 money table |
| n=4 effective cells | R3_1 runs per-cell; classification cells contribute G4 fold-stability data even if primary metric is NaN |
| Fig 5/6 caption mismatch | R3_2 spatial diagnostics will produce correct LISA/GWR for certified representation |

---

## Prerequisites (Must Complete Before R3_1)

| # | Task | Source | Effort | DGM Impact |
|---|------|--------|--------|------------|
| P1 | Fix kappa propagation (wire diag_leakage as proxy) | MMAR Issue 3 | Pipeline fix | **Blocks all DGM**: Level 1 cell routing + Level 2 block admission both depend on valid kappa |
| P2 | Fix pooled Wilcoxon NaN filtering | MMAR Issue 4 | One-line fix | R3 money table reuses this code |
| P3 | Rewrite Fig 5/6 captions with actual data | MMAR Issue 6 | Text edit | None (paper only) |
| P4 | Add leakage discussion appendix | MMAR Issue 5 | New LaTeX subsection | Needed to contextualize DGM RE_ENCODE decisions |
| P5 | Disclose n=4 effective cell count | MMAR Issue 1 | Text edit | DGM Level 1 operates on n=4 cells, not n=9 |
| P6 | Validate DGM morph thresholds against R0/R1/R2 certificate distributions | New | Analysis | Ensures kappa_good/kappa_bad/sigma boundaries are empirically grounded |

---

## Change Control

| Version | Date | Change | Reason |
|---------|------|--------|--------|
| v1.0 | 2026-06-05 | Initial R3 DOE: Certificate-Gated Feature Admission | MMAR Round 2 consensus: blind enrichment fails |
| v1.1 | 2026-06-05 | DGM integration: morph-to-admission mapping, two-level DGM, candidate graph, admission trace artifact | DGM IS the admission controller, not separate from it |
| v1.2 | 2026-06-05 | Align with production DGM stack: use EnforcementDecision (not PRUNING), add feasibility pre-check, morph operator contracts, GEOSPATIAL_CONUS27 preset, MorphDispatcher bounded retry | DOE must reference actual yrsn + yrsn-controlplane architecture |
| v1.3 | 2026-06-05 | Gearbox integration: (Decision, Gear) as atomic admission unit, tau-driven gear states (P14/P16), sigma warning bump, Oobleck dynamic threshold, Lyapunov advisory cross-validation, ECO driving mode, dual decision grids, three R3_2 variants (headline/full/stabilized), H8 hypothesis, admission tiers | Patent-aligned: tau/gear/Lyapunov/Oobleck are all patented mechanisms exercised in R3 |
| v1.4 | 2026-06-05 | Compute-awareness: encode/decode asymmetry table, instance type mapping, paper-safe framing paragraph. Future work: 5 deferred cost-aware routing items (post-NeurIPS). | Encode cost triples R0->R2 but quality doesn't follow; zero cost awareness in current stack confirmed |
