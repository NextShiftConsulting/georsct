# Metric Implementation: Kappa Registry, Gearbox, and Six Geometries

**Date**: 2026-05-29
**Scope**: How geometry-specific kappa proxies integrate with the existing RSCTController, gearbox, enforcement engine, and hex arch repo boundaries.
**Prerequisite reading**: AC_KAPPA_REGISTRY.md (registry audit), AC_SIX_KAPPAS.md (per-proxy criteria), AC_CLAIMS.md (figure claims)

---

## 1. RSCTController + Gearbox Architecture

`RSCTController` (`yrsn/core/decomposition/rsct_integration.py:195`) is the orchestration hub wiring three components:

1. **AdaptiveGearbox** (`adapters/gearbox/adaptive.py`) — 5-gear transmission mapping kappa to operational modes
2. **GearboxCurriculum** — difficulty progression (from sudoku experiments)
3. **PrioritizedReplayBuffer** (`core/memory/replay.py`) — TD-error-weighted experience replay with memristive decay

### 1.1 The Gearbox

The gearbox is a literal transmission metaphor. Each gear controls:
- **Engine type**: gradient (exploitation) vs eggroll (exploration)
- **Dimensional view**: which T4 dimensions are visible
- **Temperature range**: tau determines the gear

| Gear | Name | Engine | View | tau range |
|------|------|--------|------|-----------|
| 1st | Precision | gradient | (alpha, omega) 2D | [0, 1.0) |
| 2nd | Balanced | gradient | (theta, phi, alpha) 3D | [1.0, 1.43) |
| 3rd | Explore | eggroll | (theta, phi) polar | [1.43, 2.5) |
| 4th | Discovery | eggroll | full T4 | [2.5, inf) |
| R | Recovery | eggroll | (alpha, omega) 2D | [0, inf) |

### 1.2 Shifting Logic

**Source**: `core/gearbox/shifter.py` — `GearShifter` (stateless, pure domain logic)

- Upshifts: sequential only (1st->2nd->3rd, no skipping)
- Downshifts: always allowed (any gear to any lower gear)
- REVERSE: accessible from any gear (emergency collapse recovery)
- Driving modes control hysteresis and delay:
  - **Sport**: hysteresis=0.05, upshift_delay=0.0, downshift_aggression=1.0
  - **Eco**: hysteresis=0.2, upshift_delay=2.0, downshift_aggression=0.5
  - **Manual**: no auto-shifting

### 1.3 How Kappa Drives the Gearbox

```
RSCTController.update_gear(kappa)
  -> RSCTQualitySignals.from_kappa(kappa)
    -> QualitySignals(alpha=kappa, tau=f(kappa), collapse_risk=...)
      -> gearbox.auto_shift(quality)
        -> GearShifter.decide_shift(current_gear, quality, mode_config)
          -> ShiftDecision(should_shift, from_gear, to_gear, reason)
```

**Direction**: Low kappa = high tau = upshift toward exploration. High kappa = low tau = downshift toward precision.

### 1.4 Enforcement Engine Integration

`UnifiedEnforcementEngine` (`enforcement/engine.py`) uses kappa at **Step 7.5** — a Layer 2 compatibility check:

- **Series circuit**: `kappa_critical = min(kappa_A, kappa_E, kappa_T)` — weakest link blocks
- **Oobleck dynamic thresholding**: `required_kappa = kappa_base + sensitivity * sigma`
- **Landauer tolerance gray zone**: `[threshold - epsilon_L, threshold)` — sigma is tie-breaker in the gray zone
- **Multimodal checks**: `kappa_interface`, `kappa_H`, `kappa_L`, `sigma_H`, `sigma_L`

### 1.5 Replay Buffer

`PrioritizedReplayBuffer` (`core/memory/replay.py`) — priority-weighted experience replay:

- **SumTree**: O(log n) sampling proportional to priority
- **Priority formula**: `(|TD_error| + epsilon)^alpha`
- **Memristive decay**: priorities leak over time (`priority *= decay_rate`)
- **Similarity boosting**: recurring patterns reinforce (cosine similarity > threshold -> boost)
- **Config**: capacity=10000, alpha=0.6, beta=0.4 (importance sampling), batch_size=32

---

## 2. Hexagonal Architecture and Repo Separation

### 2.1 Three Planes (ADR-014)

```
TRAINING PLANE (yrsn, yrsn-training)
  Produces frozen artifacts — checkpoints, weights, configs
  Learning: YES

CONTROL PLANE (yrsn-controlplane)
  Promotes artifacts into deployment epochs
  Learning: NO
  Owns: SequentialGatekeeper, KappaPolicyRegistry, DeploymentEpoch

RUNTIME PLANE (yrsn DGM, yrsn-orchestration)
  Certify, gate, route, morph, log
  Learning: NO — bounded structural adaptation only
```

### 2.2 Four Layers (ADR-010)

| Layer | Patent Figure | Repo | Owns |
|-------|--------------|------|------|
| **Measurement** | FIG. 36, 26 | yrsn-orchestration | Certificate triple (alpha, kappa, sigma), R/S/N |
| **Control** | FIG. 24 | yrsn-controlplane | SequentialGatekeeper — sole gate authority (ADR-004) |
| **Coordination** | FIG. 37 | yrsn-controlplane | Decision -> execution plan mapping |
| **DGM** | FIG. 25 | yrsn | Graph modification, morph dispatch, re-certification loop |

### 2.3 Where Each Component Lives

**Per ADR-006 sub-decision 3: gearbox is owned by Coordination (DGM), not Gatekeeper.**

```
yrsn (core domain)
├── core/kappa/              ← DOMAIN: All kappa computations (15 registered)
│   ├── compat/              kappa_compat: R*(1-N) — Gate 3 primary
│   ├── gate/                kappa_compat_cosine + kappa_modal_min
│   ├── req/                 kappa_req: sigmoid oobleck threshold
│   ├── difficulty_theory/   kappa_difficulty: clip(D*/D, 0, 1)
│   ├── empirical/           kappa_empirical: domain-calibrated
│   ├── composite/           kappa_composite: multi-primitive bottleneck
│   ├── priority/            kappa_priority: replay priority from degradation
│   ├── diagnostic_simplex/  kappa_simplex: (A, E, T) decomposition
│   ├── residual_agreement/  kappa_residual_agreement: cross-family Spearman rho
│   ├── smooth/              kappa_smooth: 1 - CV(R2) leave-one-out
│   ├── spatial/             kappa_spatial: Moran's I normalized
│   ├── hierarchical/        kappa_hierarchical: 1 - eta_squared
│   ├── logical/             kappa_logical: Fisher ratio on adjacency
│   ├── periodic/            kappa_periodic: spectral concentration
│   ├── proxy_simplex/       kappa_proxy: UNREGISTERED (Claim 2 violation, s018 only)
│   ├── registry.py          Central lookup: get_kappa(), list_kappas()
│   └── types.py             KappaResult, KappaMetadata, KappaStatus
│
├── core/gearbox/            ← DOMAIN LOGIC (pure, no deps)
│   ├── gear.py              Gear enum, GearSpec, DimensionalView, GEAR_SPECS
│   ├── shifter.py           GearShifter — recommend_gear, decide_shift (stateless)
│   ├── driving_mode.py      DrivingMode enum, configs (Sport/Eco/Manual)
│   ├── views.py             ViewProjector — T4 dimensional projections
│   └── gearbox_calibrator.py GearboxCalibrator — learns thresholds from tau distribution
│
├── ports/gearbox.py         ← PORT (IGearbox protocol, GearboxState, EnforcementResult)
│
├── adapters/gearbox/        ← ADAPTER (implements IGearbox)
│   └── adaptive.py          AdaptiveGearbox — stateful, holds current gear + tau
│
├── core/memory/replay.py    ← DOMAIN (PrioritizedReplayBuffer, SumTree, Experience)
│
├── core/decomposition/
│   └── rsct_integration.py  ← WIRING (RSCTController, RSCTQualitySignals, DStarEstimate)
│                                Stitches gearbox + curriculum + replay
│
├── core/enforcement/
│   └── engine.py            ← UnifiedEnforcementEngine (BEING DECOMPOSED per ADR-006)
│                                Step 7.5: kappa compatibility check
│                                Gearbox integration via RSCTQualitySignals
│
└── framework/factories.py   ← FRAMEWORK (create_gearbox, create_rsct_controller,
                                           create_enforcement_engine)

yrsn-controlplane (separate repo — ADR-004 sole gate authority)
├── SequentialGatekeeper     5 gates, Oobleck threshold, Landauer zones
│                            Reads kappa_compat as input, never modifies it
├── GatekeeperConfig         Gate thresholds (kappa_base, lambda_turbulence, etc.)
├── CPGatekeeperInput        Input type — owns the kappa_compat: float field
├── EnforcementDecision      Output type (EXECUTE/REJECT/BLOCK/RE_ENCODE/REPAIR)
└── KappaPolicyRegistry      Policy resolution (content-addressed sha256)

yrsn-orchestration (measurement layer)
├── CertifyService           Wires measurement -> control
├── RepresentationContractPort  G_R measurement (R/S/N -> alpha)
├── CompatibilityContractPort   G_S measurement (kappa, sigma, c)
└── TreeCompatibilityAdapter    Concrete adapter for S017 tree
```

### 2.4 Boundary Rules (ADRs)

| Rule | ADR | Implication |
|------|-----|-------------|
| **Sole gate authority** | ADR-004 | No production gate outcome outside yrsn-controlplane's SequentialGatekeeper |
| **Measurement never evaluates gates** | ADR-010 B1 | yrsn-orchestration produces certificates, controlplane consumes them |
| **Control never modifies graphs** | ADR-010 B2 | SequentialGatekeeper reads certificate, returns typed decision, touches nothing else |
| **No cross-imports** | ADR-015 | yrsn core MUST NOT import yrsn-controlplane; vice versa. Bridge at adapter layer. |
| **Runtime non-learning** | ADR-014 | Gearbox may do bounded structural adaptation (gear shifts) but NEVER learning (no gradients, no threshold tuning) |
| **Gearbox owned by DGM** | ADR-006 SD3 | Gearbox is Coordination/DGM, not Gatekeeper. Reacts to gate outcomes, doesn't produce them. |
| **Policy is content-addressed** | ADR-005 | policy_id = sha256 of thresholds + overrides. Runtime reads policy, never writes. |

---

## 3. Six Geometries Integration Path

### 3.1 The Key Insight

**Geometry-specific kappa does NOT require controlplane changes.** It is a measurement-layer concern.

The controlplane already consumes `CPGatekeeperInput.kappa_compat: float`. It does not care whether that float came from `R*(1-N)`, cosine similarity, or Moran's I. The dispatch (which kappa formula to use for which task) happens upstream in measurement, not at the gate.

### 3.2 Data Flow: Geometry Kappa Through the System

```
1. MEASUREMENT (yrsn-orchestration / theory_certifier.py)
   ├── certify_group() runs the existing pipeline (steps 1-4)
   ├── Step 5 dispatches to geometry-specific kappa:
   │     geometry_dispatch[geometry_type](residuals, coords, adjacency, ...)
   │     -> KappaResult (value in [0,1])
   └── Stamps result on YRSNCertificate
         cert.kappa_compat = geometry_kappa_result.value

2. PROJECTION (adapter boundary)
   ├── certificate -> CPGatekeeperInput projection
   │     CPGatekeeperInput.kappa_compat = cert.kappa_compat
   └── Geometry type recorded in cert.extras for provenance

3. CONTROL (yrsn-controlplane — UNCHANGED)
   ├── SequentialGatekeeper.evaluate(input) — same 5 gates
   │     Gate 3: kappa_compat vs kappa_req(sigma)
   └── Returns EnforcementDecision (EXECUTE/REJECT/RE_ENCODE/...)

4. DGM / COORDINATION (yrsn — post-gate)
   ├── If EXECUTE: done
   ├── If RE_ENCODE:
   │     RSCTController.update_gear(kappa)
   │       -> RSCTQualitySignals -> QualitySignals
   │       -> gearbox.auto_shift()
   │       -> Gear determines exploration vs exploitation for re-encoding
   └── Geometry-specific kappa drives gear behavior:
         Low kappa_spatial -> high tau -> upshift (more exploration)
         High kappa_smooth -> low tau -> stay in precision mode
```

### 3.3 What Changes Where

| Component | Repo | Change Required | ADR Compliance |
|-----------|------|-----------------|----------------|
| `yrsn/core/kappa/*` | yrsn | DONE — 6 geometry proxies registered | Measurement domain |
| `theory_certifier.py` | yrsn-orchestration | ADD geometry_type dispatch at step 5 | ADR-010: measurement owns kappa computation |
| `CPGatekeeperInput` | yrsn-controlplane | NONE — already accepts kappa_compat: float | ADR-004: gate interface unchanged |
| `SequentialGatekeeper` | yrsn-controlplane | NONE — reads kappa_compat, doesn't care about source | ADR-004: sole authority, no changes needed |
| `RSCTQualitySignals` | yrsn | NONE — already converts any kappa in [0,1] to quality signals | ADR-006: gearbox in DGM layer |
| `AdaptiveGearbox` | yrsn | NONE — already reacts to any QualitySignals | ADR-014: bounded structural adaptation |
| `cert.extras` | yrsn | ADD geometry provenance metadata | ADR-024: enforcement provenance on extras |
| `KappaPolicyRegistry` | yrsn-controlplane | FUTURE — per-geometry threshold overrides | ADR-005: policy is content-addressed |

### 3.4 Geometry-Specific Gearbox Behavior (Emergent, Not Coded)

The gearbox doesn't need geometry-specific logic. Different kappa values from different geometries naturally produce different gear behavior:

| Geometry | Typical Kappa Pattern | Expected Gear Behavior |
|----------|----------------------|------------------------|
| Smooth (high R2 agreement) | High kappa -> low tau | Stays in 1st/2nd (precision) |
| Spatial (weak autocorrelation) | Low kappa -> high tau | Shifts to 3rd/4th (exploration) |
| Hierarchical (cross-level generalization) | Medium kappa | Stays in 2nd (balanced) |
| Composite (cross-family disagreement) | Variable | Gear oscillation -> Eco mode dampens |
| Logical (weak adjacency signal) | Low kappa | Shifts to 3rd (explore new representations) |
| Periodic (strong spectral peak) | High kappa | Stays in 1st (precision) |

This is emergent behavior from the existing tau-to-gear mapping — no geometry-specific gearbox code is needed. The kappa value itself carries the geometry information.

---

## 4. Testing Requirements

### 4.1 What Exists (Tested)

| Component | Test File | Tests | Status |
|-----------|-----------|-------|--------|
| kappa_residual_agreement | `tests/unit/test_kappa_residual_agreement.py` | 19 | PASS |
| kappa_smooth | `tests/unit/test_kappa_smooth.py` | 23 | PASS |
| kappa_spatial | `tests/unit/test_kappa_spatial.py` | 19 | PASS |
| kappa_hierarchical | `tests/unit/test_kappa_hierarchical.py` | 21 | PASS |
| kappa_logical | `tests/unit/test_kappa_logical.py` | 24 | PASS |
| kappa_periodic | `tests/unit/test_kappa_periodic.py` | 16 | PASS |
| **Total** | | **122** | **ALL PASS** |

These test the kappa computation in isolation. They do NOT test the signal path through the gearbox or the projection to controlplane.

### 4.2 What's Missing (Must Test)

#### A. Kappa-to-Gearbox Signal Path

**Test**: Feed geometry-specific kappa values through `RSCTQualitySignals.from_kappa()` -> `QualitySignals` -> `GearShifter.decide_shift()` and verify gear selection is sane.

```python
# Proposed: tests/integration/test_kappa_gearbox_signal_path.py

class TestGeometryKappaGearboxPath:

    def test_high_smooth_kappa_stays_precision(self):
        """High kappa_smooth (0.95) should keep gearbox in low gear."""
        result = compute_kappa_smooth(r2_values={"pca": 0.9, "lag": 0.88, "gnn": 0.91})
        signals = RSCTQualitySignals.from_kappa(result.value)
        quality = signals.to_gearbox_signals()
        # tau should be low -> 1st or 2nd gear
        assert quality.tau < 1.43  # below Explore threshold

    def test_low_spatial_kappa_triggers_exploration(self):
        """Low kappa_spatial (0.2) should push gearbox toward exploration."""
        result = compute_kappa_spatial(residuals=..., adjacency=...)
        signals = RSCTQualitySignals.from_kappa(result.value)
        quality = signals.to_gearbox_signals()
        # tau should be high -> 3rd or 4th gear
        assert quality.tau >= 1.43  # Explore or Discovery

    def test_collapse_kappa_triggers_reverse(self):
        """Near-zero kappa should trigger REVERSE gear."""
        signals = RSCTQualitySignals.from_kappa(0.01)
        quality = signals.to_gearbox_signals()
        assert quality.collapse_risk > 0.8

    def test_all_six_geometries_produce_valid_signals(self):
        """Every geometry proxy output maps to a valid QualitySignals."""
        for proxy_name in [
            "kappa_residual_agreement", "kappa_smooth", "kappa_spatial",
            "kappa_hierarchical", "kappa_logical", "kappa_periodic",
        ]:
            family = get_kappa(proxy_name)
            # Compute with fixture data -> value in [0,1]
            # Convert to signals -> no exceptions, valid tau
```

#### B. Projection Bridge (Certificate -> CPGatekeeperInput)

**Test**: Verify that geometry-specific kappa on a `YRSNCertificate` projects correctly into `CPGatekeeperInput.kappa_compat`.

```python
# Proposed: tests/integration/test_kappa_projection_bridge.py

class TestKappaProjectionBridge:

    def test_geometry_kappa_reaches_gatekeeper_input(self):
        """Geometry kappa set on certificate should appear as kappa_compat in CP input."""
        cert = YRSNCertificate(R=0.6, S_sup=0.3, N=0.1)
        # Simulate: measurement stamps geometry kappa
        cert.kappa_compat = 0.73  # from kappa_spatial
        # Project to controlplane input
        cp_input = project_to_cp_input(cert)
        assert cp_input.kappa_compat == 0.73

    def test_provenance_in_extras(self):
        """Geometry type should be recorded in cert.extras for audit."""
        cert = YRSNCertificate(R=0.6, S_sup=0.3, N=0.1)
        cert.extras["kappa_geometry"] = {
            "type": "spatial",
            "proxy": "kappa_spatial",
            "formula": "Moran's I normalized",
            "value": 0.73,
        }
        # Verify extras survive serialization
        d = cert.to_dict()
        assert d["extras"]["kappa_geometry"]["proxy"] == "kappa_spatial"
```

#### C. Boundary Compliance

**Test**: Verify ADR boundary rules are not violated.

```python
# Proposed: tests/compliance/test_kappa_boundary_compliance.py

class TestBoundaryCompliance:

    def test_kappa_modules_do_not_import_controlplane(self):
        """ADR-015: yrsn core MUST NOT import yrsn-controlplane."""
        import ast
        for path in glob("src/yrsn/core/kappa/**/*.py"):
            tree = ast.parse(open(path).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "controlplane" not in node.module, (
                        f"{path} imports from controlplane — ADR-015 violation"
                    )

    def test_kappa_modules_do_not_evaluate_gates(self):
        """ADR-010 B1: measurement never evaluates gates."""
        # Grep for SequentialGatekeeper or evaluate() in kappa/
        # Should find zero hits
```

#### D. End-to-End: Geometry Kappa Changes Gate Outcome

**Test**: Show that using a geometry-specific kappa produces a different gate outcome than universal kappa on the same task.

```python
# Proposed: tests/integration/test_geometry_gate_outcome_flip.py

class TestGeometryGateOutcomeFlip:

    def test_spatial_kappa_flips_gate_decision(self):
        """A task where kappa_spatial != kappa_compat produces a different gate verdict."""
        # Same certificate base, different kappa sources
        cert_universal = CPGatekeeperInput(kappa_compat=0.72, sigma=0.3, ...)
        cert_spatial = CPGatekeeperInput(kappa_compat=0.41, sigma=0.3, ...)

        gk = SequentialGatekeeper(config)
        result_universal = gk.evaluate(cert_universal)
        result_spatial = gk.evaluate(cert_spatial)

        # Universal passes, spatial fails (or vice versa)
        assert result_universal.decision != result_spatial.decision
```

### 4.3 Test Tiers

| Tier | What | Where | Count (est) | Priority |
|------|------|-------|-------------|----------|
| **T1: Unit** | Each proxy computes correctly | `tests/unit/test_kappa_*.py` | 122 (DONE) | DONE |
| **T2: Signal path** | Kappa -> QualitySignals -> gear shift | `tests/integration/test_kappa_gearbox_signal_path.py` | ~12 | HIGH |
| **T3: Projection** | Certificate -> CPGatekeeperInput bridge | `tests/integration/test_kappa_projection_bridge.py` | ~6 | HIGH |
| **T4: Boundary** | ADR compliance (no cross-imports) | `tests/compliance/test_kappa_boundary_compliance.py` | ~4 | MEDIUM |
| **T5: Gate flip** | Geometry kappa changes gate outcome | `tests/integration/test_geometry_gate_outcome_flip.py` | ~6 | HIGH |
| **T6: Redundancy** | Pairwise Spearman < 0.9 on CONUS-27 | `tests/integration/test_kappa_redundancy.py` | ~15 | CRITICAL |

---

## 5. Kappa Registry: Final State

**Post-consolidation (2026-05-29, commit 4a1e743)**

```
CANONICAL (5):
  kappa_compat              R*(1-N)                     Gate 3 primary
  kappa_compat_cosine       cosine(event, solver)       Routing-layer compatibility
  kappa_modal_min           min(H, L, interface)        Multimodal weakest-link
  kappa_req                 sigmoid oobleck threshold   Gate 3 dynamic threshold
  kappa_difficulty          clip(D*/D, 0, 1)            Theory kappa

OPERATIONAL (3):
  kappa_composite           min(prims) * penalty        Multi-primitive bottleneck
  kappa_empirical           R*K_R + S*K_S + N*K_N       Domain-calibrated accuracy
  kappa_priority            degradation-based replay    Experience replay priority

DIAGNOSTIC (1):
  kappa_simplex             (A, E, T) decomposition     Post-hoc bottleneck

EXPERIMENTAL (6):
  kappa_residual_agreement  Spearman rho cross-family   Composite geometry
  kappa_smooth              1 - CV(R2) LOO              Smooth geometry
  kappa_spatial             Moran's I normalized        Spatial geometry
  kappa_hierarchical        1 - eta_squared             Hierarchical geometry
  kappa_logical             Fisher ratio on adjacency   Logical geometry
  kappa_periodic            spectral concentration      Periodic geometry
```

**15 registered, 0 unregistered, 0 duplicates, 0 shadow implementations.**

### 5.1 What Was Eliminated

| Function | Location | Action | Reason |
|----------|----------|--------|--------|
| `compute_kappa_compat` | `routing/router.py` | DELETED | Shadow of kappa_compat_cosine; logic inlined into `kappa/gate/compute.py` |
| `compute_kappa_req` | `routing/router.py` | DELETED | Shadow of kappa_req; `kappa/req/compute.py` now calls `oobleck_threshold` directly |
| `compute_kappa` | `decomposition/rsct_integration.py` | DELETED | Shadow of kappa_difficulty; zero external callers (dead code) |
| `compute_kappa_from_result` | `decomposition/rsct_integration.py` | DELETED | Convenience wrapper; zero external callers (dead code) |
| `compute_kappa_req_tensor` | `compression/multi_geometry_loss.py` | DELETED | One-liner wrapper; callers now use `oobleck_threshold_tensor` directly |
| `kappa_proxy` | `kappa/proxy_simplex/` | UNREGISTERED | Claim 2 violation (R/(R+N) = alpha). Module retained for s018a-g reproduction only. |

### 5.2 Bugs Fixed

| Bug | Fix |
|-----|-----|
| `kappa_compat` overwrite: gate/ silently overwrote compat/ registration | Split into `kappa_compat` (R*(1-N)) and `kappa_compat_cosine` (cosine). Separate registry keys. |
| `kappa_modal_min` never registered | Now separately registered with its own metadata |
| `compute_kappa_compat_cosine` returned `name="kappa_compat"` | Fixed to return `name="kappa_compat_cosine"` |

---

## 6. Open Questions

| # | Question | Impact | Owner |
|---|----------|--------|-------|
| Q1 | Should `CPGatekeeperInput` gain an optional `kappa_geometry_type: str` field for provenance? | Audit trail for which proxy was used. Not required for gating. | controlplane |
| Q2 | Should `KappaPolicyRegistry` support per-geometry threshold overrides? | E.g., spatial tasks might need lower kappa_base. | controlplane (ADR-005) |
| Q3 | How does `theory_certifier.py` know the geometry type? User-specified? Auto-detected? | Dispatch mechanism — currently does not exist. | orchestration |
| Q4 | Should geometry kappa REPLACE `kappa_compat` or sit alongside as `kappa_geometry`? | Determines whether gate behavior changes or geometry kappa is advisory-only. | architecture decision |
| Q5 | How do the 6 geometry types map to the 12-mode taxonomy (GEO-A through GEO-D)? | Two unrelated taxonomies in one paper is confusing. | paper |

---

## 7. ADR Reference Index

| ADR | Title | Relevance |
|-----|-------|-----------|
| ADR-004 | Controlplane Sole Gate Authority | SequentialGatekeeper is the only gate evaluator. Geometry kappa feeds in as kappa_compat. |
| ADR-005 | KappaPolicyRegistry | Policy is content-addressed. Future: per-geometry threshold overrides. |
| ADR-006 | UEE Decomposition | Gearbox owned by DGM/Coordination, not Gatekeeper. |
| ADR-010 | Four-Layer Boundary Enforcement | Measurement -> Control -> Coordination -> DGM. Kappa computation is measurement. |
| ADR-014 | Runtime Non-Learning Boundary | Gearbox may do structural adaptation but never learning. |
| ADR-015 | Certificate Type Separation | yrsn core must not import controlplane. Bridge at adapter layer. |
| ADR-024 | Enforcement Provenance on Certificate | Geometry metadata goes in cert.extras["kappa_geometry"]. |

---

## 8. Six Geometries in Flood Crisis Management

Each geometry type asks a fundamentally different question about **why a flood prediction model fails or succeeds**. A universal kappa says "this model is 0.7 compatible" — but *why* is it 0.3 short? Each geometry diagnoses a different failure mode, which implies a different remediation path for a crisis manager.

### 8.1 The Six Questions

| Geometry | Proxy | Question for Flood Crisis | Concrete Example |
|----------|-------|--------------------------|-------------------|
| **Smooth** | `kappa_smooth` (1 - CV(R2) LOO) | "Do all our embedding families agree on flood risk?" | If PCA, GNN, and lag embeddings all produce similar R2 on FEMA flood zone prediction, the task is representationally stable — any embedding works. Low kappa_smooth = the choice of representation matters. The crisis manager should not trust a single-model output. |
| **Spatial** | `kappa_spatial` (Moran's I) | "Does the model's error have geographic structure?" | If flood depth prediction errors cluster spatially (high Moran's I) — e.g., the model systematically fails in coastal floodplains but succeeds inland — the model has a spatial blind spot. Certificate should flag: "this model is unreliable in Zone X." |
| **Composite** | `kappa_residual_agreement` (Spearman rho) | "Do different model families fail on the same locations?" | If PCA and GNN embeddings both fail on the same ZCTAs, that's a **data problem** (unmeasured confounder — missing drainage infrastructure, missing impervious surface data). If they fail on different ZCTAs, it's a **representation problem** (fixable by choosing the right embedding). |
| **Hierarchical** | `kappa_hierarchical` (1 - eta_squared) | "Does flood risk prediction generalize across administrative levels?" | Within-county flood prediction might be accurate (similar terrain, same drainage basin), but across-county might fail (different building codes, different flood infrastructure). Low kappa = model overfits to local administrative structure. Crisis response that crosses county lines needs extra scrutiny. |
| **Logical** | `kappa_logical` (Fisher ratio) | "Do adjacent areas have similar flood risk prediction errors?" | If ZCTA A and ZCTA B are neighbors sharing a floodplain, but the model predicts wildly different flood risk accuracy, the adjacency topology is not captured. High kappa = model respects neighborhood relationships. Low kappa = evacuation routing across adjacent ZCTAs is unreliable. |
| **Periodic** | `kappa_periodic` (spectral concentration) | "Does the model capture cyclical flood patterns?" | Seasonal flood risk (spring snowmelt, hurricane season), tidal flooding cycles, recurring El Nino patterns. If residuals show strong periodicity, the model is missing a temporal signal. Relevant for: coastal tidal flooding, seasonal river flooding, annual NFIP claim patterns. |

### 8.2 Remediation Paths by Geometry

Each geometry failure implies a different fix. This is the operational value — a crisis manager doesn't just get "rejected," they get "rejected because X, fix by Y."

| Geometry Failure | What's Wrong | Crisis Response |
|-----------------|-------------|-----------------|
| Low kappa_smooth | Embedding families disagree | Use ensemble or defer to the embedding with best domain-specific performance. Do not rely on any single model. |
| Low kappa_spatial | Errors cluster geographically | Deploy local ground-truth validation in the failing region. Flag specific geographic areas as "low confidence" in crisis dashboard. |
| Low kappa_composite (high agreement) | All models fail the same way | Data gap — add new data source (impervious surface, drainage infrastructure, DEM). No amount of embedding engineering fixes a missing variable. |
| Low kappa_hierarchical | Model doesn't generalize across counties | Use county-specific calibration. Cross-county response plans need manual review. FEMA region boundaries may not match model validity boundaries. |
| Low kappa_logical | Adjacent areas have inconsistent predictions | Add adjacency features (spatial lag, neighbor averaging). Evacuation routing across inconsistent areas needs ground validation. |
| Low kappa_periodic | Model misses cyclical patterns | Add temporal features (season, tidal cycle, antecedent precipitation). Static cross-sectional model may be wrong for time-varying flood risk. |

### 8.3 Gearbox Behavior by Flood Scenario

Different flood scenarios exercise different geometries, which naturally drive different gearbox behavior:

| Scenario | Dominant Geometry | Expected Kappa | Gearbox Response |
|----------|-------------------|----------------|------------------|
| Houston Harvey (riverine + urban flash) | Spatial + Hierarchical | Mixed — spatial clusters of failure near bayous | 2nd-3rd gear: balanced exploration, need multiple embedding views |
| New Orleans Ida (surge + levee) | Logical + Spatial | Low logical (levee creates hard adjacency discontinuity) | 3rd-4th gear: heavy exploration, adjacency structure is broken by levees |
| NYC Ida (urban flash) | Composite + Hierarchical | Low composite (subway infrastructure is unmodeled confounder) | 3rd gear: explore, data gap needs new features |
| Riverside Hilary (desert flash) | Smooth + Periodic | High smooth (desert terrain is uniform), periodic matters for monsoon | 1st-2nd gear: precision, task is representationally stable |
| SW Florida Ian/Helene (coastal surge) | Spatial + Periodic | Low spatial (barrier island vs mainland discontinuity) | 3rd gear: explore, coastal geometry breaks spatial smoothness |

---

## 9. AWS Data Inventory and Geometry Proxy Readiness

### 9.1 What's on AWS Now (as of 2026-05-29)

**Bucket**: `s3://swarm-floodrsct-data/` (account 865679935554, nsc-swarm)

#### Models (Ready)

| Model | S3 Path | Size | Status |
|-------|---------|------|--------|
| Prithvi-EO-2.0 | `model/prithvi_eo2/Prithvi_EO_V2_300M_TL.pt` | 1.33 GB | PASS (smoke test: 331.63M params, 402 tensors) |
| MaxFloodCast | `model/maxfloodcast/` | -- | PENDING (response deadline May 30; surrogate fallback planned) |

#### Raw Datasets (Ready)

| Dataset | S3 Path | Content | Status |
|---------|---------|---------|--------|
| FloodSimBench | `raw/floodsimbench/` | 78 GeoTIFFs, 5 scenarios (HOU, NYC, etc.) | READY |
| Sen1Floods11 | `raw/sen1floods11/` | Sentinel-1 flood segmentation | IN PROGRESS |
| GeoCertDB 2026 | `raw/geocertdb2026/` | 6 parquet files: ZCTA features, SVI, flood zones, NOAA storms, NFIP claims, TWI | READY |

#### Scheduled Pulls (May 29-31)

| Category | Datasets | Target Path |
|----------|----------|-------------|
| Hydrology | USGS NWIS gauges, NOAA MRMS Stage IV, HRRR QPF | `raw/usgs_nwis/`, `raw/noaa_mrms/`, `raw/noaa_hrrr/` |
| Coastal | NOAA tides, NHC SLOSH surge grids | `raw/noaa_tides/`, `raw/noaa_slosh/` |
| Topography | USGS 3DEP DEM, NLCD impervious, NHDPlus catchments | `raw/dem/`, `raw/nlcd/`, `raw/nhdplus/` |
| Infrastructure | USACE levees, MTA subway stations, Houston/NYC 311 | `raw/usace_levees/`, `raw/mta/`, `raw/houston_311/`, `raw/nyc_311/` |
| Hazard | MTBS burn scars, HURDAT2 storm tracks, USGS HWMs | `raw/mtbs/`, `raw/hurdat2/`, `raw/usgs_stn/` |

### 9.2 Data -> Geometry Proxy Mapping

Which AWS data feeds which geometry proxy? This is the critical mapping for experimental validation.

| Geometry Proxy | Required Input | AWS Data Source | Available? |
|----------------|---------------|-----------------|------------|
| **kappa_smooth** | Per-family R2 values (dict {emb: float}) | Computed from any certified task. No raw data needed — uses model predictions. | YES (compute at runtime) |
| **kappa_spatial** | Residuals (ndarray) + adjacency (dict) | Residuals: from model predictions. Adjacency: `zcta_adjacency.parquet` (yrsn-experiments). Coords: `zcta_features_labels.parquet` (geocertdb2026). | YES |
| **kappa_residual_agreement** | Cross-family residuals (dict {family: ndarray}) | Residuals from multiple embedding families on same task. Computed inside `certify_group()`. | YES (compute at runtime) |
| **kappa_hierarchical** | Residuals (ndarray) + group labels (ndarray) | Residuals: from model predictions. Groups: county_fips from `zcta_features_labels.parquet`. State from same. | YES |
| **kappa_logical** | Residuals (ndarray) + adjacency (dict) | Same adjacency as spatial. Queens contiguity from `zcta_adjacency.parquet`. | YES |
| **kappa_periodic** | Residuals (ndarray), ordered by time or cycle index | **NOT DIRECTLY** — CONUS-27 is cross-sectional (no time axis). BUT: NOAA MRMS Stage IV (when fetched) has hourly precipitation. NFIP claims have annual time series. USGS gauges have real-time series. | PARTIAL (requires temporal data — scheduled pulls) |

### 9.3 New Data That Unlocks New Geometry Questions

The scheduled AWS pulls (May 29-31) unlock geometry proxies that weren't possible with the static CONUS-27 dataset:

| New Data | Geometry Unlocked | New Question |
|----------|-------------------|-------------|
| **NOAA MRMS Stage IV** (hourly precip, 6 events) | **Periodic** | "Does the model capture the diurnal precipitation cycle during Harvey?" — 4-day event with clear day/night rainfall patterns |
| **USGS NWIS gauges** (6 events, real-time series) | **Periodic** | "Does the model capture the hydrograph rise/fall cycle?" — gauge timeseries have clear periodic structure |
| **USACE National Levee Database** | **Logical** | "Do levees break the adjacency assumption?" — adjacent ZCTAs on opposite sides of a levee have radically different flood risk. kappa_logical should detect this. |
| **USGS 3DEP DEM** | **Spatial** | "Does elevation gradient explain spatial clustering of errors?" — DEM provides the ground-truth terrain that Moran's I correlates against |
| **NLCD impervious surface** | **Composite** | "Is impervious surface the missing confounder?" — if all embeddings fail on high-impervious ZCTAs, kappa_composite detects the shared blind spot, and impervious data is the fix |
| **NYC sewersheds** | **Hierarchical** | "Does the model generalize across sewershed boundaries?" — NYC's combined sewer system creates a different hierarchy than county/state (sewershed -> borough -> city) |
| **NOAA SLOSH surge grids** | **Spatial** | "Does the model capture the coastal surge gradient?" — surge decays rapidly inland; strong spatial structure that Moran's I should detect |
| **HURDAT2 storm tracks** | **Periodic** | "Does landfall timing create periodic residual structure?" — storm track proximity is a spatial-temporal feature with periodic (seasonal) recurrence |

### 9.4 Data Lock Timeline and Geometry Validation Schedule

| Date | Milestone | Geometry Validation |
|------|-----------|---------------------|
| May 29 (today) | Kappa registry consolidated, 6 proxies registered | T1 unit tests DONE (122 passing) |
| May 30 | MaxFloodCast decision deadline | -- |
| May 31 | All raw data fetched to S3 | -- |
| June 1 | **Data Lock A: Houston MVP** | Run all 6 proxies on Houston scenario. T6 redundancy check. |
| June 2 | **Data Lock B: All 5 scenarios** | Run all 6 proxies on all scenarios. Cross-scenario comparison. |
| June 3-4 | Integration testing | T2 signal path, T3 projection, T5 gate flip |
| June 5 | **SIGSPATIAL abstract** | Geometry proxy results in abstract |
| June 12 | **Full paper** | Complete experimental section |

---

## 10. Floodcaster MOE Stack Integration

### 10.1 The MOE Stack (Series 035)

The Mixture of Experts flood stack is being built on AWS:

| Expert | Model | Role | S3 Location |
|--------|-------|------|-------------|
| **Foundation** | Prithvi-EO-2.0 (300M params) | Remote sensing feature extraction | `model/prithvi_eo2/` (READY) |
| **Hydraulic** | MaxFloodCast (or LSTM surrogate) | Physics-informed flood depth prediction | `model/maxfloodcast/` (PENDING) |
| **Benchmark** | FloodSimBench | Validation targets for simulated floods | `raw/floodsimbench/` (READY) |
| **Segmentation** | Sen1Floods11 fine-tuned | SAR-based flood extent mapping | `raw/sen1floods11/` (IN PROGRESS) |

### 10.2 How MOE Experts Map to Geometry Proxies

Each expert in the MOE stack produces residuals. The geometry proxies measure different properties of those residuals:

```
MOE Stack
├── Prithvi (remote sensing features)
│     └── kappa_smooth: Do Prithvi features give consistent R2 across tasks?
│     └── kappa_composite: Do Prithvi and MaxFloodCast fail on same locations?
│
├── MaxFloodCast (physics-informed)
│     └── kappa_spatial: Does MaxFloodCast error cluster geographically?
│     └── kappa_periodic: Does MaxFloodCast miss tidal/diurnal cycles?
│
├── FloodSimBench (simulation ground truth)
│     └── kappa_hierarchical: Do simulated floods generalize across admin boundaries?
│     └── kappa_logical: Does simulation respect adjacency topology?
│
└── Certificate Pipeline
      ├── Per-expert residuals → geometry-specific kappa per expert
      ├── Cross-expert residual agreement → kappa_composite
      └── Final kappa → gearbox → gear determines exploration budget
```

### 10.3 The Certification Loop for Crisis

```
1. Crisis event detected (e.g., Hurricane approaching Houston)
2. MOE stack produces predictions for affected ZCTAs
3. For each ZCTA prediction:
   a. Compute geometry-specific kappas:
      - kappa_spatial: Does error cluster near bayous? (Moran's I on residuals + adjacency)
      - kappa_hierarchical: Does prediction hold across counties? (1 - eta_squared)
      - kappa_composite: Do all experts fail the same ZCTAs? (Spearman cross-family)
   b. Select worst-case kappa: kappa_compat = min(kappa_spatial, kappa_hierarchical, ...)
   c. Gate: SequentialGatekeeper.evaluate(kappa_compat, sigma)
      - EXECUTE: prediction is trustworthy → show to crisis manager
      - RE_ENCODE: gearbox shifts → try different embedding/expert combination
      - REJECT: prediction is unreliable → flag to crisis manager with reason
4. Crisis dashboard shows:
   - Per-ZCTA flood risk predictions WITH confidence badges
   - Geographic map of kappa_spatial (where the model is blind)
   - Hierarchy map of kappa_hierarchical (where county boundaries break the model)
   - Explicit "low confidence" zones where geometry kappa failed
```
