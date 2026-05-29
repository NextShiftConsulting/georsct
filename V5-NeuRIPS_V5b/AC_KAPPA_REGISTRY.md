# Kappa Registry Audit: One Registry to Rule Them All

**Date**: 2026-05-29
**Scope**: Every `compute_kappa*` function in yrsn, registered or not
**Goal**: Decide for each: KEEP (registered, tested, documented) or ELIMINATE (remove or consolidate)

---

## THE PROBLEM

20 kappa computations across 4 modules. 13 registered, 7 unregistered. Some are duplicates with different call signatures. Some are legacy wrappers around the same formula. Some are tensor variants of scalar functions. This is how technical debt becomes legal liability -- in a patent context, every kappa variant is a potential overclaim or underclaim.

**Rule**: If it computes kappa, it lives in `yrsn/core/kappa/` and is registered. No exceptions. No shadow implementations in routing/, decomposition/, or compression/.

---

## FULL INVENTORY (20 computations)

### TIER 1: CANONICAL (patent-claimed, theory-backed, gate-load-bearing)

| # | Name | Location | Formula | Registered? | Verdict |
|---|---|---|---|---|---|
| 1 | `kappa_compat` | `kappa/compat/compute.py` | R*(1-N) | YES | **KEEP** -- primary gate metric, Claim 2 |
| 2 | `kappa_compat_cosine` | `kappa/gate/compute.py` | cosine(event, solver) -> [0,1] | YES (overwrites kappa_compat) | **KEEP but FIX** -- overwrites #1 in registry. Needs own key `kappa_compat_cosine` |
| 3 | `kappa_modal_min` | `kappa/gate/compute.py` | min(H, L, interface) | YES (via gate) | **KEEP** -- multimodal weakest-link, Claim 8 |
| 4 | `kappa_req` | `kappa/req/compute.py` | kappa_base + lambda * sigmoid(sigma) | YES | **KEEP** -- Gate 3 dynamic threshold, Claim 8 |
| 5 | `kappa_difficulty` | `kappa/difficulty_theory/compute.py` | clip(D*/D_actual, 0, 1) | YES | **KEEP** -- theory kappa, Definition 4.1 |

### TIER 2: OPERATIONAL (pragmatic proxies, production-used)

| # | Name | Location | Formula | Registered? | Verdict |
|---|---|---|---|---|---|
| 6 | `kappa_composite` | `kappa/composite/compute.py` | min(primitives) * (1 - penalty*(n-1)) | YES | **KEEP** -- multi-primitive bottleneck |
| 7 | `kappa_empirical` | `kappa/empirical/compute.py` | R*K_R + S*K_S + N*K_N | YES | **KEEP** -- domain-calibrated expected accuracy |

### TIER 3: DIAGNOSTIC (never gates, post-hoc explanation only)

| # | Name | Location | Formula | Registered? | Verdict |
|---|---|---|---|---|---|
| 8 | `kappa_simplex` | `kappa/diagnostic_simplex/` (re-exports from `decomposition/kappa_simplex.py`) | (A,E,T) decomposition, min(A,E,T) | YES | **KEEP** -- bottleneck diagnosis |

### TIER 4: GEOMETRY-SPECIFIC (experimental, new)

| # | Name | Location | Formula | Registered? | Tests | Verdict |
|---|---|---|---|---|---|---|
| 9 | `kappa_residual_agreement` | `kappa/residual_agreement/compute.py` | median pairwise Spearman rho of cross-family residuals | YES | 19 pass | **KEEP** -- composite geometry |
| 10 | `kappa_smooth` | `kappa/smooth/compute.py` | 1 - CV(R2) leave-one-out | YES | 23 pass | **KEEP** -- smooth geometry |
| 11 | `kappa_spatial` | `kappa/spatial/compute.py` | (Moran's I - E[I] + 1) / 2 | YES | 19 pass | **KEEP** -- spatial geometry |
| 12 | `kappa_hierarchical` | `kappa/hierarchical/compute.py` | 1 - eta_squared | YES | 21 pass | **KEEP** -- hierarchical geometry |
| 13 | `kappa_logical` | `kappa/logical/compute.py` | 1 - d_adj/d_non (Fisher ratio) | YES | 24 pass | **KEEP** -- logical geometry |
| 14 | `kappa_periodic` | `kappa/periodic/compute.py` | spectral concentration ratio | YES | 16 pass | **KEEP** -- periodic geometry |

### TIER 5: DEPRECATED / LEGACY

| # | Name | Location | Formula | Registered? | Verdict |
|---|---|---|---|---|---|
| 15 | `kappa_proxy` | `kappa/proxy_simplex/compute.py` | R/(R+N) [= alpha] | YES | **ELIMINATE** -- violates Claim 2 dual-path. Kept only for s018a-g reproduction. Should be removed from registry, kept as private util if needed for historical runs. |

### TIER 6: UNREGISTERED DUPLICATES (shadow implementations)

| # | Name | Location | What It Is | Verdict |
|---|---|---|---|---|
| 16 | `compute_kappa` | `decomposition/rsct_integration.py:105` | `min(1, d_star/d_actual)` -- legacy scalar, wrapped by `kappa_difficulty` | **ELIMINATE** -- make callers use `kappa_difficulty` directly |
| 17 | `compute_kappa_from_result` | `decomposition/rsct_integration.py:124` | Extracts kappa from `DStarEstimate` -- convenience wrapper | **ELIMINATE** -- inline at call sites or move to `kappa_difficulty` module |
| 18 | `compute_kappa_compat` | `routing/router.py:106` | Cosine similarity -> [0,1] -- DUPLICATE of `kappa_compat_cosine` (#2) | **ELIMINATE** -- make callers import from `kappa/gate/` |
| 19 | `compute_kappa_req` | `routing/router.py:151` | Sigmoid threshold -- DUPLICATE of `kappa_req` (#4), delegates to `oobleck_threshold()` | **ELIMINATE** -- make callers import from `kappa/req/` |
| 20 | `compute_kappa_req_tensor` | `compression/multi_geometry_loss.py:380` | PyTorch tensor version of kappa_req for training loss | **CONSOLIDATE** -- move to `kappa/req/compute.py` as `compute_kappa_req_tensor()`, register as `kappa_req_tensor` |

### TIER 7: UNREGISTERED UNIQUE

| # | Name | Location | Formula | Verdict |
|---|---|---|---|---|
| 21 | `compute_kappa_priority` | `decomposition/rsct_integration.py:185` | `(abs(old_kappa - new_kappa) + eps)^alpha` -- replay priority from kappa degradation | **KEEP but REGISTER** -- genuinely distinct, useful for experience replay |

---

## REGISTRY FIX: kappa_compat OVERWRITE BUG

**Current problem** in `registry.py:_populate_registry()`:

```python
# Line 100: registers kappa_compat from compat/ (R*(1-N))
_register("kappa_compat", KAPPA_COMPAT_METADATA, compute_kappa_compat)

# Line 112: OVERWRITES with cosine version from gate/
_register("kappa_compat", KAPPA_GATE_METADATA, compute_kappa_compat_cosine)
```

The second registration silently overwrites the first. `get_kappa("kappa_compat")` returns the cosine version, NOT R*(1-N). This is a bug.

**Fix**: Register the gate cosine version under its own name:
```python
_register("kappa_compat_cosine", KAPPA_GATE_METADATA, compute_kappa_compat_cosine)
_register("kappa_modal_min", KAPPA_MODAL_MIN_METADATA, compute_kappa_modal_min)
```

---

## ACTION PLAN

### Phase 1: Eliminate duplicates (routing/, decomposition/)

| File | Function | Action |
|---|---|---|
| `routing/router.py:106` | `compute_kappa_compat` | Deprecate. Callers import from `kappa.gate` |
| `routing/router.py:151` | `compute_kappa_req` | Deprecate. Callers import from `kappa.req` |
| `decomposition/rsct_integration.py:105` | `compute_kappa` | Deprecate. Callers use `compute_kappa_difficulty` |
| `decomposition/rsct_integration.py:124` | `compute_kappa_from_result` | Deprecate. Inline or move to difficulty_theory/ |

### Phase 2: Fix registry overwrite

| Action | Detail |
|---|---|
| Rename gate registration | `kappa_compat` -> `kappa_compat_cosine` |
| Verify downstream | Grep all `get_kappa("kappa_compat")` calls -- do they expect R*(1-N) or cosine? |

### Phase 3: Register orphans

| Function | New Registry Key | Status |
|---|---|---|
| `compute_kappa_priority` | `kappa_priority` | OPERATIONAL |
| `compute_kappa_req_tensor` | `kappa_req_tensor` | OPERATIONAL |

### Phase 4: Deprecate kappa_proxy

| Action | Detail |
|---|---|
| Remove from registry | Keep module for s018 reproduction but unregister |
| Add deprecation warning | `warnings.warn("kappa_proxy is R/(R+N) = alpha. Use kappa_compat.")` |

---

## POST-CLEANUP REGISTRY (target state: 16 entries)

```
CANONICAL (5):
  kappa_compat           R*(1-N)                    Gate 3 primary
  kappa_compat_cosine    cosine(event, solver)       Routing-layer compatibility
  kappa_modal_min        min(H, L, interface)        Multimodal weakest-link
  kappa_req              sigmoid oobleck threshold   Gate 3 dynamic threshold
  kappa_difficulty       clip(D*/D, 0, 1)            Theory kappa

OPERATIONAL (4):
  kappa_composite        min(prims) * penalty        Multi-primitive bottleneck
  kappa_empirical        R*K_R + S*K_S + N*K_N       Domain-calibrated accuracy
  kappa_priority         degradation-based replay     Experience replay priority
  kappa_req_tensor       tensor sigmoid threshold     Training loss variant

DIAGNOSTIC (1):
  kappa_simplex          (A, E, T) decomposition     Post-hoc bottleneck

EXPERIMENTAL / GEOMETRY-SPECIFIC (6):
  kappa_residual_agreement   Spearman rho cross-family    Composite geometry
  kappa_smooth               1 - CV(R2) LOO              Smooth geometry
  kappa_spatial              Moran's I normalized         Spatial geometry
  kappa_hierarchical         1 - eta_squared              Hierarchical geometry
  kappa_logical              Fisher ratio on adjacency    Logical geometry
  kappa_periodic             spectral concentration       Periodic geometry
```

16 registered, 0 unregistered, 0 duplicates, 0 shadow implementations.

---

## ACCEPTANCE CRITERIA FOR THIS AUDIT

| # | Criterion | Status |
|---|-----------|--------|
| A1 | Every `compute_kappa*` in yrsn/core/ is either registered or explicitly deprecated with a redirect | [ ] |
| A2 | No two registry entries compute the same formula under different names | [ ] |
| A3 | `kappa_compat` overwrite bug is fixed -- R*(1-N) and cosine have separate keys | [ ] |
| A4 | `kappa_proxy` removed from registry (module kept for s018 reproduction) | [ ] |
| A5 | `kappa_priority` and `kappa_req_tensor` registered | [ ] |
| A6 | All 16 target entries have: metadata, compute function, unit tests | [ ] |
| A7 | `list_kappas()` returns exactly 16 entries | [ ] |
| A8 | No `compute_kappa*` function exists outside `yrsn/core/kappa/` (all consolidated) | [ ] |
