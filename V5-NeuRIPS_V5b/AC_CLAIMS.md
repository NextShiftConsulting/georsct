# Acceptance Criteria: Six-Geometries Figure Claims

**Figure**: `georsct_six_geometries_branched_paper.svg`
**Caption**: "One pipeline. Six kappa-proxies. Three instantiated, three runnable."
**Status**: NOT INCLUDED in LaTeX as of 2026-05-29
**Standard**: Every word in a published figure must be defensible under peer review and patent prosecution. "INSTANTIATED" is a factual claim, not a design aspiration.

---

## CLAIM 1: "One pipeline"

**Figure says**: Steps 1-4 and 6 are identical across geometries.

### What actually exists

The pipeline lives in `theory_certifier.py` (225 lines). It is a single function `certify_group()` that runs:

| Step | Figure Label | Code Location | What It Does |
|------|-------------|---------------|--------------|
| 1 | Slice select | `run_s019d.py` (caller) | GroupKFold by county_fips, 5 folds |
| 2 | Embed x Solver | `theory_certifier.py:93-111` | Fits HistGBDT on each Z_train, predicts Z_test |
| 3 | OOF residuals; N_ceiling | `theory_certifier.py:113-125` | Shared tercile boundaries from pooled abs(residuals) |
| 4 | R / S_sup / N simplex | `theory_certifier.py:151-189` | MLP tercile classifier -> softmax probs -> median(R,S,N) |
| 5 | alpha / kappa / sigma | `theory_certifier.py:127-149` (theory) + `:151-189` (proxy) | Theory: RegressionKappaEvaluator D*/D. Proxy: R*(1-N) |
| 6 | Gate sequence -> verdict | Imported from `yrsn_controlplane.SequentialGatekeeper` | 5-gate pipeline with Oobleck sigmoidal threshold |

**The pipeline is geometry-agnostic.** There is no `geometry_type` parameter anywhere. Every embedding goes through the identical path. This is both the strength (the "one pipeline" claim is trivially true) and the weakness (there is no branching at step 5, so "six implementations" is false).

### Acceptance criteria

| # | Criterion | Evidence | Status |
|---|-----------|----------|--------|
| 1.1 | `certify_group()` accepts a `geometry_type` parameter that selects which step-5 computation to run | theory_certifier.py has no such parameter | FAIL |
| 1.2 | Steps 1-4 and 6 are factored into callable units (not one monolith) so step 5 can be swapped | Steps 2-5 are interleaved in a single FOR loop (lines 93-189). Step 5 (theory kappa) is inside the same loop as step 4 (MLP classifier). No clean separation exists. | FAIL |
| 1.3 | A `CertificationPipeline` class or protocol exists with a pluggable `step_5` method | No such class exists. The pipeline is a procedural function. | FAIL |
| 1.4 | Steps 1-4 and 6 have been verified to produce identical outputs when step 5 is replaced with a different computation | Never tested because step 5 has never been replaced | FAIL |

**Reality**: The "one pipeline" claim is aspirationally true -- the architecture COULD support pluggable step 5 -- but the code is a monolith with no dispatch mechanism. Making this claim true requires refactoring `certify_group()` into a pipeline with an injectable step-5 strategy.

**Engineering cost to fix**: Medium. Factor the FOR loop at lines 93-189 into:
1. `_fit_and_predict()` (step 2)
2. `_compute_boundaries()` (step 3)
3. `_compute_simplex()` (step 4)
4. `_compute_kappa(strategy: KappaStrategy)` (step 5, pluggable)
5. Gate evaluation stays external (already factored via SequentialGatekeeper import)

The hard part isn't the refactor -- it's that steps 4 and 5 share intermediate state (per-sample probs feed both simplex and proxy kappa). Separating them cleanly requires defining what data crosses the step 4->5 boundary.

---

## CLAIM 2: "Six kappa-proxies"

**Figure says**: Six distinct kappa-proxy computations, one per geometry.

### What actually exists

The yrsn kappa registry (`yrsn/core/kappa/registry.py`) has **8 registered families** -- but NONE of them are the six geometry-specific proxies from the figure:

| Registered Family | Formula | Geometry-Specific? |
|---|---|---|
| `kappa_compat` | R*(1-N) | No -- universal |
| `kappa_proxy` | R/(R+N) [= alpha, DEPRECATED] | No -- universal |
| `kappa_empirical` | R*K_R + S*K_S + N*K_N | No -- domain-table-weighted |
| `kappa_modal_min` | min(H, L, interface) | No -- modality, not geometry |
| `kappa_req` | sigmoid oobleck threshold | No -- threshold, not measurement |
| `kappa_difficulty` | clip(D*/D_actual, 0, 1) | No -- universal leave-one-out |
| `kappa_simplex` | (A, E, T) diagnostic | No -- post-hoc decomposition |
| `kappa_composite` | min(primitives) * penalty | No -- multi-primitive bottleneck |

The six geometry-specific proxies from the figure (cross-family R2 convergence, semivariogram range, cross-family residual agreement, 2pi-seam continuity, within-vs-across R2 gap, Fisher discriminant on adjacency) have **zero code, zero registry entries, zero tests**.

### How the registry extension mechanism works

Adding a new kappa family requires:
1. Create directory `yrsn/core/kappa/{family_name}/` with `metadata.py` + `compute.py`
2. Implement `KappaMetadata` (name, formula, status, patent_ref, input_contract, assumptions)
3. Implement `compute_kappa_{name}(**inputs) -> KappaResult` returning scalar value + audit fields
4. Register in `_populate_registry()` in `registry.py`

The interface is clean: `KappaResult` has `value: float`, `status: KappaStatus`, `assumptions: List[str]`. The gate pipeline consumes kappa via `CPGatekeeperInput.kappa_compat` (float). So any new kappa that produces a float in [0,1] is gate-compatible.

**But**: The gate pipeline reads `kappa_compat` as a single field. There is no mechanism to say "this task should use semivariogram kappa instead of R*(1-N) kappa." Adding geometry-type dispatch requires changes to:
- `CPGatekeeperInput` (add `kappa_geometry_type` field or multiple kappa slots)
- `SequentialGatekeeper.evaluate()` (select which kappa to gate on)
- Or: compute geometry-specific kappa upstream and pass it as `kappa_compat` (simpler, but loses provenance)

### Acceptance criteria

| # | Criterion | Evidence | Status |
|---|-----------|----------|--------|
| 2.1 | Six `compute_kappa_{geometry}()` functions exist returning `KappaResult` | Zero exist | FAIL |
| 2.2 | Each function has a distinct `input_contract` requiring geometry-specific data | No functions to inspect | FAIL |
| 2.3 | Each function is registered via `_register()` in `_populate_registry()` | Zero new registrations | FAIL |
| 2.4 | A dispatch mechanism maps `geometry_type -> kappa_function` at step 5 | No dispatch exists | FAIL |
| 2.5 | Gate pipeline can consume geometry-specific kappa without modification | Partially true: any float in [0,1] can be passed as `kappa_compat`. Full provenance tracking (which proxy was used) would require `CPGatekeeperInput` changes. | PARTIAL |

**Engineering cost to fix**: The registry pattern is solid. Adding six new families is copy-paste-compute. The real cost is in the COMPUTE -- each proxy requires different math, different inputs, and different validation (see AC_SIX_KAPPAS.md).

---

## CLAIM 3: "Three instantiated" (Smooth, Spatial, Composite)

### What "instantiated" must mean for this claim to be true

In yrsn's kappa registry, each family has a `KappaStatus`:
- `CANONICAL` -- patent-claimed, theory-backed
- `OPERATIONAL` -- pragmatic proxy when canonical unavailable
- `DIAGNOSTIC` -- post-hoc, never gates
- `EXPERIMENTAL` -- not validated

The figure uses "INSTANTIATED" which maps to none of these. If the figure means "code exists and has been run on real data," then:

### What data exists to instantiate these proxies

From the S019D experiment artifacts:

| Data Needed | On Disk? | Location | Format |
|---|---|---|---|
| Per-embedding R2 (for Smooth: cross-family convergence) | YES | `data/s019d/seed_*/s019d_results.json` | 810 rows: r2 per (task, embedding, fold) |
| Per-sample OOF residuals (for Composite: residual agreement) | NO | Computed in memory, NOT serialized | `_kappa_per_theory` stripped from JSON export |
| Spatial coordinates (for Spatial: semivariogram) | YES | `data/geocert/v24/zcta_features_labels.parquet` | lat/lon for 31,789 ZCTAs |
| Adjacency matrix (for Spatial: semivariogram lag bins) | YES | `yrsn-experiments/.../zcta_adjacency.parquet` | Queen's contiguity edge-list |
| Per-sample solver predictions (for residual vectors) | NO | Computed in memory, NOT serialized | `solver_preds_test[emb]` not in JSON output |

**Critical gap**: The two most important raw materials -- per-sample residuals and per-sample predictions -- are computed during `certify_group()` but discarded after aggregation. Building geometry-specific proxies requires either:
- (a) Re-running S019D with a modified `certify_group()` that exports per-sample arrays, or
- (b) Computing the proxies INSIDE `certify_group()` where per-sample data is live

Option (b) is the right architecture -- geometry-specific kappa is part of step 5, so it should be computed where the data lives, not reconstructed after the fact.

### Acceptance criteria

| # | Criterion | Evidence | Status |
|---|-----------|----------|--------|
| 3.1 | **Smooth proxy runs**: `compute_kappa_smooth(per_family_r2: Dict[str, float]) -> KappaResult` exists and has been called on S019D data | No function exists | FAIL |
| 3.2 | **Spatial proxy runs**: `compute_kappa_spatial(residuals: ndarray, coords: ndarray) -> KappaResult` exists and has been called with ZCTA coordinates | No function exists. No semivariogram code anywhere in yrsn or rsct-geocert. | FAIL |
| 3.3 | **Composite proxy runs**: `compute_kappa_composite_geom(residuals: Dict[str, ndarray]) -> KappaResult` exists and has been called on multi-family residual vectors | No function exists. Per-sample residuals not persisted to disk. | FAIL |
| 3.4 | Each proxy's output has been compared to theory kappa (D*/D) and proxy kappa (R*(1-N)) on the same tasks, showing the geometry-specific proxy adds information | No comparison exists | FAIL |
| 3.5 | Paper text defines "INSTANTIATED" and cites evidence for each | "INSTANTIATED" does not appear in paper text. The figure is not `\includegraphics`'d. | FAIL |

**Engineering cost to fix**:
- **Smooth**: LOW. Per-family R2 is already in `s019d_results.json`. Computing cross-family convergence is ~20 lines of numpy (variance of R2 across 6 embeddings per task). But: the result is 27 scalars (one per task), not per-sample. Is task-level kappa meaningful when the gate operates per-embedding?
- **Spatial**: HIGH. Requires a semivariogram fitting library (none in deps), spatial coordinate loading into the pipeline, and a normalization scheme from range->[0,1]. The mathematical choices (model family, anisotropy, sill handling) are underspecified.
- **Composite**: MEDIUM. Per-family residual vectors are in memory during `certify_group()`. Computing pairwise agreement is ~30 lines. But: "agreement" is undefined (Pearson? cosine? rank? -- each answers a different question).

---

## CLAIM 4: "Three supported" (Periodic, Hierarchical, Logical)

### What "supported" should mean

The figure's bottom caption says "three runnable" (not "three supported"), but the badges say "SUPPORTED." This internal inconsistency needs resolution: "runnable" implies code exists; "supported" is weaker and could mean "the architecture permits it."

### Relevance to CONUS-27

| Proxy | Applicable to CONUS-27? | Why / Why Not |
|---|---|---|
| Periodic (2pi-seam continuity) | NO | CONUS-27 has no periodic/cyclic targets. Lat/lon wraps at 180 but CONUS doesn't cross the antimeridian. No temporal periodicity in cross-sectional ZCTA data. |
| Hierarchical (within-vs-across R2 gap) | YES | ZCTA -> county -> state is a 3-level hierarchy. GroupKFold already uses county. R2 gap between within-county and across-county predictions is computable. |
| Logical (Fisher on adjacency) | YES | Queen's adjacency exists. Fisher discriminant of features between adjacent vs non-adjacent ZCTAs is computable. But: this is closely related to Moran's I, which is better established. |

**The Periodic proxy is a problem**: It's labeled SUPPORTED for a benchmark that has no periodic structure. If the paper is a geospatial paper on CONUS-27, a periodic kappa-proxy is architectural furniture -- it exists to make the taxonomy look complete, not because the data needs it.

### Acceptance criteria

| # | Criterion | Evidence | Status |
|---|-----------|----------|--------|
| 4.1 | Paper defines "SUPPORTED" with a precise meaning distinct from "INSTANTIATED" | Not in paper text | FAIL |
| 4.2 | Each SUPPORTED proxy has a mathematical definition (equation in paper or appendix) | No equations for any of the three | FAIL |
| 4.3 | The pipeline's step-5 interface is documented as extensible (showing how a new proxy plugs in) | No interface documentation. The FOR loop is monolithic. | FAIL |
| 4.4 | For proxies applicable to CONUS-27 (Hierarchical, Logical): a rationale exists for why they're SUPPORTED and not INSTANTIATED | No rationale documented | FAIL |
| 4.5 | For proxies NOT applicable to CONUS-27 (Periodic): the figure or caption notes that this proxy targets a different data regime | No such note exists | FAIL |

---

## CLAIM 5: "Step 5 has six implementations"

### How step 5 actually works

Step 5 in `theory_certifier.py` is two interleaved computations in the same FOR loop:

**Theory kappa** (lines 127-149):
```
RegressionKappaEvaluator.evidence(emb_name)
  -> _ensure_dstar(): D* = min over reference embeddings of per-sample squared residuals
  -> compute_kappa_difficulty_batch(d_star, d_actual): clip(D*/D, 0, 1) per sample
  -> median aggregation
```

**Proxy kappa** (lines 151-189):
```
MLP tercile classifier on Z_train -> labels_train (R/S/N buckets)
  -> clf.predict_proba(Z_test) -> probs_test shape (n_test, 3)
  -> aggregate_scores_from_probs: R_med, S_med, N_med, proxy_kappa = median(R_i * (1 - N_i))
```

Both are geometry-agnostic. Neither branches on geometry type. The phrase "step 5 has six implementations" requires six distinct code paths here. Currently there is **one**.

### What "six implementations" would structurally require

```python
# Current (monolithic):
for emb_name in emb_names:
    theory_kappa = evaluator.evidence(emb_name)
    proxy_kappa = aggregate_scores_from_probs(probs)

# Required (dispatched):
for emb_name in emb_names:
    theory_kappa = evaluator.evidence(emb_name)
    proxy_kappa = aggregate_scores_from_probs(probs)
    geometry_kappa = geometry_dispatch[geometry_type](
        residuals=residuals,
        coords=coords,        # only for Spatial
        adjacency=adj,        # only for Logical
        per_family_r2=r2s,    # only for Smooth
        ...
    )
```

The dispatch needs:
1. A `geometry_type` parameter passed to `certify_group()`
2. A registry mapping type -> function
3. Different input signatures per function (Spatial needs coords; Logical needs adjacency; Smooth needs cross-family R2)
4. A decision: does geometry kappa REPLACE theory/proxy kappa, or is it a THIRD kappa that sits alongside them?

That last question is architecturally load-bearing and the figure doesn't answer it.

### Acceptance criteria

| # | Criterion | Evidence | Status |
|---|-----------|----------|--------|
| 5.1 | `certify_group()` has a `geometry_type` parameter | No such parameter | FAIL |
| 5.2 | A dispatch table or strategy pattern maps geometry_type to a step-5 function | No dispatch | FAIL |
| 5.3 | The step-5 output schema is defined (does geometry kappa replace, augment, or override theory/proxy kappa?) | Not defined | FAIL |
| 5.4 | At least 3 of the 6 code paths have been executed on real data | Zero executed | FAIL |

---

## CLAIM 6: Figure is publication-ready

### Current LaTeX state

Searched all `.tex` files in `V5-NeuRIPS_V5b/georsct_v5b/`:
- `\includegraphics{...six_geometries...}` -- **not found**
- `\ref{fig:six_geometries}` -- **not found**
- The word "instantiated" appears 3 times in the paper, all referring to the 12-mode taxonomy ("instantiates D.1"), NOT to kappa proxies
- The word "six" + "geometr" appears **zero times** in any `.tex` file

The figure is an orphan artifact in `figures/`. The paper's actual contribution structure (N_ceiling, scalar insufficiency with 12-mode taxonomy, substrate-conditioned compatibility) makes no reference to six geometries or six kappa-proxies.

### Tension with the paper's actual claims

The paper's Section 4 (Contribution 2) says: *"the twelve-mode geo-substrate failure taxonomy across four families (GEO-A through GEO-D)"* and *"Methods-level injection validation on the three GEO-D modes recovers top-1 across eight seeds; the remaining nine modes ship with observable certificate-field signatures and typed gate dispositions."*

This is a 12-mode taxonomy with 3 validated + 9 signature-defined. The six-geometries figure is a DIFFERENT taxonomy (6 geometry types with 3 instantiated + 3 supported). These two taxonomies have no documented relationship. If both ship, a reviewer will ask: "How do the 12 failure modes map to the 6 geometry types?" and there's no answer.

### Acceptance criteria

| # | Criterion | Evidence | Status |
|---|-----------|----------|--------|
| 6.1 | Figure is `\includegraphics`'d in a `.tex` file | grep: not found | FAIL |
| 6.2 | Figure has a `\caption{}` with `\label{fig:...}` | Not in LaTeX | FAIL |
| 6.3 | At least one `\ref{fig:...}` in body text | Not in LaTeX | FAIL |
| 6.4 | The six-geometry taxonomy is related to the 12-mode taxonomy (mapping documented) | No mapping exists | FAIL |
| 6.5 | NeurIPS checklist (Appendix J) claims are consistent with figure | Figure not in paper, so no conflict yet -- but WILL conflict if added without backing | N/A |

---

## CURRENT ASSESSMENT (2026-05-29)

| Claim | Verdict | Core Issue |
|-------|---------|------------|
| 1. One pipeline | STRUCTURALLY TRUE but not ARCHITECTURALLY WIRED | Pipeline is monolithic; step 5 isn't pluggable |
| 2. Six kappa-proxies | FAIL | Zero of six exist in code or registry |
| 3. Three instantiated | FAIL | "Instantiated" has no backing -- zero implementations, zero results |
| 4. Three supported | FAIL | "Supported" undefined, Periodic irrelevant to CONUS-27 |
| 5. Six implementations | FAIL | One implementation (universal), not six |
| 6. Publication-ready | FAIL | Not in LaTeX, conflicts with 12-mode taxonomy |

### What's actually close vs. what's far

| Proxy | Data Available | Math Defined | Implementation Effort | Verdict |
|---|---|---|---|---|
| Smooth (R2 convergence) | YES (s019d_results.json has per-family R2) | NO (convergence vs variance vs spread?) | LOW (~20 lines) but semantics unresolved | CLOSEST |
| Composite (residual agreement) | PARTIAL (in-memory during certify_group, not persisted) | NO (correlation vs cosine vs rank?) | MEDIUM (~30 lines + pipeline refactor) | REACHABLE |
| Hierarchical (R2 gap) | YES (county/state hierarchy in crosswalk) | NO (gap = difference? ratio? effect size?) | MEDIUM (~40 lines + hierarchy loader) | REACHABLE |
| Logical (Fisher on adjacency) | YES (queen's adjacency exists in yrsn-experiments) | NO (Fisher vs Moran's I? justify choice) | MEDIUM (~50 lines + adjacency loader) | REACHABLE |
| Spatial (semivariogram) | PARTIAL (coords yes, residuals not persisted) | NO (which model? normalization? anisotropy?) | HIGH (new dependency + fitting + normalization) | FAR |
| Periodic (2pi-seam) | NO DATA (CONUS-27 has no periodic targets) | NO | HIGH (need periodic dataset) | IRRELEVANT |

---

## REMEDIATION PATHS

### Path A: Build the three INSTANTIATED proxies

**Scope**: Implement Smooth, Spatial, Composite. Run on CONUS-27. Compare to universal kappa.
**Cost**: 2-4 days. Spatial is the bottleneck (semivariogram fitting from scratch or new dependency).
**Risk**: The proxies might not add information. If cross-family R2 variance correlates perfectly with R*(1-N), the Smooth proxy is redundant and the figure's premise collapses.
**Pre-check before building**: Compute Spearman correlation between per-task R2 variance and per-task mean proxy_kappa from existing s019d_results.json. If rho > 0.9, the Smooth proxy is a restatement of what the universal proxy already captures. This takes 10 minutes and should happen BEFORE any implementation.

### Path B: Relabel to future work

**Scope**: Change INSTANTIATED -> DEFINED, SUPPORTED -> PLANNED. Change caption to "Architectural design for v2 geometry-specific certification."
**Cost**: 10 minutes (SVG edit).
**Risk**: None. Accurate labeling.

### Path C: Kill the figure

**Scope**: Don't ship it. The paper stands on N_ceiling + 12-mode taxonomy + substrate conditioning.
**Cost**: Zero.
**Risk**: Loses a visually compelling architecture diagram. But the diagram promises something the paper doesn't deliver.

### Path D: Scope it as a discussion figure

**Scope**: Add to Section 6 (Conclusion) as "future geometry-specific extensions." Caption: "The v1 pipeline uses universal kappa; this diagram shows the planned v2 branching architecture."
**Cost**: 30 minutes (add to LaTeX, write caption, add 2 sentences to conclusion).
**Risk**: Low. Honest about what exists. Gives reviewers something to cite as future work.
