# DOE-C4: Morph Dispatch — ADR-012 Gate-Fail-Morph-Regate Loop

**Experiment:** s035-model-ladder / DOE-C4
**Role:** Implement and test the dual graph morphing loop for flood certification
**Status:** DESIGNED
**Depends on:** DOE-C3 (needs discriminative Gate 3B at tract resolution)
**Blocks:** Floodcaster engine integration

---

## Motivation

ADR-012 defines the adjoint mechanism: when a gate fails, the system can
morph the representation graph (G_R) or solver graph (G_S), re-measure,
and re-gate. In yrsn, `DualGraphSystem.run_dual_graph()` implements this
as a max-2-attempt loop with `_adaptive_morph()`.

DOE-C1 through C3 establish static certificates. DOE-C4 tests whether
morphing can rescue a failing construct certificate. The key question:
if FEMA fails Gate 3B at tract resolution (kappa_reconstruct too low),
can a topology morph (e.g., adding hydrology edges) recover spatial
structure without inflating forward score?

---

## Hypothesis

**H-C4:** For constructs that fail Gate 3B at tract resolution, a single
topology morph (adding hydrology-based edges to the adjacency graph)
recovers kappa_reconstruct above the gate threshold without changing
forward score by more than 0.05.

**Null:** Topology morphs either do not improve kappa_reconstruct, or
they inflate forward score (indicating the morph introduced leakage).

**Anti-hypothesis:** If the morph improves BOTH kappa_reconstruct AND
forward score, the added edges are carrying predictive information
that should have been in the features, not the topology.

---

## Design Matrix

| Factor | Levels | Type |
|--------|--------|------|
| Scenario | houston, new_orleans | Fixed (same as DOE-C3) |
| Construct | Those failing Gate 3B in DOE-C3 | Data-dependent |
| Morph type | none (control), hydrology_edges, knn_spatial | Treatment (3 arms) |

**Morph definitions:**
- **none:** Original queen contiguity adjacency (control).
- **hydrology_edges:** Add edges between tracts sharing a HUC-12 watershed
  boundary. Physical water connectivity independent of administrative
  boundaries.
- **knn_spatial:** Replace queen contiguity with k=8 nearest-neighbor
  graph based on centroid distance. Tests whether finer spatial connectivity
  recovers topology.

---

## Method

For each (scenario, construct) that FAILS Gate 3B in DOE-C3:

1. Record baseline certificate (from DOE-C3).
2. For each morph type:
   a. Construct the morphed adjacency matrix W_morph.
   b. Re-run certification with W_morph (same features, folds, target).
   c. Record morphed certificate.
   d. Compute delta(forward_score), delta(kappa_reconstruct).
3. Evaluate:
   - RESCUE: kappa_reconstruct crosses gate threshold AND forward_score
     change < 0.05. The morph recovered spatial structure.
   - INFLATION: both improve. Suspicious — the morph added signal.
   - INEFFECTIVE: kappa_reconstruct does not improve. The construct
     genuinely does not preserve topology at this resolution.

---

## Gate 3B Threshold

From ADR-020 D7 and the oobleck threshold (ADR-023):

```
kappa_req(sigma) = kappa_base + delta_kappa * sigmoid(steepness * (sigma - sigma_c))
```

For DOE-C4, use kappa_base = 0.7 (default Gate 3 threshold).
sigma = std of per-construct kappa_reconstruct across bootstrap samples
(from DOE-C2a).

---

## Acceptance Criteria

| ID | Criterion | Test |
|----|-----------|------|
| AC-C4-1 | At least one construct is rescued by hydrology morph | kappa_reconstruct crosses threshold |
| AC-C4-2 | Rescue does not inflate forward score | abs(delta_forward) < 0.05 |
| AC-C4-3 | Physical constructs (JRC, Deltares) do not need rescue | Already pass Gate 3B in DOE-C3 |
| AC-C4-4 | knn_spatial produces different outcome than hydrology | Morph type matters |

---

## S3 Output Convention

```
results/s035/doe_c4/
  morph_dispatch_{scenario}.json
  cache/
    morph_certificates_{scenario}.parquet   # 1 row per (construct, morph_type)
    morph_adjacency_stats_{scenario}.json   # Edge counts, density per morph
```

---

## Resource Estimate

- Instance: ml.m5.xlarge
- Small experiment: ~20 certifications (only failing constructs x 3 morphs)
- Wall clock: ~1 hour per scenario
- Cost: ~$0.50

---

## Connection to ADR-012

This experiment tests the core adjoint mechanism:
1. Gate fail triggers morph dispatch (not manual intervention).
2. Morph changes topology (G_R edges), not features.
3. Re-measurement uses the same model fitter — only the adjacency changes.
4. Re-gate uses the same threshold — no threshold shopping.

The constraint from P5 (no runtime learning) is maintained: the morph
is a topology change, not a gradient update. The model is re-fit on the
same data with the same hyperparameters; only the spatial structure used
for kappa_reconstruct computation changes.
