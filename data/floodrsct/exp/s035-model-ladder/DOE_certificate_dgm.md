# DOE: RSCT Certification + DGM Routing — Quality Measurement + Orchestration

**Experiment:** s035-model-ladder / Phase 4.5 (certificates) + Phase 6 (DGM routing)
**Role:** Measurement plane (yrsn) interprets solver outputs and prescribes routing
**Status:** DESIGNED (DOE phase, not launched)
**Depends on:** All training phases + kappa diagnostics complete

---

## Motivation

The model ladder trains solvers and computes diagnostics. This phase uses yrsn's
production infrastructure to (a) certify each cell with an RSN certificate, and
(b) demonstrate that DGM routing prescribes the correct representation level
based on certificate signals. This closes the loop: the system that diagnoses
the problem also certifies the fix and routes to the optimal arm.

---

## Part A: RSCT Certification (Phase 4.5)

### Hypothesis

No formal hypothesis — certification is descriptive. The paper claim is:

> "Certificate R (signal captured) increases across the representation ladder,
> while S_sup (leakage) decreases, demonstrating progressive quality improvement
> that the RSCT certificate captures."

### Certificate Computation

For each (scenario, target, level) cell, compute from solver results:

| Certificate Field | Source | Formula |
|-------------------|--------|---------|
| R | Spatial-blocked R2 (HistGBDT) | clamp(r2_spatial_blocked, 0, 1) |
| S_sup | Leakage gap | clamp(r2_random - r2_spatial, 0, 1) |
| N | Residual | 1 - R - S_sup (simplex closure) |
| alpha | Quality signal | compute_alpha(R, N) = R / (R + N) |
| omega | In-distribution confidence | compute_omega(S_sup) = 1 - S_sup |
| kappa | Mean kappa proxy | Mean of 4 kappa diagnostics for the cell |
| sigma | Cross-fold variance | std([kappa per fold]) |

Uses:
- `yrsn.core.certificates.core.YRSNCertificate`
- `yrsn.core.quality.alpha.compute_alpha`
- `yrsn.core.quality.omega.compute_omega`

### Edge Cases

| Condition | Handling |
|-----------|---------|
| Negative R2 | R = 0, N = 1 - S_sup |
| S_sup > 1 | Clamp to 1, R = 0, N = 0 |
| All folds fail | Cell excluded from certificate table |

### Certificate Evolution Table (Paper Figure 4)

```
scenario | target | R_R0 | S_R0 | N_R0 | kappa_R0 | R_R1 | S_R1 | N_R1 | kappa_R1 | R_R2 | S_R2 | N_R2 | kappa_R2 | verdict
```

Verdict:
- PASS: kappa >= 0.7 at final level (yrsn gate 4 threshold)
- WARN: 0.5 <= kappa < 0.7
- FAIL: kappa < 0.5

### Expected Patterns

- R increases R0 -> R1 -> R2 (more signal captured)
- S_sup decreases R0 -> R1 (W-matrix fixes spatial leakage)
- N decreases overall (less unexplained variance)
- kappa increases (more cells pass quality gate at higher levels)

---

## Part B: DGM Routing Analysis (Phase 6)

### Hypothesis

**H5 (exploratory):** DGM morph routing, informed by progressive certificates,
selects the same (representation, solver) arm that exhaustive comparison shows
is best.

### DGM Architecture Mapping

| DGM Concept | s035 Equivalent |
|-------------|-----------------|
| G_R event node | (scenario, target, zcta_id) with certificate |
| G_S agent node | (solver, level) — e.g., HistGBDT_R1 |
| phi morphism | Certificate-based routing to best arm |
| MorphType.PRUNING | kappa < 0.3: unpredictable, flag for review |
| MorphType.EVENT_EXPANSION (RE_ENCODE) | S_sup > 0.2: upgrade representation level |
| MorphType.REPAIR | sigma > 0.15: ensemble solvers |
| MorphType.VERIFICATION (EXECUTE) | kappa >= 0.7: certify and proceed |

Uses:
- `yrsn.core.dgm_unified.DualGraphSystem`
- `yrsn.core.dgm_unified.MorphType`
- Thresholds from `yrsn_controlplane.SequentialGatekeeper` defaults

### Routing Decision Logic

```
For each (scenario, target) cell:
  1. Read cert_R0
  2. If kappa >= 0.7 → EXECUTE@R0 (no upgrade needed)
  3. If S_sup > 0.2 → RE_ENCODE → try R1
  4. Read cert_R1
  5. If kappa >= 0.7 → EXECUTE@R1
  6. If diag_transfer < 0.5 → RE_ENCODE → try R2
  7. Read cert_R2
  8. If kappa >= 0.7 → EXECUTE@R2
  9. If sigma > 0.15 → REPAIR (ensemble HistGBDT + Ridge)
  10. If kappa < 0.3 → PRUNING (flag for human review)
```

### DGM Routing Table (Paper Table 4)

```
scenario | target | cert_R0 | cert_R1 | cert_R2 | morph_decision | recommended_arm | actual_best_arm | correct?
```

Where `actual_best_arm` = the (level, solver) with highest spatial-blocked R2.

### Hit Rate Metrics

| Metric | Definition |
|--------|-----------|
| Exact match | DGM recommends the arm with highest R2 |
| Near-optimal | DGM recommends within 0.02 R2 of best |
| Direction correct | DGM upgrades when upgrade helps, stays when it doesn't |

With 7 cells this is descriptive. Report with exact binomial CI.

---

## Outputs

| Artifact | S3 Key | Phase |
|----------|--------|-------|
| R0 certificates | `results/s035/certificates_r0.json` | 4.5a |
| R1 certificates | `results/s035/certificates_r1.json` | 4.5b |
| R2 certificates | `results/s035/certificates_r2.json` | 4.5c |
| DGM routing | `results/s035/dgm_routing.json` | 6 |

---

## Success Criteria

| Criterion | Threshold | If FAIL |
|-----------|-----------|---------|
| Certificate R increases across levels | R_R2 > R_R0 for majority | Ladder doesn't improve signal capture |
| Certificate S_sup decreases R0→R1 | S_R1 < S_R0 for majority | W-matrix didn't fix leakage |
| DGM hit rate > random (1/6 arms) | > 16.7% | DGM routing no better than random |
| DGM near-optimal > 50% | > 50% | Routing useful but imprecise |

---

## Kill Rules

- All certificates show kappa < 0.3 at R2 → representation ladder fails to reach quality threshold; report as limitation
- DGM routes everything to same arm → thresholds miscalibrated for this domain
- Certificate R DECREASES across levels → feature additions are noise

---

## Compute

| Resource | Value |
|----------|-------|
| Instance | local or ml.m5.large |
| Est. duration | ~2 min per level (certificates), ~2 min (DGM routing) |
| Dependencies | All results JSONs, all kappa diagnostics, yrsn package |
| GPU | NOT NEEDED |

---

## DO NOT Constraints

- Do NOT train any models in this phase (certificates and routing are post-hoc analysis)
- Do NOT modify yrsn's gate thresholds to improve DGM hit rate
- Do NOT use DGM routing to re-run experiments (it's descriptive, not prescriptive)
- Do NOT report DGM hit rate without binomial CI
