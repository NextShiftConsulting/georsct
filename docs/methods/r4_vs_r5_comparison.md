# R4 vs R5: Measurement vs Adaptation

**Date:** 2026-06-09
**Context:** s035-model-ladder representation levels R4 and R5

---

## Summary

R4 measures what frozen VLMs can do. R5 optimizes how a frozen VLM is
used. R4 must run first — it establishes the baseline. R5 takes one R4
VLM and asks whether harness edits can close the gap without retraining.

---

## Structural Comparison

| Dimension | R4 (VLM Assessment) | R5 (Harness Evolution) |
|-----------|---------------------|------------------------|
| **Question** | Do VLMs see flood risk signal? | Can you improve a frozen VLM without retraining? |
| **VLM role** | Solver — one-shot assessment | Agent — repeated with evolving harness |
| **What's frozen** | VLM backbone | VLM backbone + map renderer + labels + scoring + splits |
| **What varies** | Which VLM (5-way comparison) | How it's prompted: evidence template, feature policy, rubric, scenario memory |
| **Primary metric** | Spearman rho (risk_score vs NFIP claims) | zone_macro_f1 (FEMA zone classification) |
| **Second model** | 3 Judge models (quorum: Gemini, GPT-4o, Claude) | 1 Evolver model (proposes harness patches) |
| **Quality audit** | Claim-level: R_claim / S_sup_claim / N_claim | Output-level: activation, adherence, grounding, schema validity |
| **Certificate role** | Gate routing: cert → evaluate_gates → Trust/Review/Escalate/Suppress | Trajectory tracking: kappa_compat across evolution steps |
| **Counterfactual** | Action hold-out: removing claim i changes the route? | Patch hold-out: does this harness edit improve or regress? |
| **Optimization** | None — pure measurement | Yes — iterative: failure_report → evolver patch → validate → accept/reject |
| **Failure taxonomy** | grounded / filler / fabricated (per claim) | false_negative, false_positive, schema_violation, unprovided_claim, spatial_outlier (per ZCTA) |
| **Spatial diagnostics** | Per-ZCTA route (Trust/Review/Escalate/Suppress) | Moran's I on residuals — diagnostic only, **never reward** |

---

## R4 Architecture

Five frozen VLMs each receive a rendered flood-zone map and structured
text evidence per ZCTA and produce a structured risk assessment.

**Models:** GPT-4o-mini, Gemini 2.0 Flash, Jina VL, Nova Lite, Qwen2.5-VL-72B

**Tier 1 — claim grading:** Each response claim is graded programmatically
against source feature layers as R_claim (grounded), S_sup_claim
(supported filler), or N_claim (fabricated).

**Tier 2 — relevance quorum:** Three independent judge models
(Gemini 2.5 Flash, GPT-4o, Claude Sonnet) score each claim on a
5-axis rubric. Krippendorff's alpha measures inter-rater reliability.
Calibration gate requires alpha >= 0.67 with bootstrap CI.

**Decision routing (DGM):** Claim grades aggregate into an RSN
certificate (R, S_sup, N) → evaluate_gates() → Floodcaster route
(Trust / Review / Escalate / Suppress). Claim relevance is measured
by action hold-out: phi(C) != phi(C_{-i}) means claim i is
decision-relevant.

**Hypotheses:**
- H7 (gate): >= 1 VLM achieves rho > 0.3 with NFIP claims AND null controls discriminated
- H8 (secondary): pairwise VLM rho > 0.7 (solver robustness)
- H9 (exploratory): VLM scores correlate with R0 kappa diagnostics

**Code:**
- `run_vlm_assessment.py` — Phase R4.3: VLM inference
- `score_vlm_quality.py` — Phase R4.4a: tier1 claim grading
- `run_tier2_quality.py` — Phase R4.4b: tier2 relevance quorum
- `dgm_route_engine.py` — DGM route engine (cert → route via gate pipeline)
- `r4_tier2_relevance_quorum.py` — quorum protocol (INV-1 through INV-7)
- `compute_vlm_comparison.py` — Phase R4.5: money table + H7-H9

---

## R5 Architecture

Frozen-backbone VLM harness evolution with RSCT certificate trajectories.
Tests whether Lin et al. harness-updating / harness-benefit decomposition
holds for spatial VLM flood decision tasks.

**Roles:**
- Agent VLM (pilot: gemini_flash) — frozen, produces assessments
- Evolver model (pilot: claude) — proposes harness patches from failure reports

**Editable surface** (the "harness"):
- `evidence_template` — preamble, feature ordering, max features
- `feature_policy` — which features from event_df enter the VLM prompt
- `rubric` — risk levels, scoring instructions, schema requirements
- `scenario_memory` — lessons learned from prior evolution steps

**Evolution loop:**
```
Step 0: Baseline
    VLM(H_0, train_ids) → zone_macro_f1 baseline

Step t (for t = 1..n_steps):
    1. build_failure_report()
       - Classify failures: FN, FP, schema_violation, unprovided_claim, etc.
       - Redact held-out ZCTAs
       - Cap examples per failure group
    2. evolver.propose_patch(H_t, failure_report)
       - Evolver receives system prompt with rules (no memorization, no held-out leakage)
       - Returns JSON Patch operations on editable components
    3. validate_patch()
       - Allowlisted paths only
       - No held-out ZCTA references
       - No weakening schema requirements
    4. apply_patch(H_t) → H_candidate
    5. VLM(H_candidate, validation_ids) → val_score
    6. Accept if ALL gates pass:
       a. primary delta >= 0 (zone_macro_f1 non-decreasing)
       b. schema_validity_rate not regressing (> -0.01)
       c. unprovided_claim_rate not increasing (> +0.01)

Final:
    VLM(H_final, test_ids) → test_score + certificate_trajectory.json
```

**Failure types** (attribution.py):
- `false_negative_high_reference` — VLM says low risk, reference says flood zone
- `false_positive_low_reference` — VLM says high risk, reference says no flood
- `schema_violation` — output doesn't match expected schema
- `missing_evidence_citation` — VLM didn't cite supplied evidence
- `unprovided_external_claim` — VLM invented unsupported facts
- `low_confidence_correct` — correct but low confidence
- `spatial_outlier` — disagrees with all neighbors

**Diagnostics** (scoring.py):
- `activation_rate` — all required artifacts (map, evidence, rubric) loaded
- `adherence_rate` — followed schema AND used structured evidence
- `grounding_rate` — cited visual + structured evidence, no external claims
- `schema_validity_rate` — output matches expected schema
- `unprovided_claim_rate` — fraction inventing unsupported facts
- `spatial_residual_morans_i` — diagnostic only, NEVER optimization target

**Hypotheses:**
- H1 (harness-updating flatness): different evolvers converge to similar improvements
- H2 (harness-benefit nonmonotonicity): better base VLMs may gain less from harness edits
- H3 (certificate trajectory monotonicity): accepted patches produce non-decreasing kappa/adherence
- H4 (coherence gaming): some edits improve Moran's I without improving F1
- H5 (transfer brittleness): harnesses evolved on Houston transfer poorly to NYC

**Code:**
- `r5_harness_evolution/protocol.py` — step loop and evolution runner
- `r5_harness_evolution/harness_schema.py` — HarnessVersion, EditableComponents, CertificateTrajectory
- `r5_harness_evolution/evolver.py` — evolver prompt and response parsing
- `r5_harness_evolution/scoring.py` — zone_macro_f1, JudgmentMetrics, Moran's I
- `r5_harness_evolution/attribution.py` — failure classification and report building
- `r5_harness_evolution/validators.py` — patch validation (allowlist, held-out leakage)
- `r5_harness_evolution/apply_patch.py` — JSON Patch application
- `r5_harness_evolution/splitters.py` — leave-scenario-out splits
- `r5_harness_evolution/harness_store.py` — versioned artifact persistence
- `r5_harness_evolution/constants.py` — S3 paths, model lists
- `r5_harness_evolution/contract.yaml` — experiment contract (pilot status)

---

## Relationship

R4 is the thermometer. R5 asks whether you can recalibrate the
thermometer without replacing it.

- R4 establishes the baseline measurement for each VLM
- R5 takes one R4 VLM (gemini_flash) and evolves its harness
- R4's quality audit is claim-level (individual assertions graded)
- R5's quality audit is output-level (overall adherence and schema compliance)
- R4 uses DGM for decision routing (certificate → route)
- R5 uses RSCT for trajectory tracking (certificate evolution across steps)
- Both share the same frozen map renderer and structured evidence inputs
- Both report separately from the R0-R2 money table

---

## Certificate Usage Comparison

**R4:** Certificates are per-ZCTA response. Claims aggregate to
(R, S_sup, N) → evaluate_gates() → route. The certificate answers
"is this VLM response trustworthy enough to act on?"

**R5:** Certificates are per-evolution-step. The kappa_compat at each
step tracks whether harness edits are improving representation quality.
The certificate answers "is this harness edit improving or degrading
the VLM's reliability?"

Both use `kappa_compat = R * (1 - N)` as the compatibility proxy.
R4 computes it from claim grading counts. R5 computes it from the
simplex block in `build_rsct_block()` (currently unpopulated in pilot —
`simplex=None` → warnings).

---

## Status

| Level | Pipeline implemented | Results available | Hypotheses tested |
|-------|---------------------|-------------------|-------------------|
| R4 | Yes (tier1 complete, tier2 in progress) | Tier1: 25/25 jobs on S3 | Pending tier2 + comparison |
| R5 | Yes (11 files, full protocol) | None (pilot status) | Open (H1-H5) |
