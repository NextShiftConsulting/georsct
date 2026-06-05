# DOE: RSCT Canonical Certification Ablation Study

**Date**: 2026-04-29
**Status**: DOE_DRAFT
**Paper**: "The Certification Gap: When Leaderboards Rank the Wrong Models"
**Problem**: The paper claims "canonical `yrsn` certificate code path" but s018n/p compute hand-rolled R/S/N and kappa without using the actual gatekeeper or canonical aggregation. The shared/canonical_certifier.py exists and does the right thing — s018h uses it — but the experiments that produce Table 2 (CONUS-27) do not.

---

## Motivation

The paper makes two claims that require honest yrsn code paths:

1. **Section 6**: "per-sample probabilities yield the certificate triple: alpha, kappa, sigma... The SequentialGatekeeper produces the gate decision." — TRUE for text retrieval (uses canonical code).

2. **Section 7**: "We apply the canonical yrsn certificate computation and gatekeeper to geospatial regression, using solver-relative residual terciles..." — **NOT TRUE as implemented**. s018n/p inline `_map_to_rsn()` and compute `kappa = R*(1-N)` without calling `aggregate_scores_from_probs`, `CPGatekeeperInput`, or `SequentialGatekeeper.evaluate()`.

This isn't just a code hygiene issue — it's a paper honesty issue. If we claim the canonical code path, we must use the canonical code path. The ablation study makes the gap scientifically productive: we can ask whether using the full certification protocol changes the results.

---

## Ablation Design

### Arms

| Arm | Label | Code Path | Gatekeeper | Purpose |
|-----|-------|-----------|------------|---------|
| **A0** | Canonical Certificate (4-gate) | `shared/canonical_certifier.py` → `certify_task()` | Yes (Gates 1,3,4 mandatory) | Baseline: what the paper claims |
| **A1** | Hand-Rolled Kappa | Inlined `_map_to_rsn()` + `R*(1-N)` | No | Status quo: what s018n/p actually do |
| **A2** | Kappa Only (no gate) | `canonical_certifier.compute_per_sample_scores()` | No | Isolate: does canonical aggregation matter without gates? |
| **A3** | Full 7-Gate Pipeline | `certify_task()` with all gates enabled | Yes (all 7 gates) | Full protocol: every gate the paper could claim |
| **A4** | Gate Decision Routing | `certify_task()` 7-gate decisions drive routing | Yes (all 7 gates) | Full loop: does the gatekeeper change which solver wins? |

### The 7-Gate Pipeline

The SequentialGatekeeper evaluates gates in fixed order (P2). Evaluation stops at first failure (P3).

| Gate | Name | Input | Threshold | Fail Decision | Default |
|------|------|-------|-----------|---------------|---------|
| **1** | Integrity Guard | `noise_admissibility` (or `N` fallback), `alpha` | N < 0.5 OR alpha >= 0.3 | REJECT | Mandatory |
| **1B** | Task Residual Floor | `task_residual_floor` | task_residual_floor < 0.6 | BLOCK | Off (opt-in) |
| **2** | Consensus | `coherence` | coherence >= 0.4 | BLOCK | Pass-if-absent (evaluates when provided) |
| **3** | Admissibility | `kappa_compat`, `sigma` | sigma <= 0.5 + Oobleck curve | RE_ENCODE | Mandatory |
| **3B** | Trajectory | `trajectory_aux.r_bar` | r_bar >= 0.65 | RE_ENCODE | Fires when `trajectory_aux` is populated |
| **4** | Grounding | `kappa_H`, `kappa_L`, `kappa_interface` (multimodal) or `kappa_L` (unimodal) | Dynamic per level | RE_ENCODE/REPAIR | Pass-if-not-multimodal; geo certs are unimodal |
| **5** | Contract Coverage | `contract_coverage` | coverage >= 0.8 | BLOCK | Off (opt-in) |

### Gate Input Sourcing for Geospatial Domain

Each gate requires specific inputs. For CONUS-27 geospatial regression, the sources are:

| Gate | Input Field | Source for Geo | Available? |
|------|-------------|----------------|------------|
| 1 | `noise_admissibility` / `N` | Classifier softmax → N column | Via classifier rerun |
| 1 | `alpha` | R/(R+N) from classifier softmax | Via classifier rerun |
| 1B | `task_residual_floor` | Table 2 N-ceiling values (0.155-0.593) | YES — precomputed per task |
| 2 | `coherence` | Multi-solver agreement: 3 solver families provide natural multi-source structure | COMPUTE — see design below |
| 3 | `kappa_compat` | R*(1-N) from classifier softmax | Via classifier rerun |
| 3 | `sigma` | std(kappa_per_sample) via `compute_sigma_from_kappa_array` | Via classifier rerun |
| 3B | `r_bar` | Representation stability across OOF folds | COMPUTE — see design below |
| 4 | `kappa_L` | Per-solver kappa (unimodal path) | Via classifier rerun |
| 5 | `contract_coverage` | Certificate field completeness ratio | COMPUTE — deterministic |

#### Gate 1B: Task Residual Floor Evidence

Already available from s018n CONUS-27 analysis. Each task has a precomputed N-ceiling value:
- Range: 0.155 (SoilMoisture) to 0.593 (WindSpeed)
- 14/27 tasks have N-ceiling > 0.3 (moderate noise floor)
- 3/27 tasks have N-ceiling > 0.5 (high — may trigger Gate 1B)

Populate `evidence["task_residual_floor"]` from the task metadata.

#### Gate 2: Coherence from Multi-Solver Agreement

Geospatial naturally has 3 solver families (PCA32, GNN, Spatial Lag). Phasor coherence measures multi-source agreement:

```python
def compute_geo_coherence(kappa_by_solver: dict) -> float:
    """Coherence from multi-solver kappa agreement.
    
    3 solver families provide natural multi-source structure.
    Coherence = 1 - normalized spread of per-solver kappa values.
    High coherence = solvers agree on task difficulty.
    """
    kappas = np.array(list(kappa_by_solver.values()))
    if len(kappas) < 2:
        return 1.0  # single source = no conflict
    spread = kappas.max() - kappas.min()
    # Normalize by theoretical max spread (1.0)
    coherence = 1.0 - spread
    return float(coherence)
```

This maps directly to the paper's "architecture-invariance" finding: if spread is 0.037 (Table 2), coherence = 0.963. The question is whether Gate 2's threshold (0.4) is ever breached.

#### Gate 3B: Trajectory r_bar for Geospatial

r_bar measures "conceptual dominance" — whether the model captures genuine signal vs surface cues. For geospatial regression with OOF predictions:

```python
def compute_geo_r_bar(oof_preds: np.ndarray, y_true: np.ndarray, 
                       fold_indices: list) -> float:
    """Representation stability across OOF folds.
    
    Proxy for r_bar: correlation of per-sample residual ranks
    between different OOF folds. High stability = conceptual
    understanding, not memorization.
    """
    from scipy.stats import spearmanr
    # Compute residual ranks per fold, correlate across folds
    fold_correlations = []
    for i, j in itertools.combinations(range(len(fold_indices)), 2):
        residuals_i = y_true[fold_indices[i]] - oof_preds[fold_indices[i]]
        residuals_j = y_true[fold_indices[j]] - oof_preds[fold_indices[j]]
        # Overlap samples between folds
        overlap = np.intersect1d(fold_indices[i], fold_indices[j])
        if len(overlap) > 10:
            rho, _ = spearmanr(residuals_i[overlap], residuals_j[overlap])
            fold_correlations.append(rho)
    r_bar = np.mean(fold_correlations) if fold_correlations else 0.5
    return float(np.clip(r_bar, 0, 1))
```

**Alternative**: If fold overlap is insufficient, use split-half stability of per-ZCTA residual rankings within each task.

#### Gate 5: Contract Coverage

Deterministic computation: what fraction of the certificate's required fields are populated.

```python
def compute_contract_coverage(cert: CanonicalCertificate) -> float:
    """Fraction of certificate fields that are non-null."""
    required_fields = ["alpha", "kappa", "sigma", "gate_decision",
                       "R_per_sample", "S_per_sample", "N_per_sample",
                       "kappa_per_sample", "alpha_per_sample"]
    populated = sum(1 for f in required_fields 
                    if getattr(cert, f, None) is not None)
    return populated / len(required_fields)
```

### What Changes Between Arms

| Component | A0 (Canonical 4-gate) | A1 (Hand-Rolled) | A2 (Kappa Only) | A3 (Full 7-Gate) | A4 (7-Gate Routing) |
|-----------|----------------------|-------------------|-----------------|-------------------|---------------------|
| R/S/N source | `aggregate_scores_from_probs` | Inlined `_map_to_rsn` | `compute_per_sample_scores` | `aggregate_scores_from_probs` | `aggregate_scores_from_probs` |
| kappa formula | `R*(1-N)` via canonical | `R*(1-N)` via inline | `R*(1-N)` via canonical | `R*(1-N)` via canonical | `R*(1-N)` via canonical |
| alpha | `R/(R+N)` canonical median | Not computed | `R/(R+N)` canonical | `R/(R+N)` canonical | `R/(R+N)` canonical |
| sigma | `compute_sigma_from_kappa_array` | Not computed | `compute_sigma_from_kappa_array` | `compute_sigma_from_kappa_array` | `compute_sigma_from_kappa_array` |
| Aggregation | R-labeled median (canonical) | Per-sample only | R-labeled median | R-labeled median | R-labeled median |
| Gate 1 (Integrity) | Yes | No | No | Yes | Yes |
| Gate 1B (N-ceiling) | No | No | No | **Yes (enabled)** | **Yes (enabled)** |
| Gate 2 (Coherence) | Absent → pass | No | No | **Computed → evaluated** | **Computed → evaluated** |
| Gate 3 (Oobleck) | Yes | No | No | Yes | Yes |
| Gate 3B (Trajectory) | No trajectory_aux | No | No | **trajectory_aux populated** | **trajectory_aux populated** |
| Gate 4 (Grounding) | Unimodal → pass | No | No | Unimodal → pass | Unimodal → pass |
| Gate 5 (Contract) | No | No | No | **Yes (enabled)** | **Yes (enabled)** |
| CPGatekeeperInput | Yes (minimal) | No | No | Yes (full evidence) | Yes (full evidence) |
| Policy config | Default | N/A | N/A | **Enable 1B, 3B, 5** | **Enable 1B, 3B, 5** |
| Gate decision in routing | Logged only | N/A | N/A | Logged only | **Drives solver selection** |

### Hypotheses

**H-ABL-1: Canonical vs hand-rolled kappa produces identical routing.**
If `aggregate_scores_from_probs` and `_map_to_rsn` compute the same per-sample R/S/N from the same softmax inputs, then A0 and A1 routing outcomes should be identical. If they diverge, the inlined code has drifted from canonical.
- **Null**: Routing decisions match on all 27 tasks × 3 solvers.
- **Reject if**: Any task × solver routing assignment differs.

**H-ABL-2: Canonical aggregation (R-labeled median) changes task-level kappa.**
The canonical `aggregate_scores_from_probs` computes median kappa over R-labeled samples only. The inlined code uses per-sample kappa directly for routing. If the R-labeled filter matters, task-level kappa values will differ.
- **Measure**: |kappa_canonical - kappa_inlined| per task.
- **Threshold**: > 0.01 is meaningful (1% of kappa range).

**H-ABL-3: Gate decisions are uniform RE_ENCODE (matching text retrieval finding).**
Section 7 claims "same code path, different domain." If we run the actual gatekeeper on geospatial certificates, do all 27 tasks receive RE_ENCODE (matching the text retrieval universal failure)? Or do some tasks EXECUTE?
- **Measure**: Gate decision distribution across 27 tasks × 3 solvers.
- **Paper impact**: If any task EXECUTEs, the "universal RE_ENCODE" narrative needs qualification.

**H-ABL-4: Gate-driven routing outperforms kappa-only routing.**
Arm A4 uses 7-gate decisions to influence solver selection (e.g., REJECT a solver on a task → exclude from candidate set). Does this produce better holdout R² than pure kappa-argmax routing (A0)?
- **Measure**: Mean and median holdout R² improvement over A0.
- **Threshold**: > 0.005 R² improvement is meaningful.

**H-ABL-5: Certificate values are domain-transferable.**
If canonical certification on geospatial data produces the same alpha/kappa/sigma ranges as text retrieval, the protocol is domain-transferable. If the ranges are qualitatively different, the protocol needs domain-specific calibration.
- **Measure**: Compare (alpha, kappa, sigma) distributions from A0 against s018h text retrieval certificates.
- **Threshold**: Overlapping interquartile ranges → transferable. Non-overlapping → needs calibration.

**H-ABL-6: N-ceiling gate (1B) discriminates high-noise geospatial tasks.**
Gate 1B with threshold 0.6 should BLOCK tasks with N-ceiling > 0.6 (structurally unreliable domains). From Table 2, N-ceiling ranges 0.155-0.593 — all below threshold. If zero tasks are blocked, Gate 1B is non-discriminating for geo. If the threshold were domain-calibrated (e.g., 0.4), how many tasks would be blocked?
- **Measure**: Gate 1B pass/fail distribution at default (0.6) and calibrated (0.4, 0.3) thresholds.
- **Paper impact**: If Gate 1B never fires, it's domain-irrelevant. If it fires at lower thresholds, domain calibration matters.

**H-ABL-7: Multi-solver coherence (Gate 2) is universally high.**
Cross-solver spread in Table 2 is 0.037 (median), implying coherence ≈ 0.96. Gate 2 threshold is 0.4. If all tasks pass Gate 2 trivially, coherence is not a discriminating gate for geo regression. But the *distribution* of per-task coherence may reveal outlier tasks.
- **Measure**: Per-task coherence values, min/max/median across 27 tasks.
- **Threshold**: Gate 2 fires if coherence < 0.4.
- **Paper impact**: If coherence is universally high, geo regression has a structural advantage over text retrieval (where model disagreement is common).

**H-ABL-8: Oobleck dynamic threshold (Gate 3) is the binding constraint.**
Gate 3 uses `kappa_req = 0.5 + 0.4*sigma` with Landauer tolerance (epsilon_L = 0.05). For geo certificates with moderate sigma, what fraction of tasks × solvers pass the oobleck curve? If Gate 3 is the gate that universally fires RE_ENCODE, it confirms the "certification gap" is about kappa insufficiency, not noise or coherence.
- **Measure**: Per-task sigma, kappa_req, actual kappa, and zone assignment (pass/gray/fail).
- **Paper impact**: Identifies which gate is the binding constraint for the "universal RE_ENCODE" finding.

**H-ABL-9: Trajectory r_bar (Gate 3B) separates surface-cue from conceptual models.**
r_bar computed from OOF fold stability should correlate with holdout R². Models that generalize (high R²) should have high r_bar (conceptual dominance). Surface-cue models (low r_bar) should have lower R² despite possibly high training R².
- **Measure**: Spearman correlation between r_bar and holdout R² across 27 tasks × 3 solvers.
- **Threshold**: |rho| > 0.3 indicates r_bar captures genuine signal quality.

**H-ABL-10: 7-gate pipeline produces richer decision taxonomy than 4-gate.**
With all 7 gates enabled, the decision outcomes should include REJECT, BLOCK, RE_ENCODE, REPAIR, and EXECUTE — not just RE_ENCODE. The distribution of which gate fails first reveals the *type* of certification gap, not just its existence.
- **Measure**: First-failing gate distribution across 27 × 3 = 81 certificates.
- **Paper impact**: Transforms "universal RE_ENCODE" into "Gate X is the binding constraint" — a more informative finding for the paper.

---

## Experiment Structure

### S018V: Canonical Certification Ablation

**New experiment**: `s018v_canonical_ablation/`

This is NOT a rewrite of s018n/p. It is a standalone ablation that:
1. Loads the same OOF predictions and softmax probabilities used by s018n/p
2. Runs all 5 arms on the same data
3. Compares routing outcomes, kappa values, and gate decisions (including full 7-gate)
4. Produces a single evidence package with arm-by-arm comparison

```
s018v_canonical_ablation/
  DOE_LOCKED.md
  run_s018v.py          # All 5 arms in one script
  gate_inputs.py        # Gate 1B/2/3B/5 input computation helpers
  sagemaker_s018v.py    # SageMaker launcher
  evidence/             # Results
```

### Input Artifacts (from prior experiments)

| Artifact | Source | S3 Path |
|----------|--------|---------|
| OOF predictions | s018h/s018p | `s3://yrsn-datasets/rsct_curriculum/series_018/oof_artifacts/` |
| Softmax probabilities | s018p B6 calibration | `s018p_b6_calibrated_scores.npz` |
| Tree checkpoint | s018n | `s3://yrsn-checkpoints/pair-score/` |
| PCA32 features | shared | `s3://yrsn-datasets/rsct_curriculum/series_018/processed/` |

### Dependencies

| Dependency | Status | Required For |
|-----------|--------|-------------|
| `shared/canonical_certifier.py` | EXISTS | Arms A0, A2, A3, A4 |
| `yrsn` wheel (with CPGatekeeperInput) | EXISTS (ADR-015 updated) | Arms A0, A2, A3, A4 |
| `yrsn-controlplane` (SequentialGatekeeper + GatekeeperConfig) | EXISTS | Arms A0, A3, A4 |
| `yrsn-controlplane` TrajectoryAux dataclass | EXISTS | Arms A3, A4 (Gate 3B) |
| s018p softmax outputs | NOT ON S3 — classifier rerun needed | All arms |
| s018n `_map_to_rsn` code | EXISTS (inlined in run_s018n.py) | Arm A1 |
| CONUS-27 N-ceiling values | EXISTS (Table 2 precomputed) | Arms A3, A4 (Gate 1B) |
| OOF fold indices | NEED TO VERIFY on S3 | Arms A3, A4 (Gate 3B r_bar) |
| scipy (spearmanr) | Standard pip install | Arms A3, A4 (Gate 3B) |

---

## Implementation Plan

### Step 1: Verify Input Artifacts

Before writing code, verify that the required softmax probabilities exist on S3. The ablation needs per-sample (n_zcta, 3) probability arrays for each (task, solver) pair.

```bash
aws s3 ls s3://yrsn-datasets/rsct_curriculum/series_018/oof_artifacts/ --recursive | head -20
```

If softmax probs aren't stored (only final predictions), we need to rerun the classifier stage. This is a blocker.

### Step 2: Create s018v Experiment Directory

```
yrsn-experiments/exp/series_018/s018v_canonical_ablation/
  DOE_LOCKED.md       # This DOE (locked after review)
  run_s018v.py        # Main script (all 5 arms)
  gate_inputs.py      # compute_geo_coherence, compute_geo_r_bar, etc.
  sagemaker_s018v.py  # Launcher
  evidence/           # Results
  requirements.txt    # yrsn, yrsn-controlplane, scipy
```

### Step 3: Implement run_s018v.py

```python
"""
run_s018v.py -- Canonical Certification Ablation (Full 7-Gate Pipeline)

Five arms comparing hand-rolled vs canonical RSCT certification
on CONUS-27 geospatial regression, exercising all 7 gates.

Arms:
  A0: Canonical 4-gate (shared/canonical_certifier.py, default gatekeeper)
  A1: Hand-rolled (inlined _map_to_rsn, kappa only, no gatekeeper)
  A2: Canonical kappa only (canonical aggregation, no gatekeeper)
  A3: Full 7-gate pipeline (all gates enabled, logged only)
  A4: 7-gate routing (gate decisions influence solver selection)
"""

from shared.canonical_certifier import (
    certify_task, compute_per_sample_scores, CanonicalCertificate
)
from yrsn.controlplane import SequentialGatekeeper, GatekeeperConfig

# --- Gate input computation helpers ---

def compute_geo_coherence(kappa_by_solver: dict) -> float:
    """Coherence from multi-solver kappa agreement (Gate 2).
    
    3 solver families -> pairwise agreement -> coherence = 1 - spread.
    """
    kappas = np.array(list(kappa_by_solver.values()))
    if len(kappas) < 2:
        return 1.0
    spread = kappas.max() - kappas.min()
    return float(1.0 - spread)

def compute_geo_r_bar(oof_preds, y_true, fold_indices) -> float:
    """Representation stability across OOF folds (Gate 3B).
    
    Proxy for r_bar: rank correlation of residuals across folds.
    """
    from scipy.stats import spearmanr
    fold_correlations = []
    for i, j in itertools.combinations(range(len(fold_indices)), 2):
        overlap = np.intersect1d(fold_indices[i], fold_indices[j])
        if len(overlap) > 10:
            r_i = y_true[overlap] - oof_preds[fold_indices[i]][overlap]
            r_j = y_true[overlap] - oof_preds[fold_indices[j]][overlap]
            rho, _ = spearmanr(r_i, r_j)
            fold_correlations.append(rho)
    return float(np.clip(np.mean(fold_correlations), 0, 1)) if fold_correlations else 0.5

def compute_contract_coverage(cert) -> float:
    """Certificate field completeness ratio (Gate 5)."""
    required = ["alpha", "kappa", "sigma", "gate_decision",
                "R_per_sample", "S_per_sample", "N_per_sample",
                "kappa_per_sample", "alpha_per_sample"]
    populated = sum(1 for f in required if getattr(cert, f, None) is not None)
    return populated / len(required)

# --- 7-gate gatekeeper config ---

def make_7gate_config() -> GatekeeperConfig:
    """Enable all 7 gates for full pipeline evaluation."""
    return GatekeeperConfig(
        enable_task_residual_floor_gate=True,       # Gate 1B (default off)
        gate_2_require_coherence=True,    # Gate 2 fail-closed
        enable_gate_3b=True,             # Gate 3B (default on)
        enable_contract_coverage_gate=True,  # Gate 5 (default off)
    )

# --- Arms ---

def arm_a0_canonical(probs, labels, task, model):
    """Canonical 4-gate: certify_task() with default gatekeeper."""
    return certify_task(probs, labels, task, model, "mlp")

def arm_a1_handrolled(probs, task, model):
    """Status quo: inlined _map_to_rsn from s018n."""
    # Exact copy of s018n/s018p inlined code
    ...

def arm_a2_kappa_only(probs, task, model):
    """Canonical per-sample scores, no gatekeeper."""
    scores = compute_per_sample_scores(probs)
    ...

def arm_a3_full_7gate(probs, labels, task, model, task_residual_floor,
                       coherence, r_bar, contract_coverage):
    """Full 7-gate: all gates enabled, evidence dict populated."""
    gatekeeper = SequentialGatekeeper(config=make_7gate_config())
    # Build CPGatekeeperInput with full evidence
    cert = certify_task_7gate(
        probs, labels, task, model, "mlp", gatekeeper,
        task_residual_floor=task_residual_floor,
        coherence=coherence,
        r_bar=r_bar,
        contract_coverage=contract_coverage,
    )
    return cert

def arm_a4_gate_routing(probs, labels, task, model, task_residual_floor,
                         coherence, r_bar, contract_coverage):
    """7-gate certification + gate decision drives routing."""
    cert = arm_a3_full_7gate(probs, labels, task, model,
                              task_residual_floor, coherence, r_bar, 
                              contract_coverage)
    # Use cert.gate_decision to filter solver candidates
    ...
```

### Step 4: Run on SageMaker

Single processing job, ml.m5.xlarge (CPU only, no GPU needed). All 5 arms on 27 tasks × 3 solvers = 405 certifications. Should complete in < 45 minutes.

### Step 5: Analyze Results

**Primary comparison table** (one row per task × solver):

```
task | solver | kappa_A0 | kappa_A1 | delta | gate_A0(4) | gate_A3(7) | first_fail_gate | route_A0 | route_A4 | match?
```

**7-gate diagnostic table** (one row per task × solver, A3 arm):

```
task | solver | G1(N,alpha) | G1B(n_ceil) | G2(coh) | G3(kappa,sigma,zone) | G3B(r_bar) | G4(kappa_L) | G5(cov) | decision | first_fail
```

**Summary metrics**:
- Mean |kappa_A0 - kappa_A1| across 27 tasks
- Gate decision distribution by arm:
  - A0 (4-gate): EXECUTE/RE_ENCODE/REJECT counts
  - A3 (7-gate): EXECUTE/RE_ENCODE/REJECT/BLOCK/REPAIR counts
- First-failing gate histogram (A3): which gate is the binding constraint?
- N-ceiling sensitivity: pass counts at thresholds 0.6, 0.5, 0.4, 0.3
- Coherence distribution: min, median, max across 27 tasks
- r_bar vs holdout R² correlation (Spearman rho)
- Routing agreement rate (A0 vs A1, A0 vs A4)
- R² improvement from 7-gate routing (A4 vs A0)

---

## Impact on Paper

### If Arms Agree (A0 ≈ A1)

The inlined code was functionally equivalent. Fix the code path to use canonical anyway (for honesty), note in paper: "We verified that the inlined computation matches the canonical library output (ablation in Appendix)."

### If Arms Diverge (A0 ≠ A1)

The inlined code drifted. This is scientifically interesting:
- Report the divergence as an ablation finding
- Rerun Table 2 with canonical values
- If Table 2 conclusions change, this is a major finding about certification protocol sensitivity
- If conclusions hold despite numerical differences, this demonstrates robustness

### 4-Gate vs 7-Gate Decision Comparison (A0 vs A3)

**Same outcome (4-gate ≈ 7-gate)**:
- The optional gates (1B, 3B, 5) don't change the picture. Paper can mention: "Enabling the full 7-gate pipeline produces identical outcomes, confirming the binding constraint is Gate 3 (admissibility)."

**Different outcome (4-gate ≠ 7-gate)**:
- Some tasks blocked at Gate 1B (N-ceiling) or Gate 2 (coherence) before reaching Gate 3. This reshapes the narrative from "kappa insufficiency" to "domain noise floor" or "measurement incoherence."
- Paper finding: "The certification gap has multiple sources — not just kappa deficiency."

### First-Failing Gate Distribution

The most informative single metric from the ablation. Possible outcomes:

| Pattern | Interpretation | Paper Narrative |
|---------|---------------|-----------------|
| All fail at Gate 3 | Kappa insufficiency is the binding constraint | "The certification gap is a kappa gap" |
| Mix of Gate 1 and Gate 3 | Some tasks have noise issues, others kappa issues | "Two distinct failure modes in the certification gap" |
| Gate 1B blocks high-noise tasks | Domain noise floor is a separate constraint | "Some geospatial tasks are structurally uncertifiable" |
| Gate 2 blocks some tasks | Multi-solver disagreement reveals unstable tasks | "Architecture-invariance holds in aggregate but not per-task" |
| Any task reaches EXECUTE | The certification gap is not universal | "Certification is domain-dependent — some geo tasks pass" |

### Gate Decision Finding

If all 81 certificates receive RE_ENCODE/REJECT/BLOCK (zero EXECUTE):
- Strengthens the "universal certification gap" narrative
- Section 7 can honestly say "the same protocol that flagged text models also flags geospatial models — across all 7 gates"

If some receive EXECUTE:
- Even more interesting — what distinguishes tasks that pass from those that fail?
- N-ceiling correlation with gate outcome?
- Coherence as a discriminating feature?
- New paper finding: "gate passage correlates with [N-ceiling / coherence / r_bar]" (if true)

### r_bar as a Novel Geospatial Quality Signal

If H-ABL-9 shows significant r_bar–R² correlation:
- New contribution: "Representation stability (r_bar) predicts geospatial model quality"
- Potential Section 9 (future work) item: "r_bar as a domain-agnostic quality proxy"

---

## Downstream: Rewiring S018Q

After s018v establishes the canonical baseline, s018q should consume canonical certificates (from certify_task), not hand-rolled certificate matrices. The change:

**Before** (current s018q):
```
s018p softmax → inlined kappa/alpha → certificate_matrix → PCA/SVD → spectral modes
```

**After**:
```
s018p softmax → canonical_certifier.certify_task() → certificate_matrix → PCA/SVD → spectral modes
```

The certificate_matrix input changes from inlined values to canonical values. If s018v shows A0 ≈ A1, this is a no-op. If A0 ≠ A1, the spectral modes may shift — which is itself a finding about mode stability under certification protocol variation.

**7-gate enrichment**: If A3 reveals interesting gate-level structure (e.g., Gate 1B blocks some tasks, Gate 3B separates quality), s018q's certificate_matrix could include gate-level features (per-gate pass/fail, first_fail_gate ordinal) alongside kappa/alpha/sigma. This would test whether gate decisions carry spectral information beyond the scalar certificate values.

S018R and S018S are unaffected — R uses native solver spectra (no RSCT), S bridges Q and R outputs.

---

## Sequencing

```
1. Verify S3 artifacts exist           (30 min)
2. Create s018v directory + DOE_LOCKED  (done by this doc)
3. Implement run_s018v.py              (2-3 hours)
4. Commit + push + launch SageMaker    (30 min)
5. Analyze results                     (1-2 hours)
6. Decision: rewire s018n/p or note    (depends on findings)
7. If rewired: rerun Table 2           (SageMaker job)
8. Update paper sections 7, 8, 9       (based on findings)
9. S018Q consumes canonical certs      (if Q hasn't run yet — it hasn't)
10. S018R runs independently            (no changes needed)
11. S018S bridges Q and R               (after both complete)
```

**Critical path**: Step 1 (verify S3 artifacts). If softmax probabilities aren't stored, we need a classifier rerun first.

---

## Acceptance Criteria

- [ ] All 5 arms produce complete results for 27 tasks × 3 solvers (405 certifications)
- [ ] Arm A0 uses `certify_task()` from shared/canonical_certifier.py — verified by import trace
- [ ] Arm A1 exactly reproduces s018n `_map_to_rsn()` logic — verified by code diff
- [ ] Arm A3 enables all 7 gates via `GatekeeperConfig` — verified by config dump in evidence
- [ ] Per-arm kappa/alpha/sigma values recorded in evidence JSON
- [ ] Gate decisions (A0, A3, A4) recorded per task × solver
- [ ] Per-gate pass/fail recorded for A3 (7-gate diagnostic table)
- [ ] First-failing gate histogram for A3 in evidence
- [ ] Gate 1B sensitivity analysis at thresholds 0.6, 0.5, 0.4, 0.3
- [ ] Per-task coherence values computed and recorded
- [ ] Per-task r_bar values computed and recorded
- [ ] r_bar vs holdout R² Spearman correlation computed
- [ ] Contract coverage values computed (should be 1.0 for canonical arms)
- [ ] Routing comparison table (A0 vs A1, A0 vs A4 agreement) in evidence
- [ ] H-ABL-1 through H-ABL-10 adjudicated with evidence
- [ ] Paper sections updated based on findings
