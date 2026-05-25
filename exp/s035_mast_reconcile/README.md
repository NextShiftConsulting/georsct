# S035: MAST vs GeoCert — Direct Taxonomy Comparison

**Status:** PASS  
**Date:** 2026-05-01

## Purpose

Compare MAST (Cemri et al. 2025) and GeoCert failure taxonomies directly.
Both organize multi-agent/solver failures into 3 categories but at different
levels of analysis:

- **MAST**: classifies behavioral *cause* (what agents do wrong)
- **GeoCert**: classifies evaluation *consequence* (what measurement misses)

## Data

- **Source**: [MAST repository](https://github.com/multi-agent-systems-failure-taxonomy/MAST)
- **Traces**: 61 human-annotated (31 AG2, 30 HyperAgent)
- **Annotations**: 22 binary failure-mode labels per trace
- **Loading**: `load_mast.py` reads from git objects (bypasses checkout issues)

## Scripts

| File | Purpose |
|------|---------|
| `load_mast.py` | Load + normalize all 61 annotated traces from MAST repo |
| `signals.py` | Extract 17 structural signals from trajectory data |
| `analyze.py` | Full analysis: statistics, profiles, discrimination tests |

## Key Results

### MAST Category Separation (z-score distance, 12 features)

| Pair | Distance |
|------|----------|
| FC1_Specification vs FC2_Misalignment | **3.155** |
| FC2_Misalignment vs FC3_Verification | 1.862 |
| FC1_Specification vs FC3_Verification | 1.380 |
| FC3_Verification vs baseline | 0.956 |
| FC1_Specification vs baseline | 0.868 |

Categories are well-separated in signal space. FC1 and FC2 are maximally
distant — specification failures look fundamentally different from
misalignment failures at the structural signal level.

### Distinguishing Signal Patterns

| Category | repetition | dominance | handoff | verify | code |
|----------|-----------|-----------|---------|--------|------|
| FC1 Specification | 0.17 | 0.62 | 0.75 | 0.31 | 1.39 |
| FC2 Misalignment | 0.60 | 0.89 | 0.21 | 0.12 | 0.55 |
| FC3 Verification | 0.35 | 0.74 | 0.51 | 0.24 | 1.01 |
| baseline (no failure) | 0.22 | 0.67 | 0.67 | 0.28 | 1.05 |

### Per-Mode Discrimination (modes with n>=5)

Mean pairwise distance: 3.040 (z-score space)  
Max: 6.251 ("Proceed with incorrect assumptions" vs "Step repetition")  
Min: 0.763

### Source Comparison

| Source | n | Mean failures | repetition | handoff | verify |
|--------|---|---------------|-----------|---------|--------|
| AG2 | 31 | 1.6 | 0.02 | 1.00 | 0.42 |
| HyperAgent | 30 | 2.1 | 0.67 | 0.00 | 0.06 |

AG2 and HyperAgent have radically different trajectory structures —
AG2 is multi-turn dialogue (high handoff), HyperAgent is single-agent
log streams (high repetition, zero handoff).

## Dimensionality Argument

MAST: 22 binary annotations per trace (human-labeled)  
GeoCert: continuous n-attribute certificates per (solver × target) pair

The certificate space is fundamentally higher-dimensional than binary labeling.
S018D-posthoc PCA confirms 3+ orthogonal pathology axes exist in this space
without any labels. GeoCert's structural certificates can discover failure
modes that binary annotation schemes cannot represent.

## Reproduction

```bash
# Clone MAST data (already in mast_repo/)
git clone https://github.com/multi-agent-systems-failure-taxonomy/MAST mast_repo

# Run analysis
python analyze.py

# Outputs:
#   results/summary.json  — full results
#   results/signals.csv   — per-trace signal matrix (61 rows × 39 columns)
```
