# S019D Interaction Diagnostic — Proof Artifact

## Claim being proven

Architecture spread (within-task R² gap across 3 core embeddings) is NOT correlated
with N_ceiling across the 27 CONUS-27 tasks.

**Result: Spearman r = 0.071, p = 0.72, n = 27** (stable across 3 seeds)

## What this means for §4.4

- "Ceiling-conditioned architecture sensitivity" is false — no continuous interaction
  between N_ceiling and architecture spread exists in the data.
- The conditional pattern is categorical: elevated spread concentrates on
  environmental targets (GNN spatial inductive bias advantage), not on low-ceiling tasks.
- Correct §4.4 framing: "Substrate-Conditioned Architecture Sensitivity"
- Section numbers: empirical section = §4 (line 186), architecture subsection = §4.4 (line 244).
  Labels §6/§6.3 used in an earlier agent session were wrong.

## Files in this artifact

| File | Contents |
|------|----------|
| `s019d_interaction_proof_table.csv` | 27-task vectors (arch_spread, n_ceiling) for seeds 42/123/456 |
| `s019d_interaction_correlations.csv` | Spearman + Pearson r, p, n per seed |
| `reproduce_interaction_diagnostic.py` | Script to recompute from raw JSON |
| `PROOF_README.md` | This file |

## Spread definition

- Embeddings: pca_v1, spatial_lag_v1, gnn_v2 (3 core families)
- Per task: mean R² per embedding across 5 folds, then max − min
- N_ceiling: mean of n_ceiling field across folds per task

## Primary source

- File: `data/s019d/seed_42/s019d_results.json`
- S3: `s3://swarm-yrsn-datasets/rsct_curriculum/series_019/results/s019d/seed_42/s019d_results.json`
- Local and S3 verified identical (27/27 tasks, all N_ceiling values match to 4 decimal places)

## To reproduce

```bash
python reproduce_interaction_diagnostic.py data/s019d/seed_42/s019d_results.json
```

Expected: `Spearman r=0.0714, p=0.7233, n=27`
