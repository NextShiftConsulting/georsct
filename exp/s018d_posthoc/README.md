# S018D-Posthoc: Certificate Structure Analysis

## Purpose

Show that aggregate certificate metrics contain partial structure but fail
to detect the noisy-solver pathology. This is the bridge from S018D
(raw certificates) to S018Y-U (label-free ablation deltas).

## Hypothesis

H1: Certificate-metric space has multiple orthogonal pathology axes (not a
single quality axis).

H2: Noisy solver is indistinguishable from serious solvers under aggregate
certificate metrics (nearest-neighbor test).

H3: Controls (mean_baseline, noisy_solver) are NOT a coherent group —
distance between them exceeds mean inter-solver distance.

## Inputs

- `data/s018d/certificate_rsn.parquet` — 324 certificates (12 solvers × 27 targets)
- `data/s018d/solver_metrics.parquet` — per-solver summary metrics
- `data/s018d/summary.json` — experiment metadata

## Run

```bash
python analyze.py
```

## Outputs

- `results/summary.json`
- `results/solver_metric_summary.csv`
- `results/pairwise_solver_distances.csv`
- `results/pca_pathology_axes.csv`
- `results/clustering_summary.json`
- `results/paper_note.md`

## Paper use

S018D-posthoc shows that certificate summaries are structured but incomplete:
they separate scale and instability pathologies, while hiding uniform/noisy
pathologies. This motivates S018Y-U (label-free ablation deltas that can
reveal what aggregate metrics cannot).
