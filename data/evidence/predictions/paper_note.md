# Paper Note: S018D-Posthoc

## Claim supported

Certificate-metric space has multi-dimensional unsupervised structure (3+
orthogonal pathology axes) but fails to distinguish degenerate solvers from
serious ones under aggregate metrics.

## Results

- PCA explains 100.0% variance in 5 components
- PC1 (63.9%): primarily sigma/scale axis
- Noisy solver nearest neighbor: **linear_ridge** (a serious solver)
- Control pair distance: 10.006 (> mean 9.630)
- Controls are NOT a coherent group

## Interpretation

Aggregate certificate metrics capture *some* structure (scale, instability)
but completely miss the noisy-solver pathology. A degenerate noise-generating
solver is indistinguishable from serious solvers when you only look at summary
statistics. This is GeoCert Failure Mode 1 (Scalar Projection) in action.

## Limitation

12 solvers is too few for robust PCA. The finding is directional, not
statistically definitive. S018Y-U with per-target ablation deltas provides
the higher-dimensional view needed.

## Recommended sentence for paper

> Post-hoc analysis of S018D certificate metrics reveals three orthogonal
> pathology axes but fails to separate the degenerate noisy solver from
> serious solvers---its nearest neighbor is linear_ridge
> (distance 3.34), not the other
> control (distance 10.01).
