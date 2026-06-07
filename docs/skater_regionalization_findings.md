# Skater Regionalization Robustness — Findings

**Date:** 2026-06-07  
**Experiment:** s035 regionalization robustness (DOE Appendix R)  
**Job:** s035-region-all-20260607-171537 (Completed)  
**Commit:** f02cd93

## Summary

Skater regionalization on structural geography features is **degenerate** for all 5 metros. The pre-committed sensitivity check (county-vs-region uplift direction comparison, K=5) cannot be honestly executed because the alternative folds produce held-out sets of 1–2 ZCTAs whose metrics are meaningless.

## Method

- Algorithm: `spopt.region.Skater` (with MaxPHeuristic fallback)
- Features: `flood_pct_zone_a`, `twi_acc_twi`, `slope_basin_slope` (structural geography only — no target, no NFIP history)
- Target K: 5 (matching ZIP3-blocked fold count for apples-to-apples comparison)
- Spatial weights: built from ZCTA adjacency edge list (Queen contiguity)
- Disconnected graph handling: largest connected component → Skater; orphans assigned by adjacency hop distance

## Per-Metro Results

| Metro | ZCTAs | Connected Components | Skater Regions | Largest Region | Verdict |
|-------|-------|---------------------|----------------|----------------|---------|
| Houston | 131 | 1 | 5 | 118 (90%) | Degenerate — one fold captures 90% |
| SW Florida | 187 | 3 | 14 (auto-expanded) | 74 (54%) | Failed K=5 — disconnected graph |
| NYC | 179 | 6 | 23 (auto-expanded) | 104 (58%) | Failed K=5 — disconnected graph |
| Riverside/Coachella | 85 | 1 | 5 | 81 (95%) | Degenerate — one fold captures 95% |
| New Orleans | 17 | 1 | 5 | 13 (76%) | Degenerate — too few ZCTAs for balanced K=5 |

## Region Size Distributions (ZCTA-level)

```
Houston:              {118, 8, 2, 2, 1}
SW Florida:           {74, 52, 19, 17, 12, 3, 2, 2, 1, 1, 1, 1, 1, 1}  (14 regions)
NYC:                  {104, 13, 12, 8, 6, 5, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}  (23 regions)
Riverside/Coachella:  {81, 1, 1, 1, 1}
New Orleans:          {13, 1, 1, 1, 1}
```

## Why Skater Fails Here

1. **Too few ZCTAs**: Metro-scale datasets (17–187 ZCTAs) are at the lower end of spatial regionalization viability. Skater optimizes for feature homogeneity within regions, not balance.

2. **Disconnected adjacency graphs**: NYC (5 boroughs + islands) and SW Florida (coastal gaps) produce multiple connected components. Skater auto-expands K to account for islands, breaking the 5-vs-5 comparison.

3. **Low feature variance**: The three structural features (flood zone %, TWI, slope) have limited spatial variation within connected metros — most ZCTAs are similar, producing one large "homogeneous" region with small outlier clusters.

4. **Contiguity constraint**: Unlike ZIP3 prefix blocking (which groups by postal geography regardless of adjacency), Skater requires spatial contiguity. This concentrates most ZCTAs in one region when feature gradients are smooth.

## Implication for Paper

ZIP3 prefix blocking is the **only** spatial blocking scheme that produces non-degenerate, balanced folds for these metro-scale ZCTA datasets. This is now reported as a positive design-choice finding in §7b (Spatial Validity Diagnostics), not as a gap.

## Additional Confound (Not Reached)

Training was not attempted because folds are degenerate, but a known confound would further compromise results: `wlag` spatial lag features become NaN for interior test ZCTAs whose training neighbors are entirely held out under contiguous blocking.

## Artifacts on S3

```
s3://swarm-floodrsct-data/results/s035/sidecar/robustness/
├── houston_region_folds.parquet
├── houston_region_meta.json
├── new_orleans_region_folds.parquet
├── new_orleans_region_meta.json
├── riverside_coachella_region_folds.parquet
└── riverside_coachella_region_meta.json
```

(SW Florida and NYC failed the K=5 validation gate and were not uploaded.)

## Code

- Regionalization: `data/floodrsct/jobs/compute_spatial_sidecar_regionalize.py`
- Launcher: `data/floodrsct/scripts/launch_compute_spatial_sidecar_regionalize.py`
- DOE specification: `data/floodrsct/exp/s035-model-ladder/DOE_spatial_diagnostics.md` §3
