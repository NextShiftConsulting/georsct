# DOE Amendment v1.3: FEMA Flood Zone Redundancy Experiments

**Date**: 2026-07-02
**Status**: LOCKED
**Triggered by**: Ablation Finding 4 -- hydrology features alone provide no lift over R0

## Motivation

The s035 ablation study (Finding 4 in `results/ablation_analysis.md`) revealed that R1-no-wlag (R0 + hydrology, no spatial structure) consistently underperforms the R0 baseline across all three targets. The research note (`results/research_note_hydrology_redundancy.md`) identifies three compounding mechanisms:

1. **FEMA redundancy**: R0 includes FEMA flood zone percentages derived from the same hydrological modeling that generates the R1 catchment-level features
2. **TWI multicollinearity**: TWI = ln(A_s / tan(beta)) is a deterministic function of catchment area and slope
3. **Scale mismatch**: 10-30m hydrology features aggregated to ZCTA level lose discriminative power

This amendment adds targeted ablation experiments to isolate and quantify the FEMA redundancy mechanism.

## New Experiments

### R0-Level Ablations (2 variants x 5 scenarios = 10 jobs)

| Variant | Feature set | Question |
|---------|-------------|----------|
| `R0-no-fema` | R0 minus flood_pct_zone_{a,x,x500} | How much does FEMA contribute to R0 baseline? |
| `R0-no-r0-hydro` | R0 minus elevation, slope, TWI, HAND, GFI, SPI (7 features) | Do R0-level terrain/hydrology features contribute? |

**R0_FEMA features removed** (3):
- `flood_pct_zone_a`
- `flood_pct_zone_x`
- `flood_pct_zone_x500`

**R0_HYDRO_TERRAIN features removed** (7):
- `elevation_m_msl`, `slope_mean_pct`, `twi_twi`
- `hand_mean_m`, `twi_mean`, `gfi_mean`, `spi_mean`

### R1-Level Ablations (2 variants x 5 scenarios = 10 jobs)

| Variant | Feature set | Question |
|---------|-------------|----------|
| `R1-no-fema` | (R0 - FEMA) + hydro + (W-matrix - wlag_flood_zone_pct) | Can R1 hydro/W-matrix compensate for FEMA? |
| `R1-hydro-no-fema` | (R0 - FEMA) + hydro only | Does R1 hydrology substitute for FEMA? |

Note: `R1-no-fema` also removes `wlag_flood_zone_pct` (the spatial lag of `flood_pct_zone_a`) to ensure complete removal of FEMA-derived information.

## Hypotheses

| ID | Hypothesis | Metric | Threshold |
|----|-----------|--------|-----------|
| H-FEMA-1 | Removing FEMA from R0 degrades performance | R0-no-fema < R0 | Delta > 0.02 ROC-AUC (mean across targets) |
| H-FEMA-2 | R1 hydro features partially substitute for FEMA | R1-hydro-no-fema > R0-no-fema | Delta > 0.01 ROC-AUC |
| H-FEMA-3 | Full R1 (with W-matrix) compensates for FEMA loss | R1-no-fema >= R0 | Within 0.01 ROC-AUC |
| H-FEMA-4 | R0 terrain/hydrology features are redundant with FEMA | R0-no-r0-hydro >= R0-no-fema | R0-no-r0-hydro performs equal or better |

## Decision Tree

```
IF H-FEMA-1 confirmed (FEMA contributes significantly):
  IF H-FEMA-3 confirmed (R1 compensates):
    -> FEMA and R1 hydro encode overlapping information
    -> Report as "FEMA flood zones are a compressed proxy for 
       hydrological modeling; either source suffices"
  ELSE:
    -> FEMA provides unique information not captured by R1
    -> Report as "FEMA flood zones contribute independent 
       predictive power beyond raw hydrology features"
ELSE (H-FEMA-1 rejected, FEMA contribution < threshold):
  -> FEMA features are near-redundant in the R0 feature set
  -> Investigate which R0 features absorb the signal (likely 
     nfip_historical_frequency/severity or land cover)
```

## Evaluation Protocol

Identical to base ablation study:
- Same fold assignments (loaded from existing folds)
- Spatial-blocked CV, HistGBDT solver (primary) + Ridge
- Per-metric eligibility gate (evaluation hygiene framework)
- Primary metrics: ROC-AUC (classification), R-squared (regression)

## S3 Output Keys

```
results/s035/r0_no_fema_{scenario}.json
results/s035/r0_no_r0_hydro_{scenario}.json
results/s035/r1_no_fema_{scenario}.json
results/s035/r1_hydro_no_fema_{scenario}.json
```

## DO NOT Constraints

1. Do NOT regenerate folds -- ablation variants load existing Phase 1 folds
2. Do NOT modify solver hyperparameters -- same HistGBDT/Ridge config as base
3. Do NOT add new features -- ablation is subtraction only
4. Do NOT modify the per-metric eligibility gate
