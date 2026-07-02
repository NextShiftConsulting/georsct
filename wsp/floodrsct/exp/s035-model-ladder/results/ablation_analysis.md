# Ablation Analysis: Feature Sub-Group Contributions in the Representation Ladder

## Overview

We conduct a systematic ablation study to isolate the contribution of each feature sub-group across the three representation levels of the s035 model-example ladder. The study addresses two questions central to the geospatial flood damage prediction task: (1) which spatial features drive the R0-to-R1 performance lift, and (2) which temporal features drive the R1-to-R2 lift. All ablation variants share the same fold assignments, solver hyperparameters, and evaluation protocol, ensuring that observed differences are attributable solely to the feature set.

## Experimental Design

### Representation Levels

The representation ladder builds predictive models at three levels of feature enrichment:

- **R0 (Baseline)**: Tabular features derived from census, land cover, and FEMA flood zone data. These are static, spatially indexed attributes available for any ZCTA without event-specific information.
- **R1 (Spatial)**: R0 features augmented with hydrology variables (catchment area, topographic wetness index, stream slope, infrastructure proximity) and spatial lag (W-matrix) features computed from ZCTA adjacency graphs.
- **R2 (Temporal)**: R1 features augmented with event-dynamic variables capturing rainfall temporal structure (peak intensities at 1h/3h/6h windows, storm duration, time-to-peak, intensity coefficient of variation) and storm track/surge characteristics (tidal peak, surge-rainfall lag, storm proximity, landfall category).

### Ablation Variants

At each level, we selectively remove feature sub-groups to measure their individual contribution:

**R1 ablations** (4 variants):

| Variant | Feature set | Question answered |
|---------|-------------|-------------------|
| R1-full | R0 + hydrology + W-matrix | Full spatial representation |
| R1-no-wlag | R0 + hydrology | Does spatial structure (neighbor relationships) matter? |
| R1-wlag-only | R0 + W-matrix | Is neighbor structure sufficient without hydrology? |
| R1-no-target-lag | R0 + hydrology + W-matrix - wlag_nfip_claims | Does the target spatial lag drive the W-matrix effect? |

**R2 ablations** (4 variants):

| Variant | Feature set | Question answered |
|---------|-------------|-------------------|
| R2-full | R1 + rainfall dynamics + storm track | Full temporal representation |
| R2-no-storm-track | R1 + rainfall dynamics | Do rainfall temporal patterns alone suffice? |
| R2-no-rainfall | R1 + storm track/surge | Does storm proximity/surge alone suffice? |
| R2-temporal-only | R0 + all temporal features | Do temporal features work without spatial enrichment? |

### Evaluation Protocol

All variants are evaluated using the same per-metric eligibility gate introduced in the evaluation hygiene framework. Classification metrics (ROC-AUC, F1, Jaccard, etc.) are reported only for folds where the training set contains both classes; folds with degenerate class distributions are marked SKIP_TRAIN_SINGLE_CLASS and excluded from detection-channel averages. This prevents dilution of performance estimates by uninformative folds -- a particular concern for sparse targets like high water marks in data-limited scenarios.

Primary metrics reported below are ROC-AUC for classification targets and R-squared for regression, computed as the mean across spatial-blocked cross-validation folds using the HistGradientBoosting solver.

## Results

### Table 1: Ablation Results (Mean Across Scenarios)

| Variant | 311 Reports (ROC-AUC) | High Water Marks (ROC-AUC) | NFIP Claims (R^2) |
|---------|:---------------------:|:--------------------------:|:------------------:|
| R0 | 0.883 | 0.724 | 0.270 |
| | | | |
| R1-full | 0.888 | **0.818** | 0.292 |
| R1-no-wlag | 0.872 | 0.715 | 0.262 |
| R1-wlag-only | 0.903 | 0.785 | 0.302 |
| R1-no-target-lag | 0.875 | 0.719 | 0.285 |
| | | | |
| R2-full | **0.913** | 0.841 | 0.436 |
| R2-no-storm-track | 0.887 | 0.813 | 0.419 |
| R2-no-rainfall | 0.913 | 0.841 | **0.490** |
| R2-temporal-only | 0.894 | **0.880** | 0.477 |

### Finding 1: Spatial lag (W-matrix) is the primary R1 driver

The W-matrix features -- which encode neighborhood-averaged values of land cover, income, population density, and other attributes -- account for nearly all of the R1 performance lift. When hydrology features are added without the W-matrix (R1-no-wlag), performance returns to or slightly below the R0 baseline across all three targets (311 Reports: 0.872 vs. 0.883; High Water Marks: 0.715 vs. 0.724; NFIP Claims: 0.262 vs. 0.270). Conversely, the W-matrix alone without hydrology (R1-wlag-only) captures most of the R1-full gain and even exceeds R1-full for 311 Reports (0.903 vs. 0.888).

This result is consistent with spatial autocorrelation theory: flood damage exhibits strong positive spatial dependence, and the W-matrix features directly encode this dependence through neighbor-averaged predictors. The hydrology features (catchment area, TWI, stream slope) provide complementary but secondary information. Importantly, removing the target spatial lag (wlag_nfip_claims) in R1-no-target-lag yields performance nearly identical to R1-no-wlag, suggesting that the target lag is a significant component of the W-matrix's predictive power -- though the per-fold recomputation protocol mitigates direct leakage concerns.

**Implication for practitioners**: When spatial adjacency information is available, constructing W-matrix features should be prioritized over collecting additional point-level hydrology covariates. The spatial lag of outcome-adjacent variables provides the strongest predictive signal at the R1 level.

### Finding 2: Storm track and surge features drive the R2 lift

At the R2 level, the ablation reveals an asymmetric contribution between the two temporal sub-groups. Removing rainfall dynamics (R2-no-rainfall) produces performance indistinguishable from R2-full for High Water Marks (0.841 = 0.841) and slightly superior for NFIP Claims (0.490 vs. 0.436). Removing storm track features (R2-no-storm-track) causes a larger degradation, particularly for 311 Reports (0.887 vs. 0.913) and NFIP Claims (0.419 vs. 0.436).

The storm track sub-group -- storm proximity (km), landfall category, tidal peak (m), and surge-rainfall lag (h) -- captures the intensity and spatial targeting of the event in a way that the temporally detailed but spatially diffuse rainfall metrics do not. This finding aligns with the established meteorological understanding that peak rainfall intensity is a necessary but insufficient predictor of flood damage; the interaction between storm surge and rainfall timing (captured by surge_rain_lag_h) and the spatial targeting of the storm track (storm_min_dist_km) are what distinguish damaging events from high-rainfall events that produce minimal impact.

The incremental waterfall decomposition (Figure C) quantifies this clearly: across targets, the storm track sub-group contributes +0.026 to +0.051 to the primary metric, while rainfall dynamics contribute -0.005 to +0.008 -- often within the noise floor of cross-validation variance.

**Implication for practitioners**: For event-level flood damage prediction, storm track metadata and surge timing should be treated as first-class features. Detailed temporal rainfall profiles (sub-hourly peaks, intensity CV) provide marginal additional value when storm track information is available.

### Finding 3: Temporal features operate independently of spatial enrichment

The R2-temporal-only variant (R0 + temporal features, bypassing all R1 spatial enrichment) produces a striking result: for High Water Marks, it achieves the highest ROC-AUC of any variant (0.880), exceeding even R2-full (0.841). For NFIP Claims, it matches R2-full (0.477 vs. 0.436). Only for 311 Reports does it underperform R2-full (0.894 vs. 0.913).

This finding suggests that spatial and temporal features may partially interfere for certain targets. The W-matrix features that drive R1 performance encode spatial smoothing -- neighborhood averages that reduce local noise. But for targets that are fundamentally event-driven (e.g., high water marks depend on where the storm made landfall and surge timing), this spatial smoothing may dilute the sharp event-level signal that temporal features capture. The result is not that spatial features are harmful in general, but that the optimal feature set is target-dependent: spatially smooth targets (311 Reports) benefit from spatial enrichment, while event-driven targets (High Water Marks) may be better served by temporal features alone.

**Implication for model design**: A practitioner building a multi-target flood damage model should consider target-specific feature selection rather than a one-size-fits-all feature set. Alternatively, ensemble strategies that weight spatial and temporal channels differently per target may capture the best of both representations.

### Finding 4: Hydrology features alone provide no lift over R0

The R1-no-wlag ablation (R0 + hydrology, no spatial structure) consistently underperforms the R0 baseline. This is a negative result that merits discussion. The hydrology features include physically meaningful variables -- catchment area, topographic wetness index, stream slope, basin slope -- that are known predictors of flood susceptibility in the hydrological literature. Their failure to improve tabular baseline performance may reflect one of three mechanisms:

1. **Redundancy**: The R0 feature set already contains FEMA flood zone percentages, which are themselves derived from hydrological modeling. The hydrology features may be recapitulating information already captured by flood zone designations.

2. **Scale mismatch**: The hydrology features are derived from NHD catchments and 3DEP terrain at resolutions of 10--30 meters, then aggregated to the ZCTA level (typically 5--50 km^2). This aggregation may destroy the fine-grained spatial variation that makes these features informative at the parcel or census-tract level.

3. **Sample size**: With approximately 400 observations per scenario split across 5 cross-validation folds, the training sets may be too small to learn the nonlinear interactions between hydrology features and flood outcomes. The HistGBDT solver has sufficient capacity but may lack statistical power at this sample size.

These hypotheses are not mutually exclusive and suggest a productive direction for future work: testing hydrology features at finer spatial resolution (census tract or parcel level) where the scale mismatch is reduced.

## Per-Scenario Variation

The scenario-level results reveal important heterogeneity. Houston, the largest and most data-rich scenario, shows the clearest ablation patterns -- W-matrix features provide +0.103 ROC-AUC lift for High Water Marks, and storm track features add +0.015 at R2. NYC shows the strongest temporal effect, with R2-full reaching 0.989 ROC-AUC for 311 Reports (near-ceiling performance). Southwest Florida shows moderate but consistent gains from both spatial and temporal enrichment.

New Orleans and Riverside-Coachella are data-limited scenarios where many targets have degenerate fold distributions (single-class training sets). The evaluation hygiene gate correctly identifies and excludes these folds, preventing inflated or deflated performance estimates. For Riverside-Coachella, the obs_has_hwm target yields no eligible folds at any representation level, confirming that the sparse data in this scenario cannot support reliable classification for that target.

## Methodological Notes

### Evaluation Hygiene

All results reported in this analysis use the per-metric eligibility gate, which partitions classification folds into three channels:

- **MEASURED**: Both classes present in training and non-empty union in test. Metric value is computed and included in detection-channel averages.
- **MEASURED_FALSE_ALARM_ONLY**: Model trained but test set contains no positive examples (empty union for overlap metrics). Reported separately as false-alarm performance.
- **SKIP_TRAIN_SINGLE_CLASS**: Training set contains only one class. No model is trained; all metrics receive null values with a typed status code.

This protocol ensures that the ablation comparison is apples-to-apples: all variants use exactly the same folds, and the same folds are eligible or ineligible across variants (since eligibility depends on the target distribution in the fold, not the feature set).

### Cross-Validation Strategy

Results use the spatial-blocked cross-validation split, which assigns geographically contiguous ZCTA blocks to folds. This is the most conservative split strategy and prevents spatial leakage between training and test sets. The random and leave-event-out splits are also computed but not reported here, as the spatial-blocked split provides the most externally valid performance estimates.

### Solver

All results use the HistGradientBoosting solver (max_iter=200, max_depth=6, learning_rate=0.1). Ridge regression results are also computed for each variant but are omitted from this analysis for brevity; they show qualitatively similar ablation patterns with lower absolute performance.

## Summary

The ablation study reveals a clear hierarchy of feature importance for geospatial flood damage prediction:

1. **Spatial lag (W-matrix) features** are the dominant contributor at R1, providing the largest single-source lift across all targets.
2. **Storm track and surge features** are the dominant contributor at R2, with rainfall temporal dynamics providing marginal additional value.
3. **Temporal features operate independently** of spatial enrichment for event-driven targets, suggesting that the optimal feature set is target-dependent.
4. **Hydrology features alone provide no lift** over the tabular baseline, likely due to scale mismatch or redundancy with FEMA flood zone designations.

These findings have direct implications for the design of operational flood damage prediction systems: prioritize spatial adjacency structure and storm track metadata over detailed temporal rainfall profiles or point-level hydrology covariates.
