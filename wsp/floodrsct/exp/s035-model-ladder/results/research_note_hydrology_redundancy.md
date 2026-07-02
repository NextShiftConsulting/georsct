# Research Note: Hydrology Feature Redundancy with FEMA Flood Zone Designations

## Motivation

The s035 ablation study (Finding 4) shows that adding hydrology features (catchment area, TWI, stream slope, basin slope) to the R0 baseline without spatial lag structure (R1-no-wlag) produces performance at or below R0 across all three targets. This note investigates why, drawing on flood susceptibility literature and the structure of the FEMA National Flood Hazard Layer.

## Three Reinforcing Mechanisms

### 1. Information Overlap: FEMA Flood Zones Encode Hydrology

FEMA's National Flood Hazard Layer (NFHL) flood zone designations are the output of hydrological and hydraulic modeling pipelines. The standard FEMA workflow uses discharge-frequency analysis (HEC-HMS or equivalent) to estimate flood flows, then hydraulic models (HEC-RAS) to convert flows to water surface elevations, which determine the spatial extent of 100-year and 500-year flood zones. The R0 feature set includes FEMA flood zone percentages per ZCTA (e.g., fraction of area in Zone AE, Zone X, Zone VE), so these features are effectively a compressed summary of the same physical processes that the raw hydrology features describe.

When the R0 model already has access to FEMA flood zone percentages, adding raw catchment area, stream slope, and basin slope asks the model to learn what FEMA's hydrological engineers already computed -- but from noisier, less processed inputs.

### 2. TWI Multicollinearity Is Mathematically Inevitable

The Topographic Wetness Index is defined as:

    TWI = ln(A_s / tan(beta))

where A_s is specific catchment area and beta is local slope. When TWI, catchment area, and slope all appear as features, two of the three are mathematically redundant -- TWI is a deterministic function of the other two. The Variance Inflation Factor (VIF) diagnostic consistently flags this multicollinearity in flood susceptibility studies, with VIF > 10 as the standard removal threshold (consensus across Tehrany et al. 2019, Arabameri et al. 2020, and multiple 2023-2025 studies using VIF-based feature selection for flood susceptibility mapping).

While HistGradientBoosting is more tolerant of collinearity than linear models (tree splits on one feature are not degraded by correlation with another), the practical effect with ~400 training samples per scenario and 5-fold cross-validation is that the collinear features dilute the effective signal-to-noise ratio without contributing independent information. The solver's limited sample budget is spent learning redundant splits.

### 3. Scale Mismatch Destroys Discriminative Information

The hydrology features derive from NHD (National Hydrography Dataset) catchments and 3DEP (3D Elevation Program) terrain data at 10-30 meter resolution, then are aggregated to ZCTA-level summaries (typically 5-50 km^2 in our study areas). A single ZCTA may span multiple catchments with very different hydrological characteristics -- the mean TWI or catchment area of the ZCTA conflates high-risk and low-risk sub-regions.

FEMA flood zone percentages, by contrast, survive spatial aggregation more gracefully because they are already expressed as proportions (fraction of ZCTA area in each flood zone category). The information content of "30% of this ZCTA is in Zone AE" is higher than "mean TWI = 8.3" when predicting ZCTA-level flood damage outcomes.

## Supporting Evidence

### FloodGenome (Liu & Mostafavi, 2024-2025)

The FloodGenome framework applies machine learning to NFIP claims data across four US metropolitan areas (including Houston, which overlaps with our study). Key findings relevant to this analysis:

- **HAND** (Height Above Nearest Drainage) and **impervious surface percentage** are the top two predictors of property-level flood risk, consistently outranking raw catchment area and slope.
- HAND is conceptually related to TWI but defined differently: it measures the elevation difference between a point and the nearest stream channel, rather than a flow accumulation ratio. This difference makes HAND more directly interpretable as flood exposure.
- Raw catchment metrics (area, slope) rank lower in feature importance, consistent with our finding that these features provide no lift when FEMA zone information is already available.

### Implications

The FloodGenome results suggest that even at finer spatial resolution (parcel-level rather than ZCTA-level), the specific hydrology variables in our R1 feature set may not be the most informative choices. HAND would likely outperform raw catchment area and TWI, but requires DEM processing at parcel or census-tract resolution -- a scale at which the aggregation problem (Mechanism 3) is reduced.

## Connection to Ablation Findings

The three mechanisms are not mutually exclusive and likely compound:

| Mechanism | Effect on R1-no-wlag | Testable prediction |
|-----------|---------------------|---------------------|
| FEMA redundancy | Hydrology features add no new information beyond what flood zone percentages capture | Removing FEMA flood zone features from R0 should increase the marginal value of hydrology features |
| TWI collinearity | Three correlated features split limited sample budget across redundant splits | Replacing TWI+catchment+slope with a single PCA component or HAND should improve efficiency |
| Scale mismatch | ZCTA aggregation destroys within-ZCTA spatial variation | Running at census-tract resolution should increase hydrology feature importance |

## Directions for Future Work

1. **HAND as R1 feature**: Replace TWI with HAND (Height Above Nearest Drainage), which is both less collinear with catchment area and more directly interpretive as flood exposure. Requires DEM processing with channel network delineation.

2. **Finer spatial resolution**: Test hydrology features at census-tract or parcel level where the scale mismatch (Mechanism 3) is reduced. This would require rebuilding the observation unit from ZCTA to tract, which is a significant pipeline change.

3. **Ablation of FEMA features**: Run an R0 variant that excludes FEMA flood zone percentages, then test whether hydrology features provide lift in that reduced baseline. This would directly test Mechanism 1 (redundancy).

4. **VIF-based feature selection**: Compute VIF for the R1 hydrology feature set and apply the VIF < 10 threshold to prune collinear features before model training.

## References

### Academic

- Liu, Q. & Mostafavi, A. (2024-2025). FloodGenome: Flood risk assessment using machine learning on NFIP claims. Multiple publications.
- FEMA. National Flood Hazard Layer (NFHL) methodology. Based on discharge-frequency + hydraulic analysis (HEC-HMS/HEC-RAS).
- Beven, K. J. & Kirkby, M. J. (1979). A physically based, variable contributing area model of basin hydrology. Hydrological Sciences Bulletin, 24(1), 43-69. (Original TWI definition.)
- Tehrany, M. S., et al. (2019). Flood susceptibility mapping using VIF-based feature selection. Multiple journals.

### GAO Reports on FEMA Flood Map Effectiveness

- GAO-22-104079 (2021). FEMA Flood Maps: Better Planning and Analysis Needed to Address Current and Future Flood Hazards. Finds maps do not reflect best available climate science or pluvial (rainfall) flooding; mapping investments lower for socially vulnerable populations; FEMA has not assessed usefulness of non-regulatory products since 2016.
- GAO-21-578 (2021). National Flood Insurance Program: Congress Should Consider Updating the Mandatory Purchase Requirement. Documents that FEMA maps are outdated (land development changes), do not reflect climate change, and miss some flood types (heavy rainfall events).
- GAO-11-17 (2010). FEMA Flood Maps: Some Standards and Processes in Place to Promote Map Accuracy. After $1.2B invested in modernization, only 21% of population has maps meeting national data quality thresholds. Compliance metric (FBS) does not compare relative accuracy across maps.
- GAO-23-105977 (2023). Flood Insurance: FEMA's New Rate-Setting Methodology Improves Actuarial Soundness. Risk Rating 2.0 moves beyond binary zone designations to continuous property-level risk -- conceptually parallel to our model ladder approach.

### Relevance to Ablation Findings

The GAO findings directly support the experimental results:

1. Our R0 FEMA features (pct_zone_a/x/x500) encode the outdated binary zone designations that GAO documented as insufficient. Their redundancy with R1 hydrology features at the ZCTA level is consistent with both being derived from the same underlying hydrological modeling.
2. The GAO-documented gap in pluvial/rainfall flooding coverage explains why R2 storm track features (which capture event-specific dynamics) provide the largest marginal lift -- they encode exactly the information FEMA maps lack.
3. The 21% data quality figure (GAO-11-17) provides empirical backing for the scale mismatch hypothesis: most FEMA maps don't meet their own accuracy standards, and our ZCTA-level aggregation further degrades whatever signal exists.
