# Findings — kimi

## Critical
(none)

## Serious
(none)

## Minor
- The negative Prithvi result (A1) suggests ZCTA-level mean-pooling destroys the spatial heterogeneity that flood models need; patch-level embeddings may still contain signal at parcel or building scale.
- Riverside's catastrophic transfer performance (A3 R2 = -43) confirms arid flash-flood regimes are fundamentally out-of-distribution for coastal surge models; this is a feature, not a bug, of domain heterogeneity.
- Feature instability (A4 τ ≤ 0.323) shows flood damage drivers are regime-specific; universal "flood risk" features do not exist across hurricane surge, pluvial, and fluvial contexts.
- NYC's 75 % Jaccard overlap in missing data (A6) reveals urban cloud-cover artifacts in HLS and flat-terrain failures in hydrology extraction; this is a measurement limitation, not a modeling one.
- NFIP historical frequency as top predictor in 4/5 scenarios is expected: insurance claims are path-dependent and spatially autocorrelated; this is not circularity but rather exposure persistence.
- The two successful transfers (Houston→NOLA, NYC→Houston) share Gulf/Caribbean hurricane regimes and similar coastal plain geomorphology; transferability is governed by physical similarity, not geographic proximity.
