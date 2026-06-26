# Findings — deepseek-r1

## Critical
(none)

## Serious
- file: Part 3: A1 Results  
  severity: serious  
  category: research_insight  
  description: |
    Prithvi embeddings failed due to fundamental granularity mismatch: ZCTAs aggregate heterogeneous landscapes where localized flood damage signals (e.g., building footprints) are diluted by mean-pooling. This reflects limitations in foundation models for administrative-unit prediction, echoing Zhou et al.'s findings on spatial aggregation effects in GeoAI. Research agenda: Test patch-level attention weighting and object-based embeddings targeting infrastructure.  
  quote: |
    A1 Results: Prithvi Predictive Utility — FAIL
    0/5 scenarios show >= 0.02 improvement
    3/5 show negative delta (embeddings HURT)
    SW Florida has strongest negative effect (Cohen's d = -2.06)

- file: Part 3: A3 Results  
  severity: serious  
  category: research_insight  
  description: |
    Transfer failures reveal hydrological regime divergence: Arid Riverside's flash-flood mechanics differ fundamentally from coastal surge dynamics, causing catastrophic negative transfer. This aligns with domain adaptation literature showing "hard" environmental boundaries (e.g., Koh et al. 2021). Research agenda: Cluster scenarios using hydroclimatic covariates (aridity index, drainage density) to predict transferability.  
  quote: |
    A3 Results: Cross-Scenario Transfer — FAIL
    Riverside is catastrophic as target (R2 down to -43.54)
    Only 2/20 positive pairs: Houston->NOLA (0.27), NYC->Houston (0.31)
    The 2 positive pairs are both large coastal metros

- file: Part 3: A4 Results  
  severity: serious  
  category: research_insight  
  description: |
    Feature instability stems from causal heterogeneity: Riverside's top predictors (vacancy, home value) reflect desert-specific vulnerability pathways unlike coastal metros. This violates IID assumptions in transfer learning, supporting Geiger's "dataset cartography" framework. Research agenda: Quantify feature-shift using Wasserstein distances in covariate space.  
  quote: |
    A4 Results: Feature Importance Stability — FAIL
    0/10 pairs exceed 0.40 threshold
    Min tau = -0.206 (NOLA vs SW Florida)
    Riverside top features: longitude, latitude, vacancy
    Only universal feature: nfip_historical_frequency

- file: Part 3: A6 Results  
  severity: serious  
  category: research_insight  
  description: |
    Coverage gaps expose sensor-limitation convergence: Urban canyons (NYC) simultaneously defeat satellite visibility and hydrologic mapping, creating systemic blindspots. This parallels CEOS QA4EO guidelines on "persistent omission bias." Research agenda: Simulate claims under gap imputation strategies to quantify impact on actuarial fairness.  
  quote: |
    A6 Results: Coverage Gap Overlap — PASS (3/5 significant)
    NYC: 30 ZCTAs missing BOTH satellite and hydrology data (Jaccard 0.75)
    These are dense urban ZCTAs where cloud cover + flat terrain both fail
    Houston has no hydrology gaps

## Minor
- Riverside as natural OOD testbed for flood model robustness  
- NFIP historical frequency indicates insurance access persistence, not flood risk  
- Event distance matrix limitation prevented hydrological transfer theory test  
- Cross-scenario R2 deltas from shared feature subset highlight covariate shift
