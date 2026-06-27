# Cross-Analysis Battery: Deep Learnings Synthesis

**Date:** 2026-06-26
**Source:** MMAR review (claude, deepseek-r1, kimi, nemotron, gpt-oss) + cross-critique synthesis
**Input:** DOE_cross_analysis.md v2.0 + full numerical results from A1/A3/A4/A6

## Executive Frame

The 3/4 FAIL rate is not a methodology failure. These are domain-bounded negative
results that reveal **flood regime heterogeneity** as the dominant force shaping
predictive modeling outcomes. The failures are the finding.

---

## 1. Root Cause Analysis

### 1.1 A1 — Why Prithvi Embeddings Don't Help

Four compounding causes identified across reviewers (high confidence — 4/4 reviewers converged):

1. **Granularity mismatch**: Prithvi's HLS patches (~30m resolution, mean-pooled to
   1024-dim per ZCTA) aggregate heterogeneous landscapes. A single ZCTA may contain
   floodplain, elevated ridge, and impervious surface — mean-pooling destroys exactly
   the spatial variation that matters for flood damage prediction.

2. **Information redundancy**: The static tabular features (flood_pct_zone_a/x/x500,
   elevation, slope, TWI, impervious_pct) already encode the floodplain geometry that
   Prithvi's land-cover signal captures. The satellite embeddings are a noisier version
   of what the model already has.

3. **Static embeddings for dynamic events**: Prithvi captures pre-event land cover, not
   event-specific conditions (water extent, damage patterns). The prediction target
   (NFIP claims) is driven by event dynamics that static imagery cannot see.

4. **Mean-pooling destroys spatial structure**: Patch-level attention or object-based
   embeddings might preserve flood-relevant spatial patterns (building footprints near
   channels, infrastructure density) that mean-pooling collapses.

**SW Florida's strong negative (Cohen's d = -2.06)**: Likely reflects vegetation
confounding — mangrove and subtropical vegetation signatures in Prithvi embeddings
correlate with coastal proximity but inversely with actual claim patterns in
hurricane-surge scenarios.

**Generalizable lesson**: Foundation model embeddings evaluated at coarse
administrative-unit grain with mean-pooling may systematically fail for tasks
where within-unit spatial heterogeneity drives the outcome. This echoes Zhou et al.'s
findings on spatial aggregation effects in GeoAI. The negative result is specific to
the evaluation protocol, not necessarily to Prithvi itself.

### 1.2 A3 — Why Transfer Fails

Two types of shift compound (high confidence — 4/4 reviewers converged):

1. **Covariate shift**: Feature distributions differ across metros. Flood zone
   percentages, elevation profiles, demographic composition, and coastal distance
   all have different ranges. A model trained on Houston's bayou geography sees
   feature values in Riverside that are completely out-of-distribution.

2. **Concept shift**: The feature-to-target mapping itself changes. In Houston,
   flood_pct_zone_a predicts claims because zone A floods. In Riverside, the same
   feature has a different relationship to claims because the flood mechanism
   (flash flood vs. storm surge) differs fundamentally.

**The 2 positive pairs** (Houston->NOLA 0.27, NYC->Houston 0.31) share:
- Gulf/Caribbean hurricane regimes
- Coastal plain geomorphology
- Similar flood zone structure (FEMA A/V/X distribution)
- Urban density patterns

**Riverside as catastrophic target** (R2 down to -43.54): The model extrapolates
lat/lon values outside the training range and applies inverted feature-target
mappings. Arid flash-flood mechanics have no analog in coastal surge training data.

**Directionality matters** (cross-critique insight): Riverside *as target* is
catastrophic (-43.54 from NOLA), but Riverside *as source* is merely bad (-1.447 to
Houston). Riverside lacks the patterns coastal models expect, but also fails to
export transferable patterns — the failure is asymmetric.

**Generalizable lesson**: Transferability in geospatial prediction is governed by
physical similarity of the generating process (flood regime), not geographic
proximity. This aligns with Koh et al. 2021 on "hard" environmental boundaries
in domain adaptation. The implication for deployment: transferability should be
predicted from hydroclimatic covariates (aridity index, drainage density, dominant
flood mechanism), not from distance.

### 1.3 A4 — Why Feature Importance Is Unstable

**Genuine domain heterogeneity**, not a modeling artifact (high confidence):

- Coastal metros (Houston, NOLA, NYC, SW Florida) share NFIP historical frequency
  as the top predictor, but disagree on everything else
- Riverside's top features (longitude, latitude, vacancy, home value) reflect a
  fundamentally different prediction problem: in arid regions, flood damage is
  driven by *where you are* (proximity to wash channels) and *what you have*
  (property characteristics), not by historical flood zone designation
- The 3 negative-tau pairs all involve cross-regime comparisons (NOLA-SWFl,
  NYC-Riverside, Riverside-SWFl)

**Free testable hypothesis** (claude insight): The Kendall tau matrix (A4) should
correlate with the transfer R2 matrix (A3). Scenarios with similar importance
rankings should transfer better. This is computable from existing artifacts — no
new experiment needed.

**Generalizable lesson**: The model ladder is really 2+ distinct prediction problems
wearing the same feature schema. "Universal flood features" is a category error —
flood risk drivers are regime-specific. This supports the dataset cartography
framework (Swayamdipta et al. 2020) applied to geospatial domains.

### 1.4 A6 — Why Coverage Gaps Overlap

**Sensor-limitation convergence** in urban environments (high confidence):

- Dense urban canyons (NYC) simultaneously defeat:
  - HLS satellite visibility (persistent cloud cover, tall-building shadows)
  - Terrain-based hydrology extraction (flat terrain, engineered drainage)
- This creates **correlated missingness** — not random — that biases toward
  under-representing the places with highest flood exposure

**Cross-critique correction**: Houston's 0% hydrology gaps contradict any claim
that flat terrain universally causes hydrology extraction failure. The NYC pattern
is specific to **dense urban + flat terrain + cloud cover** co-occurring. Houston's
bayou terrain provides sufficient drainage gradients despite low elevation.

**Generalizable lesson**: Data coverage gaps in geospatial ML are not random missing
data — they are systematic measurement limitations that correlate with the prediction
target's geographic distribution. Gap-aware modeling strategies (explicit missingness
indicators, imputation with uncertainty) are needed for honest uncertainty
quantification.

---

## 2. The Circularity Question

**NFIP historical frequency as #1 predictor in 4/5 scenarios.**

Reviewers split on interpretation:

- **Leakage concern** (deepseek-r1, nemotron): Past claims predicting future claims
  is circular. The model may be memorizing insurance market penetration and claims
  filing behavior, not learning flood physics. This limits utility for predicting
  novel events or under climate change.

- **Exposure persistence** (kimi): Insurance claims are path-dependent and spatially
  autocorrelated. Places that flood repeatedly continue to flood. NFIP history
  encodes persistent physical exposure, not just administrative history.

**Resolution**: Both are true simultaneously. NFIP history confounds three things:
1. Actual flood exposure (physical — valid predictor)
2. Building vulnerability and construction practices (structural — valid)
3. Insurance market penetration and claims-filing behavior (administrative — confound)

Temporal gating (using only pre-event historical data) prevents direct data leakage
but does not resolve the deeper confound. The paper should:
- Acknowledge the circularity explicitly
- Run the model with and without NFIP features to quantify dependence
- Frame NFIP history as a "ceiling predictor" — it sets the upper bound but
  conflates mechanism with measurement
- Note that for novel events in un-insured areas, NFIP history is unavailable,
  making the non-NFIP features the operationally relevant ones

---

## 3. The Riverside Question

**Riverside is the study's most valuable scenario precisely because it fails
everywhere** (unanimous across reviewers).

- A1: Only scenario where Prithvi *helps* (+0.156 delta, though not significant) —
  because Riverside lacks the dense tabular feature coverage that makes Prithvi
  redundant in coastal scenarios
- A3: Catastrophic as both source and target — natural OOD test case
- A4: Completely orthogonal importance structure — coordinates and housing
  characteristics, not flood zones or NFIP history
- A6: Minimal coverage gaps (1 hydro miss, 9 Prithvi miss) — arid clear-sky
  conditions favor satellite coverage

**Paper framing**: Riverside is not a failure of the methodology — it is the
**falsification control** that bounds the claims. Without it, the paper could
overstate generalizability. With it, the paper can honestly say: "These findings
apply to hurricane-driven coastal flood scenarios; arid inland flash-flood regimes
require different approaches."

---

## 4. Cross-Cutting Theme: Domain Heterogeneity Is the Finding

All three FAILs (A1, A3, A4) share a root cause: **flood regime heterogeneity
across US metropolitan areas**. This is not a bug in the experiment — it is the
experiment's primary contribution.

**Precise attribution** (cross-critique refinement):
- A1 (Prithvi), A3 (transfer), A4 (importance) fail due to **regime-specific
  prediction structure** — different flood mechanisms produce different
  feature-target mappings
- A6 (coverage) fails for a **different reason** — sensor-specific measurement
  limitations in urban environments, not regime heterogeneity per se

The paper should distinguish these two failure modes clearly.

---

## 5. Research Agenda (using existing infrastructure)

### Immediate (computable from existing artifacts, no new SageMaker jobs)

1. **Tau-transfer correlation test**: DONE (2026-06-26). Correlate the A4 Kendall
   tau matrix with the A3 transfer R2 matrix (bidirectional average).
   - All 10 pairs: Spearman rho=0.38, p=0.28 — not significant. Riverside's
     catastrophic transfer R2 values (-17 to -23 average) dominate the variance.
   - Coastal-only (4 metros, 6 pairs): Spearman rho=0.71, p=0.11 — strong
     positive trend but n=6 is too small for significance.
   - Interpretation: Within coastal flood regimes, scenarios with more similar
     feature importance rankings DO transfer better. But the relationship breaks
     down when arid inland (Riverside) is included. This is consistent with the
     regime heterogeneity finding — transferability is governed by flood regime
     similarity, and importance similarity is a proxy for regime similarity.
   - Paper-worthy as a descriptive finding with the coastal-only caveat.

2. **NFIP-ablated importance**: LAUNCHED as A7 (job s035-cross-a7-nfip-ablation-20260627-021411).
   Retrain R0 excluding nfip_historical_frequency and nfip_historical_severity.
   Tests: (a) within-scenario R2 delta, (b) importance stability change,
   (c) transfer matrix change.

3. **Coverage gap impact quantification**: Compare R0 performance on gap vs.
   non-gap ZCTAs within each scenario. If gap ZCTAs have systematically worse
   predictions, the coverage limitation has measurable impact on model fairness.

### Near-term (new SageMaker jobs, existing infrastructure)

4. **Regime clustering**: Compute inter-scenario feature distribution distances
   (Wasserstein on shared features) and cluster scenarios. Test whether within-cluster
   transfer works (it should: Houston-NOLA are predicted to cluster).

5. **Hierarchical model**: Train a shared base model on universal features
   (nfip_history, basic demographics) and scenario-specific heads on local features.
   Does this outperform the current single-model approach?

6. **Patch-level Prithvi**: Instead of mean-pooling, use attention-weighted or
   max-pooled Prithvi embeddings. Or: use Prithvi at sub-ZCTA grain (census tract
   or building footprint) and aggregate with a learned function.

### Longer-term (section 9 / future work)

7. **Causal validation of NFIP circularity**: Instrumental variable analysis using
   flood insurance reform dates as natural experiments. Or: synthetic control comparing
   newly-mapped flood zones (NFIP-naive) to historically-mapped zones.

8. **Multi-temporal Prithvi**: Pre/post-event satellite imagery difference to capture
   damage signal, not just static land cover.

9. **Transfer with domain adaptation**: Apply domain adaptation techniques
   (feature alignment, adversarial domain training) to the A3 transfer problem.
   The existing transfer matrix provides a clean baseline.

---

## 6. Paper Framing

**These results are not failures — they are findings that bound the claims.**

Suggested section structure:
- "Cross-analysis battery reveals that flood damage prediction models are
  regime-specific, not geographically universal"
- "Satellite foundation model embeddings are redundant with tabular features
  at administrative-unit grain — a negative result that bounds the utility of
  mean-pooled embeddings for this class of prediction tasks"
- "Feature importance heterogeneity across flood regimes implies that 'universal
  flood risk features' is a category error"
- "Systematic data coverage gaps in dense urban areas create correlated missingness
  that must be disclosed and addressed"

**The 3/4 FAIL rate is the contribution**, not the embarrassment. Honest negative
results with clear scope boundaries are more valuable than overclaimed positive results.

---

## 7. MMAR Process Notes

- 5 reviewers, 15 cross-critiques, 624s wall-clock
- Key intellectual disagreements: circularity interpretation (leakage vs persistence),
  severity of transfer failure (catastrophic vs expected), generalizability of
  NYC coverage gap pattern
- Cross-critiques caught: overgeneralization from NYC to "flat terrain" (Houston
  contradicts), imprecise directionality in transfer claims, counting discrepancy
  in universal feature threshold
- gpt-oss produced no substantive findings (empty response)
- Synthesis was empty (synthesizer failure) — manual synthesis performed
