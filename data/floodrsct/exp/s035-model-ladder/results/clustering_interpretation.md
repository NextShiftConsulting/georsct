# Why Context Increases Clustering: Target-Lag Propagation Mechanism

**Status:** Interpretation (descriptive). Does not alter the INSUFFICIENT verdict.
**Scope:** Paper appendix section explaining the monotonic Moran's I increase
Houston R0=0.207, R1=0.246, R2=0.489 and LISA significant-cluster fractions
6.8%, 8.3%, 15.9%.

---

## 1. The Observation

The representation ladder was designed so that each successive level (R0 tabular,
R1 hydrology+W-matrix, R2 temporal) provides richer spatial context. The naive
expectation is that richer context RESOLVES spatial error structure: Moran's I on
residuals should decrease as the model captures the spatial process.

Houston NFIP shows the opposite:

| Level | Moran's I (residuals) | LISA significant fraction |
|-------|----------------------|--------------------------|
| R0    | 0.207                | 6.8%                     |
| R1    | 0.246                | 8.3%                     |
| R2    | 0.489                | 15.9%                    |

Each context upgrade INCREASES residual spatial autocorrelation. The W-matrix
features EXPOSE rather than ABSORB spatial structure.

---

## 2. Mechanism: Target-Lag Propagation

### 2.1 Construction of wlag_nfip_claims

The key R1 feature `wlag_nfip_claims` is computed as:

```
wlag_nfip_claims_i = sum_j(w_ij * y_j)   for j in training fold
```

where `w_ij` is the row-standardized Queen contiguity weight and `y_j` is the
NFIP event claim count at ZCTA j. This is the spatial lag of the TARGET variable,
computed per-fold on training observations only (no cross-fold leakage).

### 2.2 Why spatial claims cluster

NFIP claims are not spatially random. They concentrate along:
- Riverine flood corridors (Buffalo Bayou, Brays Bayou, White Oak Bayou)
- Low-income neighborhoods with aging drainage infrastructure
- Contiguous low-elevation zones within the same FEMA flood map polygon

The spatial lag of a spatially-clustered variable is itself spatially smooth and
spatially autocorrelated -- it is a kernel-smoothed version of the target.

### 2.3 The propagation chain

When the model receives `wlag_nfip_claims` as an input feature:

1. **Learning phase:** The gradient-boosted trees learn that high neighbor claims
   predict high local claims. This is a strong signal (the DOE ablation confirms
   that target-lag drives ALL R0-to-R1 uplift; structural features alone perform
   at or below R0).

2. **Prediction phase:** The model's predictions become spatially smoother because
   they partially inherit the spatial structure of wlag_nfip_claims. Predictions
   in flood corridors are pulled upward; predictions in low-risk areas are pulled
   downward.

3. **Residual phase:** Where the local spatial lag DIVERGES from local reality --
   for example, a ZCTA with high neighbor claims but low own claims (levee-
   protected pocket), or vice versa -- the model commits a spatially-structured
   error. These errors cluster because the input feature (spatial lag) is itself
   spatially smooth.

4. **Moran's I increases:** The residual field inherits autocorrelation from the
   mismatch between the smooth spatial-lag input and the discontinuous local
   reality. The model is MORE wrong in a MORE spatially-structured way.

### 2.4 The R2 amplification

R2 adds temporal context (event-sequence features, lagged event indicators). These
features further sharpen the model's reliance on RECENT neighbor activity patterns.
The temporal features identify which flood events are "similar" to the current one,
effectively upweighting the most relevant historical neighbor values. This double
conditioning (spatial lag + temporal relevance) produces even more spatially-
coherent predictions -- and thus even more spatially-structured residuals where
the pattern breaks.

---

## 3. Evidence: Houston Ablation

The DOE-mandated ablation battery (DOE_R1_spatial.md) specifies four variants:

| Variant        | Features                           | Result interpretation          |
|----------------|------------------------------------|--------------------------------|
| full           | R0 + hydro + W-matrix              | Baseline R1                    |
| no-wlag        | R0 + hydro (no spatial structure)  | Uplift from point features     |
| no-target-lag  | R0 + hydro + W minus wlag_nfip     | Target lag contribution        |
| wlag-only      | R0 + W-matrix (no hydro)           | Spatial structure sufficiency   |

The Relational SME finding (confirmed in DOE_R1_spatial.md Kill Rules):

> "wlag_nfip_claims ablation shows ALL uplift from target lag -- report honestly;
> R1 story is 'neighbor claims predict' not 'hydrology helps'"

Structural features (hydrology, infrastructure, catchment area) do NOT improve
on R0. The entire R0-to-R1 uplift concentrates in the target spatial lag. This
confirms the propagation mechanism: the model's improvement comes from learning
the spatial autocorrelation pattern of the target itself, not from learning the
physical flood process.

---

## 4. Implication for the Benchmark

### 4.1 The benchmark detects amplification

The INSUFFICIENT verdict on the clustering geometry is correct and informative.
The spatial diagnostics framework (LISA + Moran's I) detects that spatial features
can AMPLIFY rather than RESOLVE error structure. This is precisely the kind of
failure mode that a geometric compatibility assessment should flag:

- A model that PASSES on prediction accuracy (R1 > R0 in fold-level R-squared)
- But FAILS on spatial structure (residuals are MORE clustered, not less)

The benchmark separates these dimensions. Prediction uplift (the H2a Wilcoxon)
measures whether the model is better on average. Clustering assessment (Moran's I
on residuals) measures whether the improvement is spatially uniform or whether it
introduces new structured failures.

### 4.2 Spatial features expose the problem, they don't solve it

The tex scaffold (results_spatial_diagnostics.tex) originally framed the narrative
as "R1's spatial features attenuate the clusters." The data says the opposite.
The honest framing:

> Spatial features (specifically, target spatial lag) improve aggregate prediction
> but concentrate the remaining error into spatially-coherent structures. The model
> trades uniformly-distributed small errors for spatially-clustered larger errors
> in zones where the spatial lag signal breaks down.

This is analogous to the bias-variance tradeoff in spatial form: the spatial lag
reduces variance (predictions are smoother, closer to the spatial mean) but
introduces structured bias (predictions are wrong in predictable spatial patterns).

---

## 5. Connection to Relational Geometry

The relational geometry assesses whether the W-matrix specification captures the
true spatial dependence structure. The same mechanism explains BOTH:

1. **Relational PROVISIONAL verdict:** The W-matrix features (specifically
   wlag_nfip_claims) drive uplift, but the uplift is entirely attributable to
   target-lag rather than structural topology. Queen contiguity is a coarse proxy
   for the actual flood propagation network (drainage basins, bayou corridors,
   storm-sewer connections). The relational structure is provisionally useful but
   not causally aligned.

2. **Clustering INSUFFICIENT verdict:** The target lag propagates spatial
   autocorrelation INTO the predictions. Where the Queen contiguity neighborhood
   diverges from the actual flood dependency structure (e.g., neighborhoods
   separated by a levee, or upstream/downstream relationships that cross non-
   contiguous ZCTAs), the model commits spatially-coherent errors.

The shared root cause: **Queen contiguity is not the flood network.** The target
lag computed on Queen contiguity is a spatially-smoothed version of the target that
captures PROXIMITY but not FLOW. Errors cluster where flow-based relationships
dominate proximity-based relationships.

---

## 6. GWR Non-Stationarity Connection

The GWR diagnostic (DOE_spatial_diagnostics.md Section 2) shows that the
feature-target relationship is non-stationary: local R-squared varies substantially
across Houston ZCTAs, and AICc improvement over global OLS confirms that
coefficients vary in space.

This non-stationarity is the STRUCTURAL explanation for why a global spatial lag
fails locally. The target lag uses a single W-matrix (Queen contiguity) applied
uniformly. But the actual relationship between neighbor claims and local claims
varies by geography:

- In riverine corridors: neighbor claims strongly predict local claims (water
  flows downstream through contiguous ZCTAs)
- In pluvial zones: local flooding depends on micro-topography, not neighbors
- Near levees: spatial lag systematically over-predicts (neighbors flood, but the
  levee protects the target ZCTA)

Where GWR identifies low local R-squared, the global spatial lag provides a
misleading signal. These are the same zones where LISA identifies residual
clusters -- the model's spatially-structured failures concentrate where the
global relationship assumption fails.

---

## 7. Framework Extension: S-Dimension Amplification

This finding extends the RSN (Relevance-Stability-Novelty) framework's
understanding of the S (Stability) dimension in spatial contexts:

**Standard expectation:** Adding spatial context (S-dimension features) should
stabilize predictions by capturing the data-generating process's spatial
structure, reducing residual autocorrelation.

**Observed behavior:** When the S-dimension is implemented as TARGET SPATIAL LAG
rather than as STRUCTURAL TOPOLOGY (drainage networks, flow accumulation paths,
physical connectivity), it can AMPLIFY spatial error structure.

**Diagnostic implication:** The S-dimension must distinguish between:
- **Topological S-features** (zcta_degree, drainage connectivity, catchment
  membership): these describe the spatial STRUCTURE independent of the outcome
- **Target-derived S-features** (wlag_nfip_claims, spatial lag of residuals):
  these describe the spatial PATTERN of the outcome itself

Only topological features can reduce residual autocorrelation without risking
amplification. Target-derived features improve prediction but propagate (and
potentially amplify) the spatial structure they encode.

---

## 8. Honest Summary

The monotonic Moran's I increase (0.207 -> 0.246 -> 0.489) is not a failure of the
model or the experiment. It is a DETECTION by the benchmark of a well-understood
phenomenon: spatial lag features propagate spatial structure into predictions and
thus into residuals. The benchmark correctly classifies this as INSUFFICIENT on the
clustering geometry while acknowledging that prediction accuracy improves (the
model is better on average, just more spatially-structured in its errors).

The path to resolving INSUFFICIENT is not more spatial lag features. It requires
either:
1. Replacing Queen contiguity with a flow-aligned W-matrix (drainage network
   topology) so that the spatial lag matches the physical process
2. Using spatially-varying coefficients (GWR-style) rather than a global spatial
   lag, so that the spatial relationship is allowed to be non-stationary
3. Explicitly modeling the residual spatial process (spatial error model or
   spatial lag model in the econometric sense) rather than treating spatial lag
   as a feature

None of these are in scope for the current experiment. The INSUFFICIENT verdict
stands, honestly documented, with a clear mechanistic explanation.
