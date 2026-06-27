# Cross-Analysis Battery — Paper Notes

**Date:** 2026-06-27
**Status:** 6 analyses complete (A1-A4, A6-A7), 1 cancelled (A5)
**Companion files:** cross_analysis_battery.md (results), cross_analysis_learnings.md (MMAR synthesis), a5_tjepa_status.md, prithvi_eo2_embeddings.md

---

## 1. Headline Narrative

The cross-analysis battery was designed to test generalizability of the s035
model ladder across scenarios, data sources, and feature sets. The dominant
finding is **flood regime heterogeneity** — not a methodology failure, but the
experiment's primary scientific contribution.

**Score: 3 FAIL, 1 PASS, 1 MIXED, 1 CANCELLED.**

The 3/4 initial FAIL rate (A1/A3/A4) reveals that flood damage prediction is
regime-specific, not geographically universal. A7 (NFIP ablation) confirms that
the single "universal" predictor (NFIP historical frequency) is actually a
metro-specific confounder that trades within-scenario accuracy for
transferability.

---

## 2. Per-Analysis Paper Notes

### A1: Prithvi Redundancy (FAIL)
- **Paper section:** Results, negative results subsection
- **One-liner:** Satellite foundation model embeddings are redundant with tabular features at ZCTA grain
- **Key numbers:** 0/5 scenarios improved; SW Florida Cohen's d = -2.06 (Prithvi hurts)
- **Scope bound:** Specific to mean-pooled embeddings at admin-unit grain. Patch-level or sub-ZCTA evaluations may differ.
- **Root causes (MMAR, 4/4 convergence):** granularity mismatch, information redundancy, static-for-dynamic, mean-pooling destroys structure
- **Cite:** Zhou et al. on spatial aggregation effects in GeoAI

### A3: Transfer Failure (FAIL)
- **Paper section:** Results, transferability subsection
- **One-liner:** Flood models don't transfer across US metros (2/20 positive pairs)
- **Key numbers:** Houston->NOLA 0.27, NYC->Houston 0.31 (only positives); Riverside as target: -23 to -44
- **Scope bound:** Transfer governed by flood regime similarity, not distance
- **Root causes (MMAR, 4/4 convergence):** covariate shift + concept shift compound
- **Directionality:** Riverside-as-target is catastrophic (-44); Riverside-as-source is merely bad (-1.4). Asymmetric.
- **Cite:** Koh et al. 2021 on hard environmental boundaries in domain adaptation

### A4: Importance Instability (FAIL)
- **Paper section:** Results, feature analysis subsection
- **One-liner:** Feature importance rankings are scenario-dependent; only NFIP freq is universal (and circular)
- **Key numbers:** tau range [-0.206, 0.323]; 0/10 pairs > 0.40; 1 universal feature
- **Scope bound:** The model ladder is 2+ distinct prediction problems wearing the same schema
- **Connection to A3:** tau-transfer correlation: coastal-only rho=0.71 (p=0.11, descriptive)
- **Cite:** Swayamdipta et al. 2020 (dataset cartography)

### A5: TJEPA Fusion (CANCELLED)
- **Paper section:** One sentence in methods ("pre-registered, cancelled by gate condition")
- **Reason:** Moot by transitivity (Prithvi redundant with R0; TJEPA is f(R0))
- **Design flaw found:** Dimensionality bug (1024+128 != 2048)
- **MMAR:** 4/4 unanimous CANCEL

### A6: Coverage Gap Overlap (PASS)
- **Paper section:** Data quality / limitations
- **One-liner:** Satellite and hydrology gaps overlap systematically in dense urban areas
- **Key numbers:** NYC Jaccard=0.75 (30 ZCTAs), Fisher OR=320.6; 3/5 significant
- **Scope bound:** Pattern is dense urban + flat terrain + cloud cover (not flat terrain alone; Houston contradicts)
- **Disclosure obligation:** Must characterize geographic distribution of missingness
- **Equity note:** Correlated missingness under-represents highest-exposure urban ZCTAs

### A7: NFIP Ablation (MIXED)
- **Paper section:** Results, circularity analysis + discussion
- **One-liner:** Removing NFIP costs ~10% within-R2 but doubles transfer and improves stability
- **Key numbers:**
  - Within-R2: mean delta -0.106 (Houston -0.292, NYC **+0.149**, Riverside 0.000)
  - Stability: mean tau 0.075 -> 0.109
  - Transfer: 2/20 -> 5/20 positive pairs; mean R2 -5.91 -> -1.98
- **NYC is the smoking gun:** Removing a "predictor" *improves* the model = confounding evidence
- **Riverside unchanged:** NFIP features had zero importance (arid, minimal NFIP market)
- **Paper framing:** Present full and ablated as paired analysis; NFIP is a "ceiling predictor"
- **Operational relevance:** For un-insured or newly-mapped areas, the NFIP-free model is what matters

---

## 3. Cross-Cutting Themes for Paper

### Theme 1: Regime Heterogeneity Is the Finding
- A1, A3, A4 all fail for the same root cause: different flood mechanisms (hurricane surge, riverine, flash flood) produce different feature-target mappings
- A6 fails for a different reason (sensor limitations), must be distinguished
- A7 confirms: NFIP history papers over regime differences within scenarios but creates metro-specific overfitting

### Theme 2: Riverside as Falsification Control
- Not a failure — it bounds the claims
- Only scenario where Prithvi helps (marginally), only scenario with zero NFIP importance, only scenario with orthogonal feature structure
- Without Riverside, the paper overstates generalizability
- With it: "findings apply to hurricane-driven coastal flood scenarios; arid inland requires different approaches"

### Theme 3: The Circularity Problem Is Empirically Resolved
- NFIP historical frequency: #1 predictor in 4/5 scenarios
- Confounds: physical exposure + building vulnerability + insurance market penetration
- A7 disambiguates: removing it improves NYC (confounder), doesn't affect Riverside (irrelevant), hurts Houston/NOLA (genuine signal there)
- The feature is valid but regime-specific and non-portable

### Theme 4: Honest Negative Results > Overclaimed Positives
- The 3/4 FAIL rate IS the contribution
- Each FAIL has clear scope boundaries and root causes
- Paper should not apologize for failures — should present them as findings that bound the claims

---

## 4. Paper-Ready Claim Inventory

Each claim traces to an S3 artifact:

| # | Claim | Artifact |
|---|-------|----------|
| 1 | Prithvi adds no predictive value at ZCTA grain (0/5 improve) | a1_prithvi_utility.json |
| 2 | Transfer positive in only 2/20 directional pairs | a3_transfer_matrix.json |
| 3 | Feature importance tau range [-0.21, 0.32], 0/10 > 0.40 | a4_feature_importance_stability.json |
| 4 | Coverage gaps overlap in 3/5 scenarios (NYC Jaccard=0.75) | a6_coverage_gap_overlap.json |
| 5 | NFIP ablation: mean -10.6% within-R2, 2/20->5/20 transfer | a7_nfip_ablation.json |
| 6 | NYC R2 improves +14.9% without NFIP (confounder removal) | a7_nfip_ablation.json |
| 7 | Importance stability improves without NFIP (tau 0.075->0.109) | a7_nfip_ablation.json |
| 8 | Tau-transfer correlation: coastal rho=0.71 (descriptive, p=0.11) | computed from A3+A4 |
| 9 | Prithvi CLS tokens are near-degenerate (cosine > 0.99) | prithvi_meta.json |
| 10 | Mean-pooled Prithvi cosine range 0.41-0.97 (genuine discrimination) | prithvi_meta.json |

All artifacts at `s3://swarm-floodrsct-data/results/s035/cross_analysis/`.

---

## 5. Figures and Tables Needed

### Tables
- **Table X:** Cross-analysis battery summary (A1-A7 results, one row each)
- **Table X+1:** A7 within-scenario R2 comparison (full vs ablated, 5 rows)
- **Table X+2:** Transfer matrix side-by-side (full vs ablated, 5x5 each)
- **Table X+3:** Top-5 features with and without NFIP (5 scenarios, 2 panels)

### Figures
- **Figure X:** A3 transfer heatmap (5x5, diverging colormap, positive pairs highlighted)
- **Figure X+1:** A4 importance profile comparison (radar or bar chart, 5 scenarios)
- **Figure X+2:** A7 transfer improvement visualization (side-by-side heatmaps or delta heatmap)

---

## 6. Remaining Work

### Computable now (no new SageMaker)
- [ ] Coverage gap impact quantification (R0 accuracy on gap vs non-gap ZCTAs)
- [ ] Generate paper figures from existing JSON artifacts

### Future work (section 9)
- [ ] Regime clustering (Wasserstein inter-scenario distances)
- [ ] Hierarchical model (shared base + scenario-specific heads)
- [ ] Patch-level Prithvi (attention-pooled, sub-ZCTA grain)
- [ ] Causal NFIP validation (instrumental variables)
- [ ] Multi-temporal Prithvi (pre/post-event difference)
- [ ] Transfer with domain adaptation

### Not started (from DOE)
- [ ] A2: kappa_reconstruct at R1/R2 levels (R0 was null because no spatial-lag features)
