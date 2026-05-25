# Paper Outline — The Certification Gap

**Title:** The Certification Gap: When Leaderboards Rank the Wrong Models
**Subtitle:** A Benchmark and Audit Protocol for Structure-Preserving Evaluation
**Track:** NeurIPS 2026 Datasets & Benchmarks
**Author:** Rudy Martin (Next Shift Consulting LLC)
**Abstract deadline:** May 4, 2026 AOE
**Full paper deadline:** May 6, 2026 AOE

---

## What Actually Exists

| Asset | Status | Location |
|---|---|---|
| RSCT patent 19/575,615 | Filed March 23, 2026 | USPTO |
| `yrsn-train` monorepo with issuer/mixer/auditor | Working code | `apps/geo_cert/certificates/` |
| N-ceiling estimator with block bootstrap | Working code | `apps/geo_cert/models/ceiling/` |
| 16 embedding models with full certificates | Complete | `apps/train_rsn_head/paper_data/` |
| `geo_cert` pipeline — PCA32, spatial-lag, GNN | Trained, on S3 | `s3://yrsn-checkpoints/geo-cert/` |
| `geo_dual_graph_rsct` migration (85 tests pass) | Complete | `apps/geo_cert/` |
| S016C tree classifier for text retrieval R/S/N | Trained | `yrsn-checkpoints` |
| 4 existing figures (substrate, ladder, features, pooling) | Done | `apps/train_rsn_head/paper_data/` |
| CONUS-27 — 3 families x 27 tasks with R² metrics | On S3 (2026-04-20) | `s3://yrsn-checkpoints/geo-cert/` |

## Key Empirical Results (Measured)

**Text retrieval (16 models):**
- 56% of models have >= 3-position rank inversion (accuracy vs alpha)
- Spearman rho(accuracy rank, alpha rank) = 0.312
- nova_embed: accuracy #1 -> alpha #15 (14-position inversion)
- titan_v2: accuracy #15 -> alpha #4 (11-position inversion)
- openai_3_small: accuracy #6 -> alpha #1 (5-position inversion)
- ALL 16 models fail Gate 3 with RE_ENCODE decision

**Geospatial (3 families x 27 CONUS tasks):**
- Mean N_ceiling = 0.328, range 0.156 (night_lights) to 0.593 (cholesterol_screening)
- Cross-family R² spread is small (~0.037) — representation choice matters less than task noise floor
- Task difficulty varies 4x while model family ranking is nearly constant

---

## Section Outline

### S1. Introduction (1.0 pages)

**Opening:** "A model that tops a leaderboard is not always the model we should trust. Across 16 embedding models evaluated on MIRACL retrieval, 56% exhibit rank inversions of three or more positions when re-evaluated with a structural certificate — and the rank-order correlation between accuracy and certificate quality is 0.31."

**The problem in plain English.** Text embedding leaderboards rank models by a single scalar. That scalar conflates three different things: whether the model captured task-relevant structure, whether it failed to exploit structure it could have used, and whether it got noise that nothing could have recovered.

**Four contributions:**
1. The R/S/N decomposition: error split into representation adequacy (R), recoverable shortfall (S), and irreducible noise (N), on the simplex R+S+N=1.
2. A simplex audit protocol with four formal properties (integrity, S-monotonicity, N-invariance, alpha-consistency).
3. A block-bootstrap N-ceiling estimator with seed-collapse diagnostic.
4. Certificates for 16 text embedding models + 27 geospatial tasks, released with code for reproducibility.

### S2. Related Work (0.75 pages)

- Bias-variance decomposition
- Aleatoric vs epistemic uncertainty
- Bootstrap CIs in ML evaluation
- MTEB and BEIR retrieval benchmarks
- Trustworthiness evaluations (DecodingTrust as precedent for going beyond scalar metrics)

### S3. The R/S/N Decomposition (1.25 pages)

Formal definition. Simplex constraint R+S+N=1. The certificate triple (alpha, kappa, sigma):
- alpha = signal quality (from DecompositionScore)
- kappa = R*(1-N) = compatibility
- sigma = representational turbulence (0.4*sigma_var + 0.3*sigma_growth + 0.3*sigma_entropy)
- The oobleck threshold: kappa_req = 0.5 + 0.4*sigma (adaptive gate)

### S4. Certification and Audit (1.0 pages)

Issuer -> mixer -> auditor pipeline. Four audit properties:
1. Simplex integrity (R+S+N=1)
2. S-monotonicity (better representation -> lower S)
3. N-invariance (noise floor stable across model families)
4. Alpha-consistency (certificate agrees with per-class analysis)

Gate logic: PASS / CAUTION / REFUSE with REFUSE_KAPPA_MAX = 0.35.

### S5. The N-Ceiling Estimator (1.25 pages)

Block-bootstrap algorithm. Seed-collapse test. Residual-correlation correction.

Validation: synthetic dataset with known ground-truth noise level, showing block-bootstrap recovers it within CI while naive bootstrap underestimates.

### S6. Empirical Study: 16 Embedding Models (1.75 pages)

The primary empirical surface. Data source: `apps/train_rsn_head/paper_data/leaderboard.json`.

**Models:** jina (3 variants), nemotron (3 variants), titan_v2, minilm, voyage4 (2 variants), openai_3_small, gemini, nova_embed, cohere_v4_bedrock, bge_m3, qwen3_8b.

**Experiments:**
- Per-model R/S/N certificates
- Rank inversion scatter: accuracy rank vs alpha rank (rho=0.31)
- Confusion matrix analysis: nova_embed vs openai_3_small (why #1 accuracy has worst certificate)
- Gate audit: all 16 fail Gate 3, oobleck threshold analysis
- Simplex geometry: where models cluster in R/S/N space

**Headline claim:** Models that score similarly on scalar metrics sit in different regions of the R/S/N simplex. Scalar parity hides substantive differences in HOW models succeed.

### S7. Cross-Domain Validation: Geospatial (1.0 pages)

Same R/S/N code, completely different domain.

**Setup:** 27 CDC health outcome variables predicted from ACS census features across ~25K ZCTAs. 3 model families: PCA (linear), Spatial Lag (spatial econometric), GNN (graph neural).

**Results:**
- N_ceiling spectrum: 0.156 (night_lights) to 0.593 (cholesterol_screening)
- Cross-family spread ~0.037 R² — small vs 4x task difficulty variation
- Policy implication: for cholesterol_screening (N_ceil=0.59), no model improvement can recover more than 41% of variance — the right response is better data, not a better model

**Why this matters:** The decomposition works identically on text retrieval and geospatial regression. The audit protocol is domain-agnostic.

### S8. Limitations (0.5 pages)

- R/S/N requires OOF predictions — not applicable to black-box APIs without modification
- N-ceiling assumes iid-blockable residuals — degrades under strong non-stationarity
- Current text results use a single benchmark (MIRACL); more benchmarks strengthen the claim
- Patent disclosure footnote (19/575,615)

### S9. Conclusion (0.25 pages)

Accuracy ranks models. The certificate audits them. The two disagree 56% of the time. Until evaluation protocols can distinguish a model that succeeds from a model that merely passes, leaderboards will keep recommending models that fail in deployment.

---

## Page Budget

S1(1.0) + S2(0.75) + S3(1.25) + S4(1.0) + S5(1.25) + S6(1.75) + S7(1.0) + S8(0.5) + S9(0.25) = **8.75 pages** + figures/tables within budget.

---

## Figures and Tables

| # | Content | Data Source | Status |
|---|---------|-------------|--------|
| Fig 1 | Accuracy rank vs alpha-rank scatter (16 models) | leaderboard.json | DATA READY |
| Fig 2 | R/S/N simplex with 16 models plotted | leaderboard.json | DATA READY |
| Fig 3 | nova_embed vs openai_3_small confusion matrices | leaderboard.json | DATA READY |
| Fig 4 | N_ceiling heatmap across 27 CONUS tasks | geo_cert S3 data | DATA READY |
| Fig 5 | Layer 1 substrate comparison | existing | DONE |
| Fig 6 | Layer 2 ladder | existing | DONE |
| Fig 7 | Layer 3 features | existing | DONE |
| Fig 8 | Pooling comparison | existing | DONE |
| Tbl 1 | Full certificate leaderboard (16 models) | leaderboard.json | DATA READY |
| Tbl 2 | CONUS-27 per-task N_ceiling + best family | geo_cert S3 data | DATA READY |

---

## Discipline Notes

- All certificates come from real YRSN code (`yrsn.DecompositionScore`, `SequentialGatekeeper`)
- No hand-rolled formulas — every metric traces to a yrsn import
- All code and OOF predictions released for reproducibility
- The "certification gap" is an empirical finding (rho=0.31, 56% inversion), not a coined term
