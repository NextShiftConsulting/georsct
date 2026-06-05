# Paper Outline v2 (merged) — Two-Findings Structure

## Title

**The Certification Gap: When Leaderboards Rank the Wrong Models**

*Subtitle (abstract only):* A certified evaluation study across text retrieval and geospatial regression.

*Fallback:* The Certification Gap and the N-Ceiling: Certified Evaluation Across Text Retrieval and Geospatial Regression

**Author:** Rudy Martin (sole)
**Track:** NeurIPS 2026 Evaluations & Datasets
**Abstract deadline:** May 4, 2026 AOE
**Full paper deadline:** May 6, 2026 AOE

---

## Thesis (one paragraph)

A model that tops a leaderboard is not always the model we should trust — and sometimes the model barely matters at all. We contribute a certification protocol that decomposes predictive error into three sources (relevance R, recoverable shortfall S, irreducible noise N), audits the decomposition on the simplex R+S+N=1, and issues PASS/CAUTION/REFUSE certificates. Applied to 16 text embedding models on retrieval, the protocol reveals a **certification gap**: rank-correlation rho = 0.31 between accuracy and certificate quality, with 56% of models inverting by three or more positions. Applied to 27 ZCTA-level geospatial regression tasks across three model families, the same protocol reveals a complementary finding: architectural choice is nearly invisible (spread ~0.037 R-squared) against the dominant axis of task intrinsic noise (N-ceiling range 0.155--0.593). Together, the two findings identify when evaluation needs to discriminate models and when it needs to characterize tasks.

---

## Core paper promise

Scalar metrics can fail in two different ways: in text, they mis-rank models; in geo, they hide that task noise matters more than architecture. We give a protocol that tells you which regime you are in.

---

## Four contributions

1. **The R/S/N decomposition and certification protocol** — formal simplex decomposition, four-property audit (integrity, S-monotonicity, N-invariance, alpha-consistency), PASS/CAUTION/REFUSE gate.
2. **A practical block-bootstrap N-ceiling estimator** — a principled irreducible-noise estimator with seed-collapse diagnostic and residual-correlation correction, validated on synthetic ground truth.
3. **The certification gap (text finding)** — rho = 0.31 and 56% inversion rate across 16 embedding families on MIRACL retrieval; scalar leaderboards and structural audits tell different stories.
4. **Architecture-invariance under N-ceiling (geospatial finding)** — across 27 CONUS tasks x 3 model families, cross-family R-squared spread is ~0.037 while N-ceiling varies 4x, meaning task noise dominates model choice.

Released: `yrsn` certificate library, OOF predictions for all 16 text models and 27 geo tasks, audit test suite, CONUS-27 task manifests.

---

## Section structure — 9 pages main

**Design rule:** Sections 6 and 7 are the paper's center of gravity. Sections 3-5 support them; they do not overshadow them.

| S | Section | Pages | Notes |
|---|---|---|---|
| 1 | Introduction | 1.0 | Opens with rho = 0.31 and 56% headline; motivates two-findings framing |
| 2 | Related work | 0.5 | Bias-variance, aleatoric/epistemic, MTEB/BEIR, PDFM/SustainBench, trustworthiness evals |
| 3 | R/S/N decomposition | 1.25 | Simplex, kappa-gate, alpha reconstruction. Keep formal but concise — reader needs definitions, not derivation history |
| 4 | Certification and audit | 1.0 | Issuer -> mixer -> auditor; four audit properties; PASS/CAUTION/REFUSE gate logic |
| 5 | N-ceiling estimation | 1.25 | Block-bootstrap algorithm + synthetic validation + CONUS-27 real spectrum |
| 6 | **Finding 1: the certification gap in text retrieval** | 1.5 | 16 models, rho = 0.31, 56% inversion, simplex positioning, confusion matrix analysis |
| 7 | **Finding 2: architecture-invariance under N-ceiling in geo** | 1.5 | 27 tasks x 3 families, 0.037 R-squared spread, N-ceiling 0.155-0.593, policy implications |
| 8 | When evaluation should rank models vs characterize tasks | 0.5 | Synthesis: the protocol identifies which regime you're in |
| 9 | Limitations, ethics, conclusion | 0.5 | OOF requirement, non-stationarity caveats, patent footnote |

**Total:** 9.0 pages.

---

## S1 draft — opening in real prose (~350 words)

> A model that tops a leaderboard is not always the model we should trust. Across 16 open text embedding families evaluated on MIRACL retrieval, the rank-order correlation between scalar accuracy and a structural certificate of the same models is only 0.31 — statistically distinguishable from random, but far from the implicit promise of a leaderboard that a higher number means a better model. Fifty-six percent of evaluated models invert by three or more positions when re-ranked by the certificate. One model, **nova_embed**, ranks first by accuracy and fifteenth by certificate. Another, **titan_v2**, ranks fifteenth by accuracy and fourth by certificate. The certificate and the leaderboard are measuring different things.
>
> In a different domain, the same protocol surfaces a different but complementary finding. Across 27 ZCTA-level geospatial regression tasks covering health, socioeconomic, and environmental outcomes, three architecturally distinct model families (PCA32-projected GBDT, spatial-lag regression, and GraphSAGE with dual-graph morphing) produce task-level R-squared values whose cross-family spread averages 0.037. The same tasks exhibit an irreducible noise ceiling ranging from 0.155 to 0.593 — a four-fold variation. In other words: for geospatial regression on these tasks, the architectural choice is nearly invisible against the axis of task intrinsic noise. The ceiling, not the model, is the constraint.
>
> These two findings define two regimes in which evaluation has different jobs. In the text regime, models reach comparable scalar scores by structurally distinct paths, and the evaluation's job is to discriminate among them. In the geospatial regime, models converge, and the evaluation's job is to characterize the task. A single scalar metric cannot do either job on its own.
>
> We contribute a certification protocol that does both. The R/S/N decomposition splits predictive error into relevance (R), recoverable shortfall (S), and irreducible noise (N) on the probability simplex R+S+N=1. A four-property audit certifies the decomposition is well-formed. A block-bootstrap N-ceiling estimator provides the irreducible-noise component with calibrated confidence intervals. We release the certificate library, the out-of-fold predictions for all 16 text models and 27 geospatial tasks, and the audit test suite, so the findings above can be reproduced and extended.

---

## What actually exists (ground truth inventory)

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
| CONUS-27 — 3 families x 27 tasks with R-squared metrics | On S3 (2026-04-20) | `s3://yrsn-checkpoints/geo-cert/` |

## Key empirical results (measured, not projected)

**Text retrieval (16 models):**
- 56% of models have >= 3-position rank inversion (accuracy vs alpha)
- Spearman rho(accuracy rank, alpha rank) = 0.312
- nova_embed: accuracy #1 -> alpha #15 (14-position inversion)
- titan_v2: accuracy #15 -> alpha #4 (11-position inversion)
- openai_3_small: accuracy #6 -> alpha #1 (5-position inversion)
- ALL 16 models fail Gate 3 with RE_ENCODE decision

**Geospatial (3 families x 27 CONUS tasks):**
- Mean TRF = 0.328, range 0.156 (night_lights) to 0.593 (cholesterol_screening)
- Cross-family R-squared spread is small (~0.037) — representation choice matters less than task noise floor
- Task difficulty varies 4x while model family ranking is nearly constant

---

## Figure list — locked

| # | Figure | Data status | Script |
|---|---|---|---|
| Fig 1 | Accuracy rank vs alpha-rank scatter, 16 text models | Ready | `figures/fig1_rank_inversion_scatter.py` |
| Fig 2 | R/S/N simplex ternary plot, 16 text models | Ready | `figures/fig2_simplex_ternary.py` |
| Fig 3 | nova_embed vs openai_3_small confusion matrices | Ready | `figures/fig3_confusion_comparison.py` |
| Fig 4 | N-ceiling heatmap across 27 CONUS tasks | Ready | `figures/fig4_nceiling_heatmap.py` |
| Fig 5 | Cross-family R-squared spread vs N-ceiling, 27 tasks | Ready | `figures/fig5_spread_vs_nceiling.py` |
| Fig 6 | Existing: layer 1 substrate | Done | existing |
| Fig 7 | Existing: layer 2 ladder | Done | existing |
| Fig 8 | Synthetic N-ceiling validation (ground truth recovery) | Pending synthetic experiment | Not built |
| Tbl 1 | 16-model certificate leaderboard | Ready | Reformat leaderboard.json |
| Tbl 2 | CONUS-27 per-task N-ceiling with CIs | Ready | Reformat |
| Tbl 3 | Audit property pass/fail matrix | Ready | Build |

---

## Artifact release checklist

- [ ] `yrsn` package on PyPI (certificate library, installable)
- [ ] HuggingFace repo: `rudymartin/conus-27` — tasks, splits, OOF predictions, Croissant metadata
- [ ] HuggingFace repo: `rudymartin/text-retrieval-16-oof` — 16 models x MIRACL OOF predictions, Croissant metadata
- [ ] GitHub repo: audit test suite + reproduction scripts + MIT/Apache license
- [ ] Dataset cards with RAI fields
- [ ] Maintenance plan (one page in each repo)
- [ ] Patent disclosure footnote in S1, cross-referenced in ethics statement

---

## Deliberately avoided

(Kept as a discipline record — do not re-import these.)

- Conditional language around geo (data exists, section is confirmed)
- "I invented this" disclaimers (findings are measured)
- MoE-centered framing (architecture-neutral)
- "Recipe" framing (no model-win anchor)
- Tabular domain as third substrate (two is cleaner)
- PDFM as branded artifact (motivation only, not claimed)
- "Substrate-independent" in the title
- Overly detailed threshold constants in main narrative (appendix)
- "First principled" novelty claims without literature defense

---

## Discipline rules (from v1)

- All certificates come from real YRSN code (`yrsn.DecompositionScore`, `SequentialGatekeeper`)
- No hand-rolled formulas — every metric traces to a yrsn import
- All code and OOF predictions released for reproducibility
- The "certification gap" is an empirical finding (rho=0.31, 56% inversion), not a coined term
- Sections 6 and 7 are the center of gravity; sections 3-5 serve them
- Write findings sections (6, 7) before polishing methods (3, 4, 5)
