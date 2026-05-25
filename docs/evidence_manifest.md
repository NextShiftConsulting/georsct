# Evidence Manifest — NeurIPS 2026 Submission

**Paper:** *The Certification Gap: When Leaderboards Rank the Wrong Models*
**Author:** Rudy Martin, Next Shift Consulting LLC
**Track:** Evaluations & Datasets
**Manifest created:** 2026-04-23

This document maps every empirical claim in the paper to its source artifact,
experiment series, and commit hash. Reviewers or future-self can trace any
number back to runnable code and raw data.

---

## Finding 1: The Certification Gap in Text Retrieval (Section 6)

### Headline claims

| Claim | Value | Source artifact | Repo | Path |
|-------|-------|-----------------|------|------|
| Spearman rho (accuracy vs alpha rank) | 0.31 | leaderboard.json | yrsn-train | `apps/train_rsn_head/paper_data/leaderboard.json` |
| Models with rank inversion >= 3 | 9/16 (56%) | leaderboard.json | yrsn-train | same |
| nova_embed: accuracy #1, alpha #15 | 14-position inversion | leaderboard.json | yrsn-train | same |
| openai_3_small: accuracy #6, alpha #1 | best certificate | leaderboard.json | yrsn-train | same |
| titan_v2: accuracy #15, alpha #4 | 11-position positive inversion | leaderboard.json | yrsn-train | same |
| All 16 models fail Gate 3 | universal RE_ENCODE | leaderboard.json | yrsn-train | same |
| Alpha range | 0.482 -- 0.514 (spread 0.032) | Table 1 | rsct-geo_cert | `paper/table1_leaderboard.tex` |
| Kappa range | 0.207 -- 0.234 (spread 0.027) | Table 1 | rsct-geo_cert | same |
| Median absolute rank inversion | 3.0 positions | computed from Table 1 | rsct-geo_cert | same |

### Experiment series

| Series | Purpose | Job artifacts |
|--------|---------|---------------|
| S016C | Text retrieval baseline (MIRACL 30K balanced pairs, 16 models) | `s3://yrsn-datasets/s016c/` |
| S017 | Substrate independence + geometry sufficiency DOEs | `s3://yrsn-datasets/s017/` |

### Figures

| Figure | Script | Data source | Key result |
|--------|--------|-------------|------------|
| Fig 1 | `figures/fig1_rank_inversion_scatter.py` | `leaderboard.json` via `../../yrsn-train/` | Accuracy rank vs alpha rank scatter |
| Fig 2 | `figures/fig2_simplex_ternary.py` | leaderboard.json | 16-model R/S/N ternary plot |
| Fig 3 | `figures/fig3_confusion_comparison.py` | per-model confusion matrices | nova_embed vs openai_3_small |

### Per-model OOF predictions

All 16 models' OOF predictions, embeddings, and certificate triples are stored in:
- `yrsn-train/apps/train_rsn_head/paper_data/`
- S3: `s3://yrsn-datasets/s016c/` (frozen embeddings)

---

## Finding 2: Architecture-Invariance Under N-Ceiling (Section 7)

### Headline claims

| Claim | Value | Source artifact | Repo | Path |
|-------|-------|-----------------|------|------|
| Cross-family R-squared spread (mean) | 0.037 | Table 2 | rsct-geo_cert | `paper/table2_conus27.tex` |
| N-ceiling range | 0.155 -- 0.593 (4x variation) | Table 2 | rsct-geo_cert | same |
| Mean N-ceiling | 0.328 | Table 2 | rsct-geo_cert | same |
| Number of tasks | 27 (CONUS-27) | Table 2 | rsct-geo_cert | same |
| Max cross-family spread | 0.166 (elevation) | Table 2 | rsct-geo_cert | same |
| Easiest task | night_lights (N-ceil 0.156) | Table 2 | rsct-geo_cert | same |
| Hardest task | cholesterol_screening (N-ceil 0.593) | Table 2 | rsct-geo_cert | same |
| Elevation: GNN advantage | R-sq 0.517 vs PCA 0.350 (spread 0.166) | Table 2 | rsct-geo_cert | same |
| PCA32 mean R-sq | 0.655 | Table 2 | rsct-geo_cert | same |
| GNN mean R-sq | 0.651 | Table 2 | rsct-geo_cert | same |
| Spatial Lag mean R-sq | 0.666 | Table 2 | rsct-geo_cert | same |

### Experiment series

| Series | Purpose | Job artifacts | Key commits |
|--------|---------|---------------|-------------|
| S018a | Smoke test: proxy certifier on 26 tasks | `s3://yrsn-datasets/rsct_curriculum/series_018/` | `46d0e51` (v2 formula), `48d6bc0` (H1/H2 results) |
| S018c | Proxy vs canonical rotor comparison | same S3 prefix | bucket_ratio 1.301 (proxy) vs 0.826 (canonical inverted) |
| S018e | GNN pilot: 27 tasks, GraphSAGE-32 | S3: `representations/graph_latent_v1.pt`, `certifiers/proxy_certifier_graph_v1.json` | `f508cf6` (L2-norm removal) |
| S018g | Gate-controlled super-resolution | `evidence/s018g_results.json` | H4 PASS: 3/3 tasks improved |
| S018h | Canonical certification attempt | `evidence/s018h_results.json` | H6 KILL: 4/27 tasks (canonical rotor fails on geo) |

### Certifier provenance (proxy_spearman)

| Version | Formula | Status | Commit |
|---------|---------|--------|--------|
| proxy_spearman_v1 | `kappa = 0.5*r + 0.3*alpha + 0.2*(1-n); kappa_gate = kappa*(1-0.5*sigma)` | SUPERSEDED | pre-`46d0e51` |
| proxy_spearman_v2 | `kappa = R/(R+N)` (canonical RSCT) | CURRENT | `46d0e51` (2026-04-14) |

**v2 validation:** H1 18/26 negative rho, H2 21/26 gated R-sq improvement @ 25% coverage,
H4 3/3 tasks improved, zero rollbacks. Source: `yrsn-experiments/exp/series_018/evidence/s018a_summary.json`

### Figures

| Figure | Script | Data source | Key result |
|--------|--------|-------------|------------|
| Fig 4 | `figures/fig4_nceiling_heatmap.py` | Table 2 data / S018 results | 27-task N-ceiling heatmap |
| Fig 5 | `figures/fig5_spread_vs_nceiling.py` | Table 2 data | Spread vs N-ceiling (architecture invisible) |

### Three model families (training artifacts)

| Family | Repo | Export script | S3 artifact |
|--------|------|---------------|-------------|
| PCA32-GBDT | yrsn-train | `models/v2/train_and_export_v2.py` | `geo_cert_pca32_structure_v2` |
| Spatial Lag-GBDT | yrsn-train | `models/v1/train_and_export_spatial_lag.py` | `geo_cert_spatial_lag_proxy_v1` |
| GraphSAGE GNN | yrsn-train | `models/v2/train_and_export_gnn_v2.py` | `geo_cert_gnn_structure_v2` |

---

## Cross-Domain Protocol Validation (Section 7.6)

| Property | Text (S016C/S017) | Geo (S018) | Same code? |
|----------|-------------------|------------|------------|
| Decomposition | `yrsn.DecompositionScore` | `yrsn.DecompositionScore` | Yes |
| Gate logic | `yrsn_controlplane.SequentialGatekeeper` | `yrsn_controlplane.SequentialGatekeeper` | Yes |
| Certificate schema | RSCTCertificate v2.1 | RSCTCertificate v2.1 | Yes |
| Domain adaptation | None (same pipeline) | None (same pipeline) | Yes |

---

## Synthetic Validation (Section 5 / Appendix)

| Claim | Source | Status |
|-------|--------|--------|
| N-ceiling estimator validated on synthetic ground truth | `figures/fig8_synthetic_nceiling_validation.py` | Pending synthetic experiment completion |
| Block-bootstrap with seed-collapse diagnostic | `rsct/` library | Implemented |

---

## Data Sources

| Dataset | Access | License | Paper section |
|---------|--------|---------|---------------|
| MIRACL English | HuggingFace `miracl/miracl` | Apache 2.0 | Section 6 |
| CDC PLACES | CDC open data | Public domain | Section 7 |
| ACS (American Community Survey) | Census Bureau | Public domain | Section 7 |
| PDFM CONUS features | Google Research (requires approval) | Research license | Section 7 |

---

## Repo Cross-References

| Repo | Role in paper | Key paths |
|------|---------------|-----------|
| `rsct-geo_cert` | Paper manuscript, figures, DOE scaffolding | `paper/`, `figures/`, `experiments/` |
| `yrsn-train` | Model training, leaderboard data, export scripts | `apps/train_rsn_head/paper_data/`, `apps/geo_cert/models/` |
| `yrsn-experiments` | S018 experiment series (geo certifier) | `exp/series_018/evidence/` |
| `yrsn` | Core library (DecompositionScore, certificates) | `src/yrsn/core/` |
| `yrsn-controlplane` | SequentialGatekeeper, CertificateEstimate | `src/yrsn_controlplane/` |

---

## Version Pins (at time of submission)

To be filled after s018c re-certification:

| Repo | Commit | Tag |
|------|--------|-----|
| rsct-geo_cert | _TBD_ | _TBD_ |
| yrsn-train | `db9f4ef` (proxy_spearman_v2 manifests) | — |
| yrsn-experiments | `9b1f46b` (CERTIFIER_VERSION flip) | — |
| yrsn | _TBD_ | — |
| yrsn-controlplane | _TBD_ | — |

---

## Checklist Before Submission

- [ ] s018c re-certification completed with `certifier = "proxy_spearman_v2"`
- [ ] Evidence artifacts regenerated and archived (v1 → `archive/s018c-v1-pre20260414/`)
- [ ] `Dep-s018c-v2-migration.md` written
- [ ] All figure scripts run clean against final data
- [ ] Table 1 and Table 2 verified against latest JSON sources
- [ ] Version pins filled in above
- [ ] Patent footnote in Section 9 references correct claims
- [ ] Artifact release checklist (OOF predictions, audit suite, CONUS-27 manifests)
