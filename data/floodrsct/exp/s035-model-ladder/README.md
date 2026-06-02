# S035-Model-Ladder: Diagnosis-Driven Representation Ablation

Controlled experiment testing whether GeoRSCT audit flags predict which
representation fixes improve flood risk prediction. Target: SIGSPATIAL 2026.

## Experiment Architecture

```
Phase 1 (R0 train)
  -> Phase 4a (kappa R0) -> Phase 4.5a (cert R0)
    -> Phase 2 (R1 train)
      -> Phase 4b (kappa R1) -> Phase 4.5b (cert R1)
        -> [Conditional] Phase 2.5 (R1.5 FAST features)
          -> Phase 3 (R2 train)
            -> Phase 4c (kappa R2) -> Phase 4.5c (cert R2)
              -> Phase 5 (money table + hypothesis tests)
                -> Phase 6 (DGM routing)
                  -> Phase 7 (FAST validation)
```

## DOE Documents

| Document | Scope | Hypotheses |
|----------|-------|------------|
| [DOE_LOCKED.md](DOE_LOCKED.md) | Original locked design (v1.1) | H1-H4 |
| [DOE_AMENDMENT_v1.2.md](DOE_AMENDMENT_v1.2.md) | Amendments v1.2-v1.6 (13 changes) | H5-H6 added |
| [DOE_R0_baseline.md](DOE_R0_baseline.md) | R0 control arm — 33 static tabular features | H1: baseline skill |
| [DOE_R1_spatial.md](DOE_R1_spatial.md) | R1 treatment — hydrology + W-matrix (61-63 features) | H2: diag_leakage predicts R0->R1 uplift |
| [DOE_R2_temporal.md](DOE_R2_temporal.md) | R2 treatment — event dynamics (70-72 features) | H3: diag_transfer predicts R1->R2 uplift |
| [DOE_kappa_cascade.md](DOE_kappa_cascade.md) | Progressive diagnostics + pre-registration | H4: cascade predicts + confirms |
| [DOE_FAST_validation.md](DOE_FAST_validation.md) | FEMA FAST features (R1.5) + external validation | H6: engineering model correlation |
| [DOE_certificate_dgm.md](DOE_certificate_dgm.md) | RSCT certification + DGM routing | H5: DGM routes to optimal arm |
| [DOE_R4_vlm.md](DOE_R4_vlm.md) | R4 VLM arm — Gemini/Jina/Nova/Qwen 4-way comparison | H7-H9: VLM signal + solver robustness |
| [future-work-llm-representation-encoder.md](future-work-llm-representation-encoder.md) | LLM text embeddings (deferred) | Section 9 |

## Hypotheses Summary

| ID | Statement | Type | Test |
|----|-----------|------|------|
| H1 | R0 achieves R2 > 0 on >= 2 targets | Gate | Baseline skill |
| H2 | diag_leakage predicts R0->R1 uplift | **PRIMARY** | Spearman rho + Holm-Bonferroni |
| H3 | diag_transfer predicts R1->R2 uplift | Secondary | Spearman rho |
| H4 | Kappa cascade shows progressive flag clearing | Secondary | Movement table |
| H5 | DGM routing matches exhaustive best arm | Exploratory | Hit rate + binomial CI |
| H6 | R2 predictions correlate with FAST estimates | Exploratory | Spearman rho across levels |
| H7 | VLM extracts flood risk signal from map + text (rho > 0.3) | Gate | Spearman rho vs NFIP claims |
| H8 | VLM choice is not significant (pairwise rho > 0.7) | Secondary | Inter-rater agreement |
| H9 | VLM scores correlate with R0 kappa diagnostics | Exploratory | Flagged vs unflagged comparison |

## Representation Ladder

```
R0:   36 features  — Static tabular (ACS, SVI, FEMA zones, TWI, HIFLD)
R1:   61-63        — + Hydrology (16) + scenario-specific (1-3) + W-matrix spatial (8)
R1.5: 67-69        — + FAST engineering model outputs (6) [conditional on NSI data]
R2:   76-78        — + Temporal event dynamics (9)
R3:   raster       — Image patch representation (existing CNN/YRSN) [adapt, don't invent]
R4:   map + text   — VLM: Gemini Flash / Jina VLM / Nova Lite / Qwen2.5-VL (4-way comparison)
R5:   evidence     — VLA/agent: evidence + action choice [demo/action layer]
```

## Scenarios

| Scenario | Rows | Targets | FAST Tier |
|----------|------|---------|-----------|
| Houston | 396 | 3 (NFIP, 311, HWM) | Primary |
| SW Florida | 606 | 1 (NFIP) | Stretch |
| NYC | 422 | 2 (NFIP, 311) | Primary |
| Riverside | 172 | 1 (NFIP) | Excluded |
| New Orleans | 20 | Illustrative only | N/A |

Total modelable cells: 7 (scenario x target)

## Key Files

| File | Purpose |
|------|---------|
| `CHECKLIST.md` | Pre-flight validation checklist |
| `MODELS.md` | Solver selection rationale |
| `configs/*.yaml` | Per-scenario regime configurations |
| `requirements.txt` | SageMaker dependencies |
| `evidence/` | Hypothesis evidence JSONs (post-execution) |
| `results/` | Metrics, predictions, figures (post-execution) |

## Job Scripts (in `data/floodrsct/jobs/`)

| Script | Phase | Description |
|--------|-------|-------------|
| `train_r0_baseline.py` | 1 | HistGBDT + Ridge on R0 features; generates folds |
| `train_r1_hydrology.py` | 2 | Same solvers on R0 + R1 features |
| `train_r2_temporal.py` | 3 | Same solvers on R0 + R1 + R2 features |
| `compute_diagnostics.py` | 4a-c | Progressive kappa diagnostics at each level |
| `compute_certificates.py` | 4.5a-c | RSN certificates from yrsn (planned) |
| `compute_uplift_table.py` | 5 | Money table + hypothesis tests + Holm-Bonferroni |
| `compute_dgm_routing.py` | 6 | DGM morph routing analysis (planned) |
| `run_fast_zcta.py` | 7a | FAST ZCTA features — calls floodcaster (planned) |
| `compute_fast_validation.py` | 7b | External validation correlations (planned) |

## yrsn Integration

yrsn is the measurement plane, not the solver plane:

| yrsn Component | s035 Role |
|----------------|-----------|
| `core.kappa.spatial.compute` | Moran's I for diag_residual_spatial |
| `core.certificates.core.YRSNCertificate` | RSN certificate per cell |
| `core.quality.alpha` / `omega` | Quality signals |
| `core.dgm_unified.DualGraphSystem` | Routing analysis |
| `core.dgm_unified.MorphType` | Typed morph operators |

Solvers (HistGBDT, Ridge) remain transparent sklearn for paper auditability.

## floodcaster Integration

floodcaster is the flood analysis API (wraps sphere Hazus engine):

| floodcaster Component | s035 Role |
|-----------------------|-----------|
| `analysis.run_nsi_flood_analysis()` | NSI buildings + depth raster -> per-building losses (Phase 7a) |
| `nsi_sources.fetch_nsi()` | NSI 2.0 building fetch from USACE API |
| `aggregation.aggregate_by_zcta()` | Per-building losses -> ZCTA-level features |

sphere classes used under the hood: `HazusFloodAnalysis`, `NsiBuildings`, `DefaultFloodVulnerability`.
s035 does NOT import sphere directly — floodcaster is the interface.
