# S035-Model-Ladder: Diagnosis-Driven Representation Ablation

Controlled experiment testing whether GeoRSCT audit flags predict which
representation fixes improve flood risk prediction.

## Quick Start

```bash
# Phase 0: Lock data + run audit battery
python scripts/launch_generate_data_lock_manifest.py --scenario houston

# Phase 1: Generate fixed folds
python scripts/launch_generate_folds.py --scenario houston

# Phase 2: Train R0 baseline (3 targets x 2 solvers x 3 splits)
python scripts/launch_train_r0_baseline.py --scenario houston

# Phase 3-4: R1/R2 (same folds, different features)
python scripts/launch_train_r1_hydrology.py --scenario houston
python scripts/launch_train_r2_temporal.py --scenario houston

# Phase 5: Compute uplift table (local)
python jobs/compute_uplift_table.py --scenario houston
```

## Key Files

| File | Purpose |
|------|---------|
| `DOE_LOCKED.md` | Experiment design, hypotheses, decision tree |
| `CHECKLIST.md` | Pre-flight validation checklist |
| `configs/*.yaml` | Per-phase run configurations |
| `evidence/` | Hypothesis evidence JSONs |
| `results/` | Metrics, predictions, figures |

## Job Scripts (in `data/floodrsct/jobs/`)

| Script | Phase | Description |
|--------|-------|-------------|
| `generate_data_lock_manifest.py` | 0 | Run 18-audit battery, produce manifest |
| `generate_folds.py` | 1 | Create 3 deterministic fold assignments |
| `train_r0_baseline.py` | 2 | HistGBDT + Ridge on R0 features |
| `train_r1_hydrology.py` | 3 | Same solvers on R0 + hydrology features |
| `train_r2_temporal.py` | 4 | Same solvers on R0 + hydrology + temporal |
| `compute_uplift_table.py` | 5 | Paired deltas, diagnostic gain, money table |

## The Money Table

| Audit Flag | Treatment (Representation) | Response (Metric) |
|------------|---------------------------|-------------------|
| B.1 MAUP | R1 adds HUC/catchment | uplift_R1_vs_R0 |
| B.2 Scale | R1 adds stream density | uplift_R1_vs_R0 |
| C.1 Vintage | R2 adds temporal dynamics | uplift_R2_vs_R1 |
| C.3 Missingness | R1 fills spatial gaps | uplift_R1_vs_R0 |

Core claim: `diagnostic_gain = uplift_when_flagged - uplift_when_not_flagged > 0`
