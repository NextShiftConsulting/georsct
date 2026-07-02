# S040a: NFIP Bias Correction Ablation

**Parent:** s035-model-ladder (shares scenarios, folds, assembled features)
**Pattern:** Single-variable-under-test with certificate as instrument (s019g)

## Purpose

Ablation study of a 7-layer NFIP claims bias correction pipeline.
Measures RSCT certificate trajectories at each correction layer to
identify which corrections unblock which quality gates.

## Correction Ladder

| Condition | Correction | Data Source |
|-----------|-----------|-------------|
| C0 | Raw NFIP claims (baseline) | Existing `raw/openfema/nfip_claims_dr*.parquet` |
| C1 | Tobit censoring adjustment | Computed from C0 |
| C2 | IPW selection bias | `raw/openfema/s040a/nfip_policies_*.parquet` |
| C3 | Private market adjustment | External (First Street / Verisk) |
| C4 | Event-specific fraud exclusion | `processed/s040a/nfip_claims_c4_dr*.parquet` |
| C5 | Bai-Perron structural breaks | Computed from C4 |
| C6 | Buhlmann-Straub credibility | Computed from C5 |
| C7 | Bayesian multi-source fusion | `raw/openfema/s040a/ia_registrations_dr*.parquet` |

## Minimum Viable Ablation

C0 -> C4 -> C7 (raw -> event-corrected -> fused)

## Key Files

| File | Purpose |
|------|---------|
| `DOE_LOCKED.md` | Full experiment design (DRAFT -- lock before launch) |
| `CHECKLIST.md` | Pre-flight validation |
| `EXPERIMENT_STATUS.yaml` | Phase tracking |
| `jobs/s040a_bias_correction/` | Data pull scripts (in `georsct/wsp/floodrsct/jobs/`) |
| `scripts/launch_s040a_*.py` | SageMaker launchers (in `georsct/wsp/floodrsct/scripts/`) |

## Commands

```bash
# Data pulls (parallel -- no dependencies between them)
python scripts/launch_s040a_fetch_ia.py --dry-run
python scripts/launch_s040a_fetch_policies.py --dry-run

# C4 event correction (reads existing raw claims)
python scripts/launch_s040a_build_c4.py --dry-run
```
