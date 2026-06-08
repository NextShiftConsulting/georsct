# Repository Split: HuggingFace vs GitHub

GeoRSCT is distributed across two repositories with distinct roles.

## HuggingFace (`huggingface.co/datasets/rudymartin/georsct`)

**Purpose**: Data distribution and user-facing benchmark consumption.

Users clone or download from HuggingFace to run baselines, evaluate models, and reproduce paper results.

| Content | Files |
|---------|-------|
| Benchmark data | `georsct_table.parquet`, `georsct_simplified_001.geoparquet` |
| Uncertainty sidecars | `cdc_places_ci.parquet`, `zcta_acs_margins_of_error.parquet` |
| User-facing code | `load_georsct.py`, `quickstart.py` |
| Pre-computed representations | `representations/*.npz` (PCA32, spatial lag, GNN latents) |
| Dataset metadata | `croissant.json`, `georsct_schema.json`, `build_manifest.json` |
| Integrity | `georsct_checksums.sha256`, `validation_report.json`, `VALIDATION_CROSS_CHECK.md` |
| README | Dataset card with usage, schema, evaluation protocols |

## GitHub (`github.com/NextShiftConsulting/georsct`)

**Purpose**: Full pipeline source code, experiment infrastructure, and research artifacts.

Researchers use the GitHub repo to audit the build pipeline, reproduce the dataset from source, run diagnostics, and access experiment evidence.

| Content | Directory | Files |
|---------|-----------|-------|
| Build pipeline | `code/benchmark/` | 11 scripts: `build_geoparquet.py`, `build_flood_zones.py`, `fetch_acs_moe.py`, etc. |
| Diagnostic tools | `code/diagnostics/` | 8 scripts: `certificate_audit.py`, `task_residual_floor_estimator.py`, etc. |
| Solver training | `code/solvers/` | `train_and_export_gnn_v2.py`, `train_and_export_v2.py` |
| Figures and inference | `code/` | `figures.py`, `inference.py` |
| User-facing code (source) | `code/` | `load_georsct.py`, `quickstart.py` (canonical copies; deployed to HF) |
| Intermediate pipeline data | `data/` | Crosswalks, SVI, HIFLD, drive times, splits |
| Experiment predictions | `predictions/` | Solver metrics, certificate RSN, leaderboards |
| Specifications + metadata | `specifications/` | Taxonomy, injection validation, evidence manifest, schema, checksums, build manifest, validation report |
| Certificates | `certificates/` | RSN certificate parquet |

## Canonical Source for Shared Files

All non-data files on HuggingFace have canonical copies in this GitHub repo. Edit here, deploy to HF.

| HF path | GitHub canonical path | Category |
|---------|----------------------|----------|
| `README.md` | `README.md` | Dataset card |
| `load_georsct.py` | `code/load_georsct.py` | User-facing code |
| `quickstart.py` | `code/quickstart.py` | User-facing code |
| `croissant.json` | `croissant.json` | Metadata |
| `georsct_schema.json` | `specifications/georsct_schema.json` | Metadata |
| `build_manifest.json` | `specifications/build_manifest.json` | Metadata |
| `georsct_checksums.sha256` | `specifications/georsct_checksums.sha256` | Integrity |
| `validation_report.json` | `specifications/validation_report.json` | Integrity |
| `VALIDATION_CROSS_CHECK.md` | `specifications/VALIDATION_CROSS_CHECK.md` | Integrity |
| `data/georsct_croissant_neurips_compliant.json` | `data/georsct_croissant_neurips_compliant.json` | Metadata |
| `specifications/croissant.json` | `specifications/croissant.json` | Metadata |

**Rule**: Edit in GitHub, then upload to HF. Never edit directly on HF.

## What Does NOT Go on HuggingFace

- Build pipeline scripts (reproducibility, not consumption)
- Diagnostic and certification tools (research tooling)
- Solver training code (experiment infrastructure)
- Intermediate pipeline artifacts (crosswalks, raw SVI, etc.)
- Experiment predictions and evidence (research outputs)
- Specification JSONs (taxonomy, injection validation)

These are GitHub-only because they serve the paper's reproducibility claims, not the benchmark user's workflow.

## Upload Procedure

The `wsp/georsct-hf/` folder mirrors the HF repo layout. To deploy:

```bash
# Preview what will be uploaded
python wsp/georsct-hf/upload_to_hf.py --dry-run

# Upload all non-data files to HuggingFace
python wsp/georsct-hf/upload_to_hf.py
```

After editing a canonical source (e.g., `code/load_georsct.py`), copy it into `wsp/georsct-hf/` before uploading.
See `wsp/georsct-hf/DEPLOY.md` for the full canonical-source mapping.

Note: on Windows, the upload script uses the Python API (not `huggingface-cli`) to avoid cp1252 Unicode errors.
