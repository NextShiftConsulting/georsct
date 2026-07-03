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
| User-facing code (source) | `georsct/loaders/` | `georsct.py` (canonical loader; deployed to HF as `load_georsct.py`) |
| Intermediate pipeline data | `data/` | Crosswalks, SVI, HIFLD, drive times, splits |
| Experiment predictions | `predictions/` | Solver metrics, certificate RSN, leaderboards |
| Specifications + metadata | `code/georsct_hf/` | Schema, checksums, build manifest, validation report (v24 versions) |
| Certificates | `certificates/` | RSN certificate parquet |

## Canonical Source for Shared Files

All non-data files on HuggingFace have canonical copies in this GitHub repo. Edit here, deploy to HF.

| HF path | GitHub canonical path | Category |
|---------|----------------------|----------|
| `README.md` | `README.md` | Dataset card |
| `load_georsct.py` | `georsct/loaders/georsct.py` | User-facing code |
| `quickstart.py` | `code/quickstart.py` | User-facing code |
| `croissant.json` | `croissant.json` | Metadata |
| `georsct_schema.json` | `code/georsct_hf/georsct_schema.json` | Metadata |
| `build_manifest.json` | `code/georsct_hf/build_manifest.json` | Metadata |
| `georsct_checksums.sha256` | `code/georsct_hf/georsct_checksums.sha256` | Integrity |
| `validation_report.json` | (removed during restructuring) | Integrity |
| `VALIDATION_CROSS_CHECK.md` | `code/georsct_hf/VALIDATION_CROSS_CHECK.md` | Integrity |
| `data/georsct_croissant_neurips_compliant.json` | `data/georsct_croissant_neurips_compliant.json` | Metadata |
| `specifications/croissant.json` | `code/georsct_hf/croissant.json` | Metadata |

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

The `code/georsct_hf/` folder mirrors the HF repo layout. To deploy:

```bash
# Preview what will be uploaded
python code/georsct_hf/upload_hf.py --dry-run

# Upload all non-data files to HuggingFace
python code/georsct_hf/upload_hf.py
```

After editing a canonical source (e.g., `georsct/loaders/georsct.py`), copy it into `code/georsct_hf/` before uploading.
See `code/georsct_hf/DEPLOY.md` for the full canonical-source mapping.

Note: on Windows, the upload script uses the Python API (not `huggingface-cli`) to avoid cp1252 Unicode errors.
