# wsp/georsct-hf — HuggingFace Deployment Folder

This folder mirrors the non-data file layout of `huggingface.co/datasets/rudymartin/georsct`.

## Workflow

1. **Edit here** (or in the canonical source location, then copy here)
2. **Review** the files in this folder
3. **Upload** with `python upload_to_hf.py`

## What's in this folder

Everything that goes on HuggingFace except large binary data files (parquet, geoparquet, npz).
The folder structure matches the HF repo layout exactly.

```
wsp/georsct-hf/
├── README.md                          → HF root
├── load_georsct.py                    → HF root
├── quickstart.py                      → HF root
├── croissant.json                     → HF root
├── georsct_schema.json                → HF root
├── build_manifest.json                → HF root
├── georsct_checksums.sha256           → HF root
├── validation_report.json             → HF root
├── VALIDATION_CROSS_CHECK.md          → HF root
├── data/
│   └── georsct_croissant_neurips_compliant.json  → HF data/
├── specifications/
│   └── croissant.json                 → HF specifications/
└── representations/
    └── (npz files not stored here — too large for git)
```

## What is NOT in this folder

Large binary files stay on HF only (uploaded separately):

- `georsct_table.parquet` (19 MB)
- `georsct_simplified_001.geoparquet` (66 MB)
- `cdc_places_ci.parquet` (1.7 MB)
- `zcta_acs_margins_of_error.parquet` (7.9 MB)
- `representations/*.npz` (8 MB total)

## Canonical source mapping

| This folder | Canonical source in repo |
|-------------|-------------------------|
| `load_georsct.py` | `code/load_georsct.py` |
| `quickstart.py` | `code/quickstart.py` |
| `README.md` | `README.md` (repo root) |
| `croissant.json` | `croissant.json` (repo root) |
| `georsct_schema.json` | `specifications/georsct_schema.json` |
| `build_manifest.json` | `specifications/build_manifest.json` |
| `georsct_checksums.sha256` | `specifications/georsct_checksums.sha256` |
| `validation_report.json` | `specifications/validation_report.json` |
| `VALIDATION_CROSS_CHECK.md` | `specifications/VALIDATION_CROSS_CHECK.md` |
| `data/georsct_croissant_neurips_compliant.json` | `data/georsct_croissant_neurips_compliant.json` |
| `specifications/croissant.json` | `specifications/croissant.json` |
