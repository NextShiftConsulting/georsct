#!/usr/bin/env python3
from swarm_auth import get_credential
from huggingface_hub import HfApi, CommitOperationAdd
import re

api = HfApi(token=get_credential('HF_TOKEN'))
REPO = 'rudymartin/georsct'

readme_path = api.hf_hub_download(REPO, 'README.md', repo_type='dataset', force_download=True)
with open(readme_path, encoding='utf-8') as f:
    readme = f.read()

# 1. Dataset Summary: 63 → 70
readme = readme.replace(
    'it curates 27 regression targets from multiple sources, 63 solver-usable input features,',
    'it curates 27 regression targets from multiple sources, 70 solver-usable input features (33 ACS + 37 enrichment),'
)

# 2. Field Schema: remove spatial-lag from parquet column list
readme = readme.replace(
    'Each row includes identifiers, centroid coordinates, ACS input features, spatial-lag features, geospatial enrichment features, target labels, coverage flags, geography-aware split assignments, and optional geometry.',
    'Each row includes identifiers, centroid coordinates, 33 ACS features, 37 geospatial enrichment features, 27 target labels, coverage flags, and geography-aware split assignments. Spatial-lag features are computed at runtime and are not stored in the parquet.'
)

# 3. feature_columns() comment
readme = readme.replace(
    'By default, `feature_columns(df)` should return the 63 solver-usable v24.0.1 features: 33 ACS features, 14 spatial-lag features, and 16 enrichment features. To reproduce the original v23.001 ACS-only baseline, use only columns beginning with `acs_`.',
    'By default, `feature_columns(df)` returns the 70 solver-usable v24.0.1 features: 33 ACS + 37 enrichment. For an ACS-only baseline, filter to columns beginning with `acs_`.'
)

# 4. Remove lag_acs_ from manual feature_cols
old_feature_cols = (
    'feature_cols = [\n'
    '    c for c in df_clean.columns\n'
    '    if c.startswith("acs_")\n'
    '    or c.startswith("lag_acs_")\n'
    '    or c.startswith("svi_")\n'
    '    or c.startswith("flood_")\n'
    '    or c.startswith("hifld_")\n'
    '    or c.startswith("drive_min_")\n'
    ']'
)
new_feature_cols = (
    'feature_cols = [\n'
    '    c for c in df_clean.columns\n'
    '    if c.startswith("acs_")\n'
    '    or c.startswith("svi_")\n'
    '    or c.startswith("flood_")\n'
    '    or c.startswith("nfip_")\n'
    '    or c.startswith("twi_")\n'
    '    or c.startswith("slope_")\n'
    '    or c.startswith("hifld_")\n'
    '    or c.startswith("drive_min_")\n'
    ']'
)
readme = readme.replace(old_feature_cols, new_feature_cols)

# 5. Input Features section
readme = readme.replace(
    'GeoRSCT v24.0.1 includes 63 solver-usable input features.',
    'GeoRSCT v24.0.1 includes 70 solver-usable input features: 33 ACS and 37 geospatial enrichment.'
)
old_lag_section = (
    '### 2. Spatial-Lag Features\n\n'
    'The 14 spatial-lag features are prefixed with `lag_acs_`. They are computed as queen-contiguity weighted neighbor means over selected ACS fields.\n\n'
    'These are **spatial lags, not time lags**. They encode neighboring-area context, not temporal history.'
)
new_lag_section = (
    '### 2. Spatial-Lag Features\n\n'
    'Spatial-lag features (`lag_acs_*`) are **not stored in the parquet**. They are computed at runtime from '
    '`zcta_adjacency.parquet` using queen-contiguity weighted neighbor means. To include them in training, '
    'call `compute_spatial_lags(df, acs_cols, adjacency)` from the build pipeline.'
)
readme = readme.replace(old_lag_section, new_lag_section)

readme = readme.replace(
    'The 16 enrichment features include:',
    'The 37 enrichment features include (prefixes: `svi_`, `flood_`, `nfip_`, `twi_`, `slope_`, `hifld_`, `drive_`):'
)
readme = readme.replace('the full 63-feature representation.', 'the full 70-feature representation.')
readme = readme.replace(
    'Reported benchmark comparisons should document whether they use ACS-only features or the full 70-feature representation.',
    'Reported benchmark comparisons should document whether they use ACS-only (33) or full (70) features.'
)

# 6. Files table
old_files = '## Files\n\n|---|---:|---|\n\nReplace the size placeholders after the final v24.0.1 files are uploaded.'
new_files = (
    '## Files\n\n'
    '| File | Size | Description |\n'
    '|------|-----:|-------------|\n'
    '| `georsct_table.parquet` | 17.8 MB | Main table: 31,789 ZCTAs x 106 columns (no geometry) |\n'
    '| `georsct_simplified_001.geoparquet` | ~66 MB | Same + ZCTA boundary polygons (EPSG:4326, 0.001 deg simplified) |\n'
    '| `cdc_places_ci.parquet` | 1.8 MB | CDC PLACES 95% CI sidecar: 44 fields |\n'
    '| `zcta_acs_margins_of_error.parquet` | 7.9 MB | ACS 5-year MOE sidecar: 33 fields |\n'
    '| `noaa_storm_events_long.parquet` | 1.1 MB | NOAA flood history 1996-2024: year-level rows for temporal experiments |'
)
readme = readme.replace(old_files, new_files)

# 7. Stale language
readme = readme.replace(
    'Known target coverage from the initial release remains:',
    'Target coverage:'
)
readme = readme.replace(
    'All build steps, joins, and validation checks for this initial public release are specified',
    'All build steps, joins, and validation checks are specified'
)

# 8. Version scheme
readme = readme.replace(
    'Version scheme: `{PLACES_vintage}.{release}`. For example, `24.0.1` means PLACES 2023, second internal build and first public release.',
    'Version scheme: `{major}.{minor}.{patch}`. v24.0.1 = first public release with v24 enrichment layers (NOAA, NFIP, TWI).'
)

# 9. Citation
readme = readme.replace('georsct2026v23002', 'georsct2026v24001')
readme = readme.replace(
    'note         = {31,789 U.S. ZCTAs, 63 solver-usable input features, 27 regression targets, fixed geography-aware evaluation protocols, and uncertainty sidecars}',
    'note         = {31,789 U.S. ZCTAs, 70 solver-usable input features (33 ACS + 37 enrichment), 27 regression targets, NOAA/NFIP/TWI flood layers, fixed geography-aware evaluation protocols, and uncertainty sidecars}'
)

# 10. Remove remaining broken table stubs
readme = re.sub(r'\n\n\|[-|: ]+\|\n', '\n', readme)
readme = re.sub(r'\n\|[-|: ]+\|\n', '\n', readme)

api.create_commit(
    repo_id=REPO,
    repo_type='dataset',
    operations=[CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=readme.encode('utf-8'))],
    commit_message='docs(readme): fix all v24.0.1 inconsistencies — counts, lags, files table, citation, stubs',
)
print('Done.')
