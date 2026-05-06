# Series 019 Representation Artifact Naming Convention

**Overrides** `FILE_NAMING_CONVENTION.md` (repo root) for representation artifacts in this series.

---

## Two Artifact Types

| Suffix | Content | Use Pattern | Required Keys |
|--------|---------|-------------|---------------|
| `*_transform.npz` | Scaler params + PCA components | Apply at runtime to raw features | `scaler_mean`, `scaler_scale`, `components`, `component_mean`, `feature_schema` |
| `*_latents.npz` | Pre-computed embedding array | Load directly as Z | `Z`, `zcta_id` |

The suffix (`_transform` vs `_latents`) tells the loader HOW to use the file.

---

## Canonical Names (Series 019+)

| Embedding | Canonical Filename | Type |
|-----------|--------------------|------|
| pca_v1 | `pca32_v1_transform.npz` | transform |
| spatial_lag_v1 | `spatial_lag_v1_transform.npz` | transform |
| gnn_v2 | `gnn_v2_latents.npz` | latents |

---

## Legacy Names (Series 018 / Pre-existing S3)

These files already exist on S3 and will NOT be renamed. The loader handles them transparently.

| Embedding | Legacy Filename | Key Differences |
|-----------|----------------|-----------------|
| pca_v1 | `pca32_v1.npz` | Keys: `pca_components`, `pca_mean` (not `components`, `component_mean`) |
| spatial_lag_v1 | `spatial_lag_v1.npz` | Keys: `pca_components`, `pca_mean` |
| gnn_v2 | `zcta_latents_v1.npz` | Key: `latents` (not `Z`); ID key: `zcta_id` |

---

## Loader Resolution Order

1. Try canonical name first (`*_transform.npz` / `*_latents.npz`)
2. Fall back to legacy name
3. If neither found, fit locally (PCA) or error (GNN)

Key name fallbacks within a file:
- `components` -> `pca_components`
- `component_mean` -> `pca_mean`
- `Z` -> `latents`

---

## For New Artifacts

When creating new representation artifacts for this series:

1. Use the canonical naming: `{family}_{version}_{type}.npz`
2. Always include the type suffix (`_transform` or `_latents`)
3. Use standardized keys (`components`, `component_mean`, `Z`, `zcta_id`)
4. Include `feature_schema` in all transform artifacts
5. Include `zcta_id` in all latent artifacts (for alignment verification)

---

## S3 Location

```
s3://swarm-yrsn-datasets/rsct_curriculum/series_018/artifacts/representations/
```

Both legacy and canonical names coexist in the same prefix. The loader tries both.
