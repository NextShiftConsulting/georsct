# Findings â€” nemotron

## Serious
- file: MMAR_Input.md:18
  severity: serious
  category: design-flaw
  description: The TJEPA encoder's latent representations are not exported in the s037 results, containing only y_true/y_pred columns. This technical blockade prevents the core A5 analysis (cosine similarity, PCA) between Prithvi and TJEPA embeddings, rendering the experiment infeasible as designed without modifying the job script to serialize encoder outputs.
  quote: |
    Blocker 1 (technical): s037 TJEPA predictions parquet contains only `y_true`/`y_pred` columns â€” no latent embeddings are serialized. The TJEPA encoder produces 128-dim internal representations during training but the job script (`run_vertical_slice.py`) does not export them. Computing cosine similarity between Prithvi 1024-dim embeddings and 1-dim prediction residuals is type-invalid.

- file: MMAR_Input.md:28
  severity: serious
  category: design-flaw
  description: The original A5 hypothesis is scientifically moot because A1 demonstrated Prithvi-EO-2.0 embeddings add no predictive value beyond R0 tabular features (0/5 scenarios show R2 improvement >= 0.02). Since TJEPA learns a 128-dim representation strictly from the same R0 tabular features, Prithvi-TJEPA redundancy is implied, making fusion testing redundant with established A1 results.
  quote: |
    Blocker 2 (scientific): The DOE stated: "Defer until A1 results are known. If Prithvi adds no predictive value (A1 FAIL), fusion is moot."
    ...
    Key Observation About TJEPA
    TJEPA learns from R0 tabular features â€” the SAME features that A1 showed are already redundant with Prithvi. So the original A5 question ("are Prithvi and TJEPA redundant?") has a simpler answer:
    - Prithvi captures satellite land-cover information
    - TJEPA captures learned structure in R0 tabular features
    - A1 showed Prithvi is redundant WITH R0 tabular features
    - Therefore Prithvi is likely redundant with TJEPA (which IS a function of R0 features)

- file: MMAR_Input.md:142
  severity: serious
  category: domain-limitation
  description: All cross-analysis failures (A1, A3, A4) trace to fundamental domain heterogeneity across flood regimes, invalidating the assumption of a universal flood prediction model. The model ladder's performance is metro-specific, with transfer learning failing in 18/20 scenario pairs and feature importance showing near-zero stability (Kendall's tau range: -0.206 to 0.323), necessitating regime-specific modeling approaches.
  quote: |
    Cross-Cutting Theme
    All failures trace to domain heterogeneity across flood regimes. The model ladder is regime-specific. "One model fits all" doesn't work for flood prediction across US metros.

## Minor
- file: MMAR_Input.md:11095
  severity: minor
  category: interpretation-limitation
  description: The persistent importance of nfip_historical_frequency across scenarios (appearing in top-3 of >= 3 scenarios) reveals a circular prediction problem where historical claims are used to predict future claims, limiting the model's ability to capture evolving risk factors and potentially inflating performance metrics unrealistically.
  quote: |
    Only nfip_historical_frequency appears in top-3 of >= 3 scenarios
    ...
    The circular predictor problem (NFIP history predicting NFIP claims)

- file: MMAR_Input.md:042
  severity: minor
  category: opportunity
  description: Riverside's consistent failure as a transfer target (all R2 deeply negative, down to -43) and orthogonal top features (longitude, latitude, vacancy) identify it as a natural out-of-distribution test case for evaluating model generalization to inland, arid flood regimes distinct from coastal metros.
  quote: |
    Riverside is catastrophically bad as a transfer target (all R2 deeply negative, down to -43)
    ...
    Riverside top features (longitude, latitude, vacancy) are completely orthogonal to coastal metros

- file: MMAR_Input.md:128
  severity: minor
  category: thematic-connection
  description: The latest ADRs' emphasis on domain-specific measurement (ADR-054: memristor measurement probe, ADR-058: per-domain oobleck coupling sign calibration) directly parallels and reinforces the cross-analysis finding that flood prediction requires regime-specific approaches, validating the ADR framework's focus on contextual adaptation.
  quote: |
    Cross-Cutting ADR Theme
    The latest ADRs all focus on domain-specific measurement â€” calibrating the RSCT framework per-domain rather than assuming universal behavior. This directly parallels the cross-analysis finding that flood prediction is regime-specific, not universal.
