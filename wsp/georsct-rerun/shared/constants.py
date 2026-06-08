"""
constants.py -- Shared constants for series_019 experiments.

Series 019 runs canonical RSCT certification with:
  - compute_sigma_request(N) for per-request turbulence
  - SequentialGatekeeper with ADR-024 enforcement provenance
  - Path A (MLP tercile classifier -> softmax -> aggregate_scores_from_probs)
"""

# Inherited from series_018 -- same benchmark, same tasks.
CONUS27_TASKS = [
    "annual_checkup", "arthritis", "asthma", "binge_drinking", "bp_medicated",
    "cancer", "cholesterol_screening", "chronic_kidney_disease", "copd",
    "coronary_heart_disease", "dental_visit", "diabetes", "elevation",
    "high_blood_pressure", "high_cholesterol", "home_value", "income",
    "mental_health_not_good", "night_lights", "obesity",
    "physical_health_not_good", "physical_inactivity", "population_density",
    "sleep_less_7hr", "smoking", "stroke", "tree_cover",
]

# S019A targets: three regimes spanning the invariance gradient.
S019A_TARGETS = ["diabetes", "population_density", "elevation"]

# Embedding families: same three as Table 1 in the paper.
EMBEDDINGS = ["pca_v1", "spatial_lag_v1", "gnn_v2"]

# Solver families for S019A (S019B uses HistGBDT only).
S019A_SOLVERS = ["histgbdt", "ridge", "mlp"]

# CV protocol
N_FOLDS = 5
SEED = 42
