# S035-Model-Ladder: Solver Selection Rationale

## Design Constraint

The experiment tests **representation** differences (R0 vs R1 vs R2), not solver differences. Solvers are held constant across representation levels. The gap between solvers is a diagnostic signal (D.2 flattening), not the objective.

DOE constraint: no hyperparameter tuning per representation. Default parameters only.

## Selected Solvers

### HistGBDT — `sklearn.ensemble.HistGradientBoostingRegressor`

**Why:**
- Native NaN handling — 20-40% of columns have missing values; no imputation artifacts
- Fast on 400 rows (<1 sec train)
- Strong nonlinear baseline — if signal exists, gradient boosting finds it
- Built into sklearn — no extra wheel to vendor on SageMaker

**Parameters (frozen):**
```python
max_iter=200, max_depth=6, learning_rate=0.1, random_state=42
```

**For binary target (HWM):** `HistGradientBoostingClassifier` with same parameters.

### Ridge — `sklearn.linear_model.Ridge`

**Why:**
- Linear probe — if Ridge matches HistGBDT, signal is linear
- The gap between Ridge and HistGBDT is informative:
  - Small gap = representation ceiling binding (D.2 flattening active)
  - Large gap = nonlinear interactions matter, solver choice matters
- Closed-form solution — deterministic, no convergence issues

**Pipeline:**
```python
Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
```

**For binary target (HWM):** `RidgeClassifier` in same pipeline.

## Rejected Alternatives

| Solver | Why Rejected |
|--------|-------------|
| XGBoost | Extra dependency for marginal gain over HistGBDT at n=400. Tuning headroom is neutralized by DOE "no tuning" rule. |
| Random Forest | Slower, no NaN handling, no advantage over HistGBDT at this scale. |
| Neural net | Underdetermined at n=400. Would require regularization tuning, violating DOE constraint. |
| LightGBM | Same niche as HistGBDT. sklearn version avoids external dep. |
| GAM | Interesting but adds complexity without serving the representation-comparison goal. |

## Why Exactly 2 Solvers

Cross-solver spread is the D.2 flattening diagnostic. Two is the minimum to measure spread. Three adds estimation noise without adding information at n=400.

If both solvers converge to similar scores on R0, adding a third solver won't help — the representation is the bottleneck, not the solver. That's the D.2 signal.

## Assumptions to Revisit

1. **HistGBDT defaults are adequate.** If R0 baseline shows no skill (H1 FAIL), check whether defaults are too conservative before concluding the data has no signal.
2. **Ridge imputation (median) doesn't distort.** At 20-40% missingness, median imputation could flatten real variance. If Ridge underperforms suspiciously, compare with HistGBDT feature importances to see if imputed columns are driving predictions.
3. **No interaction between solver choice and representation level.** The DOE assumes uplift(R1-R0) is solver-independent. If HistGBDT shows uplift but Ridge doesn't (or vice versa), that's a finding — the representation interacts with solver capacity.
4. **200 iterations is enough for HistGBDT.** At n=400 with max_depth=6, overfitting is the risk, not underfitting. Monitor train-vs-val gap.
5. **alpha=1.0 for Ridge is reasonable.** No cross-validation of alpha — this is intentional. If Ridge scores are pathologically bad, alpha might be wrong, but the DOE says don't tune.
