# F001-KAPPA-VS-SIMPLEX: Scalar kappa vs R/S/N Simplex

Tests whether the R/S/N simplex decomposition provides evaluation information
that scalar kappa alone cannot. Title-level claim for the paper.

## Claims

- C1: Representation adequacy is not scalar
- C2: R/S/N decomposition exposes failure modes hidden by scalar kappa

## Four Tests

| # | Test | Pass Condition |
|---|------|----------------|
| H1 | Rank correlation | rho_alpha != rho_kappa |
| H2 | Calibration bins | simplex bins tighter than kappa quartiles |
| H3 | Error separation | kappa-matched pairs show confusion distance |
| H4 | Inversion detection | simplex detects inversions kappa misses |

## Running

```bash
# Analytical experiment — no SageMaker needed
python run_f001.py
```

## Input Data

Text leaderboard data from MIRACL evaluation (16 embedding families).
Existing certificates, confusion matrices, and OOF predictions.

## Circuit Breaker

If scalar kappa >= simplex on ALL four metrics:
- C1 weak: soften title
- C1 null: slip deadline
- C1 reverse: negative result paper
