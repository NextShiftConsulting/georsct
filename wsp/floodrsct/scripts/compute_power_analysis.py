#!/usr/bin/env python3
"""Post-hoc power analysis for H2a pooled Wilcoxon signed-rank test.

Addresses PhD benchmark gap #5: contextualizes the INCONCLUSIVE H2a verdict
by distinguishing "underpowered" from "truly null."

The H2a test pools fold-level metric deltas (R1 - R0) across all regression
cells and applies a one-sided Wilcoxon signed-rank test.  The pooled result
was p=0.114, d=0.27 — INCONCLUSIVE.  A decomposition shows classification
cells pass (p=0.004, d=0.50) while regression cells are heterogeneous.

This script computes:
  - Achieved power at observed effect sizes with actual sample sizes
  - Required N for 80% power at the observed effect
  - Minimum detectable effect at current N with 80% power
  - Interpretation of whether underpowered vs. target-type heterogeneity

Uses the normal approximation for the Wilcoxon signed-rank test with
asymptotic relative efficiency (ARE) correction: for continuous symmetric
distributions, Wilcoxon has ARE = pi/3 ~ 0.955 relative to the paired t-test.
The effective sample size for power purposes is n_eff = n * ARE.

Output: results/s035/power_analysis.json
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower


# ---------------------------------------------------------------------------
# Constants from the experiment
# ---------------------------------------------------------------------------

# Wilcoxon ARE relative to paired t-test (for continuous symmetric data)
WILCOXON_ARE: float = math.pi / 3  # ~0.955

# From per_target_h2_breakdown.json:
# Regression cells: houston, sw_florida, nyc, riverside — 4 cells x 5 folds
# Classification cells: houston/obs_has_311, sw_florida/obs_has_hwm,
#   nyc/obs_has_311, nyc/obs_has_hwm — 4 cells x 5 folds
# Some cells may have fewer folds (Houston has 4 events, but folds are
# spatial CV folds, not events). The breakdown reports fold_d per cell,
# implying 5 folds per cell consistently.

N_FOLDS_PER_CELL: int = 5
N_REGRESSION_CELLS: int = 4
N_CLASSIFICATION_CELLS: int = 4

# Observed from the pooled test results
OBSERVED_D_POOLED: float = 0.27
OBSERVED_P_POOLED: float = 0.1137
OBSERVED_D_CLASSIFICATION: float = 0.497
OBSERVED_P_CLASSIFICATION: float = 0.0036

ALPHA: float = 0.05


# ---------------------------------------------------------------------------
# Power computation functions
# ---------------------------------------------------------------------------

def wilcoxon_achieved_power(
    n_paired: int,
    cohens_d: float,
    alpha: float = ALPHA,
    alternative: str = "larger",
) -> float:
    """Compute achieved power for Wilcoxon signed-rank via normal approximation.

    Args:
        n_paired: Number of paired observations.
        cohens_d: Observed standardized effect size (Cohen's d).
        alpha: Significance level.
        alternative: Direction of test ("larger" for one-sided).

    Returns:
        Achieved statistical power (probability of rejecting H0).
    """
    # Effective n after ARE correction
    n_eff = n_paired * WILCOXON_ARE

    # Use NormalIndPower as proxy: for a one-sample z-test,
    # power = P(Z > z_alpha - d*sqrt(n_eff))
    # This is equivalent to the paired t-test power with n_eff
    solver = NormalIndPower()
    # NormalIndPower.solve_power uses "ratio" as n2/n1 for two-sample;
    # for one-sample equivalent, we use nobs1 = n_eff directly
    # with effect_size = d (standardized mean / sd)
    power = solver.solve_power(
        effect_size=cohens_d,
        nobs1=n_eff,
        alpha=alpha,
        alternative=alternative,
        power=None,
    )
    return float(power)


def wilcoxon_required_n(
    cohens_d: float,
    power: float = 0.80,
    alpha: float = ALPHA,
    alternative: str = "larger",
) -> int:
    """Compute required N for target power at given effect size.

    Args:
        cohens_d: Target standardized effect size.
        power: Desired power level.
        alpha: Significance level.
        alternative: Direction of test.

    Returns:
        Required number of paired observations (rounded up, pre-ARE).
    """
    solver = NormalIndPower()
    n_eff = solver.solve_power(
        effect_size=cohens_d,
        nobs1=None,
        alpha=alpha,
        power=power,
        alternative=alternative,
    )
    # Convert effective n back to actual n (divide by ARE)
    n_actual = math.ceil(float(n_eff) / WILCOXON_ARE)
    return n_actual


def wilcoxon_minimum_detectable_effect(
    n_paired: int,
    power: float = 0.80,
    alpha: float = ALPHA,
    alternative: str = "larger",
) -> float:
    """Compute minimum detectable effect size at given N and power.

    Args:
        n_paired: Number of paired observations.
        power: Target power level.
        alpha: Significance level.
        alternative: Direction of test.

    Returns:
        Minimum detectable Cohen's d.
    """
    n_eff = n_paired * WILCOXON_ARE
    solver = NormalIndPower()
    d = solver.solve_power(
        effect_size=None,
        nobs1=n_eff,
        alpha=alpha,
        power=power,
        alternative=alternative,
    )
    return float(d)


def interpret_power(
    achieved_power: float,
    n_paired: int,
    n_required: int,
    mde: float,
    observed_d: float,
) -> str:
    """Generate interpretation string for the power analysis.

    Args:
        achieved_power: Computed power at observed effect.
        n_paired: Actual sample size.
        n_required: Sample size needed for 80% power.
        mde: Minimum detectable effect at current N.
        observed_d: Observed Cohen's d.

    Returns:
        Interpretation string.
    """
    if achieved_power >= 0.80:
        return (
            f"Adequately powered (power={achieved_power:.2f}). "
            f"The INCONCLUSIVE result reflects a genuine absence of "
            f"a strong pooled effect, not insufficient sample size."
        )
    elif achieved_power >= 0.50:
        return (
            f"Moderately underpowered (power={achieved_power:.2f}). "
            f"The test had a {achieved_power*100:.0f}% chance of detecting "
            f"the observed effect d={observed_d:.2f}. "
            f"Would need N={n_required} paired observations for 80% power. "
            f"The MDE at current N is d={mde:.2f}."
        )
    else:
        return (
            f"Substantially underpowered (power={achieved_power:.2f}). "
            f"The test had only a {achieved_power*100:.0f}% chance of detecting "
            f"d={observed_d:.2f}. Need N={n_required} for 80% power "
            f"(current N={n_paired}). MDE at current N is d={mde:.2f}."
        )


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_power_analysis() -> dict[str, Any]:
    """Run the full post-hoc power analysis.

    Returns:
        Dictionary with power analysis results for pooled and classification
        subsets, plus overall conclusion.
    """
    # Pooled regression: 4 cells x 5 folds = 20 paired observations
    n_reg = N_REGRESSION_CELLS * N_FOLDS_PER_CELL
    power_reg = wilcoxon_achieved_power(n_reg, OBSERVED_D_POOLED)
    n_req_reg = wilcoxon_required_n(OBSERVED_D_POOLED)
    mde_reg = wilcoxon_minimum_detectable_effect(n_reg)

    # Classification: 4 cells x 5 folds = 20 paired observations
    n_clf = N_CLASSIFICATION_CELLS * N_FOLDS_PER_CELL
    power_clf = wilcoxon_achieved_power(n_clf, OBSERVED_D_CLASSIFICATION)
    n_req_clf = wilcoxon_required_n(OBSERVED_D_CLASSIFICATION)
    mde_clf = wilcoxon_minimum_detectable_effect(n_clf)

    # All 8 cells pooled (the reported p=0.114 test)
    # This pooled ALL fold deltas: 4 reg + 4 clf = 8 cells x 5 folds = 40
    # But the reported pooled_h2 uses regression cells only per the code.
    # The per_target_h2_breakdown.json reports pooled p=0.1137 as the
    # regression-only pooled test. Classification is reported separately.
    # So n_pooled = n_reg = 20 for the INCONCLUSIVE result.
    n_pooled = n_reg

    pooled_result = {
        "n_paired": n_pooled,
        "n_cells": N_REGRESSION_CELLS,
        "n_folds_per_cell": N_FOLDS_PER_CELL,
        "observed_d": OBSERVED_D_POOLED,
        "observed_p": OBSERVED_P_POOLED,
        "alpha": ALPHA,
        "wilcoxon_are": round(WILCOXON_ARE, 4),
        "effective_n": round(n_pooled * WILCOXON_ARE, 1),
        "achieved_power": round(power_reg, 4),
        "n_required_80pct": n_req_reg,
        "minimum_detectable_d_80pct": round(mde_reg, 4),
        "interpretation": interpret_power(
            power_reg, n_pooled, n_req_reg, mde_reg, OBSERVED_D_POOLED,
        ),
    }

    clf_result = {
        "n_paired": n_clf,
        "n_cells": N_CLASSIFICATION_CELLS,
        "n_folds_per_cell": N_FOLDS_PER_CELL,
        "observed_d": OBSERVED_D_CLASSIFICATION,
        "observed_p": OBSERVED_P_CLASSIFICATION,
        "alpha": ALPHA,
        "effective_n": round(n_clf * WILCOXON_ARE, 1),
        "achieved_power": round(power_clf, 4),
        "n_required_80pct": n_req_clf,
        "minimum_detectable_d_80pct": round(mde_clf, 4),
        "interpretation": interpret_power(
            power_clf, n_clf, n_req_clf, mde_clf, OBSERVED_D_CLASSIFICATION,
        ),
    }

    # Determine overall conclusion
    if power_reg < 0.80 and power_clf >= 0.80:
        conclusion = (
            "The INCONCLUSIVE pooled H2a result (regression cells only) reflects "
            "genuine underpowering at d=0.27 with N=20 paired folds. However, the "
            "decomposition reveals the deeper issue: target-type heterogeneity. "
            "Classification cells show a large, well-powered effect (d=0.50, "
            f"power={power_clf:.2f}), while regression cells have a small, "
            "heterogeneous effect diluted by one negative cell (riverside). "
            "The pooled test conflates two distinct response profiles. "
            "Collecting more regression folds would increase power, but the "
            "scientific conclusion is already clear: graph context reliably helps "
            "observation-based classification, and conditionally helps claims "
            "regression depending on scenario data quality."
        )
    elif power_reg >= 0.80:
        conclusion = (
            "The pooled regression test is adequately powered. The INCONCLUSIVE "
            "result reflects a genuinely weak or absent pooled effect across "
            "heterogeneous regression targets, not insufficient sample size."
        )
    else:
        conclusion = (
            "Both subsets are underpowered. The experiment would benefit from "
            "additional scenarios or cross-validation folds to resolve H2a."
        )

    return {
        "analysis": "post_hoc_power_analysis",
        "purpose": "Contextualize INCONCLUSIVE H2a verdict (PhD benchmark gap #5)",
        "method": (
            "Normal approximation for Wilcoxon signed-rank power with "
            "ARE correction (efficiency=pi/3 relative to paired t-test)"
        ),
        "pooled_h2a": pooled_result,
        "classification_only": clf_result,
        "conclusion": conclusion,
        "recommendations": {
            "for_paper": (
                "Report achieved power alongside the INCONCLUSIVE verdict. "
                "The decomposition by target type is the primary explanatory "
                "finding — the pooled test mixes two distinct response profiles."
            ),
            "for_future_work": (
                f"To achieve 80% power at d=0.27 for regression cells, "
                f"N={n_req_reg} paired fold observations are needed "
                f"(currently N={n_pooled}). This requires "
                f"{math.ceil(n_req_reg / N_FOLDS_PER_CELL)} regression cells "
                f"(scenarios x targets) vs current {N_REGRESSION_CELLS}."
            ),
        },
    }


def main() -> int:
    """Run power analysis and save results."""
    result = compute_power_analysis()

    out_dir = Path(__file__).parent / "results" / "s035"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "power_analysis.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written: {out_path}")
    print(f"\n--- POWER ANALYSIS SUMMARY ---")
    print(f"  Pooled H2a (regression):")
    print(f"    N={result['pooled_h2a']['n_paired']}, d={OBSERVED_D_POOLED}, "
          f"power={result['pooled_h2a']['achieved_power']:.3f}")
    print(f"    N needed for 80% power: {result['pooled_h2a']['n_required_80pct']}")
    print(f"    MDE at current N: d={result['pooled_h2a']['minimum_detectable_d_80pct']:.3f}")
    print(f"\n  Classification subset:")
    print(f"    N={result['classification_only']['n_paired']}, "
          f"d={OBSERVED_D_CLASSIFICATION}, "
          f"power={result['classification_only']['achieved_power']:.3f}")
    print(f"\n  Conclusion: {result['conclusion'][:120]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
