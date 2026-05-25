"""Tests for ADR-021 diagnostic simplex ratio metrics.

Validates:
  - alpha remains R/(R+N) (canonical lock)
  - SCR_norm = R/(R+S_sup) detects clutter shifts
  - CNR_norm = S_sup/(S_sup+N) separates noise-bound from clutter-bound
  - All values bounded [0, 1]
  - Leaderboard scalar L = R + S_sup is preserved when SCR_norm differs
  - No gate enforcement dependency
"""

import sys
from pathlib import Path

import pytest

# Add evidence/diagnostics to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "evidence" / "diagnostics"))

from certificate_issuer import compute_simplex_diagnostic_ratios


def test_simplex_ratio_alpha_is_canonical():
    """alpha must equal R/(R+N) — the patent canonical definition."""
    R, S_sup, N = 0.3, 0.1, 0.6
    ratios = compute_simplex_diagnostic_ratios(R, S_sup, N)
    assert ratios["alpha"] == pytest.approx(R / (R + N))


def test_simplex_ratio_bounds():
    """All diagnostic ratios must be in [0, 1]."""
    R, S_sup, N = 0.3, 0.1, 0.6
    ratios = compute_simplex_diagnostic_ratios(R, S_sup, N)
    for key, value in ratios.items():
        if key != "leaderboard_scalar":
            assert 0.0 <= value <= 1.0, f"{key} = {value} out of bounds"


def test_scr_norm_detects_clutter_shift():
    """SCR_norm must distinguish R-heavy from S_sup-heavy configurations."""
    e1 = compute_simplex_diagnostic_ratios(0.30, 0.10, 0.60)
    e2 = compute_simplex_diagnostic_ratios(0.10, 0.30, 0.60)

    # Same leaderboard scalar
    assert (0.30 + 0.10) == pytest.approx(0.10 + 0.30)

    # Different SCR_norm
    assert e1["SCR_norm"] == pytest.approx(0.75)
    assert e2["SCR_norm"] == pytest.approx(0.25)


def test_worked_example_does_not_preserve_alpha():
    """The ADR-021 worked example: same L, different alpha."""
    e1 = compute_simplex_diagnostic_ratios(0.30, 0.10, 0.60)
    e2 = compute_simplex_diagnostic_ratios(0.10, 0.30, 0.60)

    assert e1["alpha"] == pytest.approx(0.333333, rel=1e-4)
    assert e2["alpha"] == pytest.approx(0.142857, rel=1e-4)
    assert e1["alpha"] != pytest.approx(e2["alpha"])


def test_leaderboard_scalar_preserved_in_worked_example():
    """L = R + S_sup must be identical for both configurations."""
    e1 = compute_simplex_diagnostic_ratios(0.30, 0.10, 0.60)
    e2 = compute_simplex_diagnostic_ratios(0.10, 0.30, 0.60)

    assert e1["leaderboard_scalar"] == pytest.approx(0.40)
    assert e2["leaderboard_scalar"] == pytest.approx(0.40)
    assert e1["leaderboard_scalar"] == pytest.approx(e2["leaderboard_scalar"])


def test_cnr_norm_separates_clutter_from_noise():
    """CNR_norm must be high when S_sup dominates residual, low when N dominates."""
    # Clutter-bound: S_sup >> N in the non-relevant portion
    clutter_bound = compute_simplex_diagnostic_ratios(0.10, 0.70, 0.20)
    # Noise-bound: N >> S_sup in the non-relevant portion
    noise_bound = compute_simplex_diagnostic_ratios(0.10, 0.20, 0.70)

    assert clutter_bound["CNR_norm"] > 0.7
    assert noise_bound["CNR_norm"] < 0.3


def test_degenerate_inputs():
    """Edge cases: zero denominators should not crash."""
    # All R (no noise, no clutter)
    ratios = compute_simplex_diagnostic_ratios(1.0, 0.0, 0.0)
    assert ratios["alpha"] == pytest.approx(1.0)
    assert ratios["SCR_norm"] == pytest.approx(1.0)

    # All N (no signal, no clutter)
    ratios = compute_simplex_diagnostic_ratios(0.0, 0.0, 1.0)
    assert ratios["alpha"] == pytest.approx(0.0)
    assert ratios["SCR_norm"] == pytest.approx(0.0)


def test_two_dof_caveat():
    """Any two ratios determine the third (2-DoF simplex constraint)."""
    R, S_sup, N = 0.4, 0.35, 0.25
    ratios = compute_simplex_diagnostic_ratios(R, S_sup, N)

    # Reconstruct R, S_sup, N from alpha and SCR_norm
    alpha = ratios["alpha"]
    scr = ratios["SCR_norm"]

    # alpha = R/(R+N), scr = R/(R+S_sup), R+S_sup+N=1
    # From alpha: N = R*(1-alpha)/alpha
    # From scr: S_sup = R*(1-scr)/scr
    # From simplex: R + S_sup + N = 1
    # => R + R*(1-scr)/scr + R*(1-alpha)/alpha = 1
    # => R * (1 + (1-scr)/scr + (1-alpha)/alpha) = 1
    # => R * (1/scr + 1/alpha - 1) = 1
    R_reconstructed = 1.0 / (1.0 / scr + 1.0 / alpha - 1.0)
    assert R_reconstructed == pytest.approx(R, rel=1e-6)
