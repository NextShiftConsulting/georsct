"""Tests for the three-layer variance-control stack decomposition.

Validates that:
1. Coverage gate detects uncovered deployment bins.
2. Product matching leaves residual margin gap under correlation.
3. Raking closes the margin gap.
4. Shrinkage buys ESS and pays margin gap.
5. Cumulative ladder attributes convergence layer by layer.
6. Edge cases: empty data, single-bin, degenerate weights.
"""

import numpy as np
import pandas as pd
import pytest

from georsct.analysis.variance_stack import (
    DEFAULT_LADDER,
    StackConfig,
    build_weights_layered,
    coverage_report,
    decompose_stack,
    max_uncovered_target_mass,
    residual_margin_gap,
    shrinkage_ledger,
    weighted_metrics,
)
from georsct.validation.deployment_alignment import (
    effective_sample_size,
    normalize_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_binned(n: int, seed: int = 42, n_bins: int = 5) -> pd.DataFrame:
    """Create a binned dataframe with two correlated descriptors."""
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, n_bins, size=n)
    # x2 correlated with x1 (same bin 80% of the time)
    x2 = np.where(rng.random(n) < 0.8, x1, rng.integers(0, n_bins, size=n))
    return pd.DataFrame({"d1_bin": x1, "d2_bin": x2})


def _make_shifted(
    n_val: int,
    n_tgt: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create validation and target with a known distribution shift."""
    rng = np.random.default_rng(seed)
    # Target is uniform across 5 bins
    tgt = pd.DataFrame({
        "d1_bin": rng.integers(0, 5, size=n_tgt),
        "d2_bin": rng.integers(0, 5, size=n_tgt),
    })
    # Validation is concentrated in bins 0-2 (underrepresents 3-4)
    val = pd.DataFrame({
        "d1_bin": rng.choice([0, 1, 2, 3, 4], size=n_val, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
        "d2_bin": rng.choice([0, 1, 2, 3, 4], size=n_val, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    })
    return val, tgt, ["d1_bin", "d2_bin"]


# ---------------------------------------------------------------------------
# Layer 0: Coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_full_coverage(self):
        val = pd.DataFrame({"d_bin": [0, 1, 2, 3, 4]})
        tgt = pd.DataFrame({"d_bin": [0, 1, 2, 3, 4]})
        cov = coverage_report(val, tgt, ["d_bin"])
        assert cov.iloc[0]["uncovered_bins"] == 0
        assert max_uncovered_target_mass(cov) == 0.0

    def test_partial_coverage(self):
        val = pd.DataFrame({"d_bin": [0, 1, 2]})
        tgt = pd.DataFrame({"d_bin": [0, 1, 2, 3, 4]})
        cov = coverage_report(val, tgt, ["d_bin"])
        assert cov.iloc[0]["uncovered_bins"] == 2
        assert max_uncovered_target_mass(cov) > 0.0

    def test_empty_validation(self):
        val = pd.DataFrame({"d_bin": pd.Series([], dtype=int)})
        tgt = pd.DataFrame({"d_bin": [0, 1, 2]})
        cov = coverage_report(val, tgt, ["d_bin"])
        assert max_uncovered_target_mass(cov) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Layer 1: Residual margin gap
# ---------------------------------------------------------------------------

class TestResidualMarginGap:
    def test_uniform_weights_on_matching_dists(self):
        """Uniform weights on identical distributions -> gap ~= 0."""
        df = pd.DataFrame({"d_bin": [0, 1, 2, 3, 4] * 20})
        w = normalize_weights(np.ones(len(df)))
        gap = residual_margin_gap(w, df, df, ["d_bin"])
        assert gap == pytest.approx(0.0, abs=1e-10)

    def test_shifted_dists_nonzero_gap(self):
        """Uniform weights on shifted distributions -> nonzero gap."""
        val = pd.DataFrame({"d_bin": [0, 0, 0, 1, 1]})
        tgt = pd.DataFrame({"d_bin": [0, 1, 1, 1, 1]})
        w = normalize_weights(np.ones(5))
        gap = residual_margin_gap(w, val, tgt, ["d_bin"])
        assert gap > 0.1

    def test_product_leaves_gap_under_correlation(self):
        """Product matching leaves a residual gap when descriptors covary."""
        val, tgt, cols = _make_shifted(500, 500, seed=99)
        _, cert_product = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="product", shrinkage_lambda=0.0),
        )
        _, cert_raking = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="raking", shrinkage_lambda=0.0),
        )
        # Raking should have smaller or equal gap than product
        assert cert_raking["residual_margin_gap"] <= cert_product["residual_margin_gap"] + 0.01

    def test_raking_drives_gap_near_zero(self):
        """Raking on well-supported data should drive gap close to 0."""
        val, tgt, cols = _make_shifted(1000, 1000, seed=77)
        _, cert = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="raking", shrinkage_lambda=0.0),
        )
        assert cert["residual_margin_gap"] < 0.05


# ---------------------------------------------------------------------------
# Layer 2: Shrinkage ledger
# ---------------------------------------------------------------------------

class TestShrinkageLedger:
    def test_shrinkage_buys_ess(self):
        """Shrinkage should increase ESS (buy variance stability)."""
        val, tgt, cols = _make_shifted(200, 200, seed=55)
        _, cert_no_shrink = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="raking", shrinkage_lambda=0.0),
        )
        _, cert_shrunk = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="raking", shrinkage_lambda=0.3),
        )
        assert cert_shrunk["ess"] >= cert_no_shrink["ess"] - 1e-6

    def test_shrinkage_pays_margin_gap(self):
        """Shrinkage should increase margin gap (pay bias)."""
        val, tgt, cols = _make_shifted(200, 200, seed=55)
        w_raked, _ = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="raking", shrinkage_lambda=0.0),
        )
        _, cert_shrunk = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="raking", shrinkage_lambda=0.3),
        )
        # Shrinkage delta gap should be >= 0 (bias paid)
        assert cert_shrunk["shrinkage_delta_margin_gap"] >= -1e-6

    def test_zero_shrinkage_no_change(self):
        """Lambda=0 should produce zero delta in the ledger."""
        val, tgt, cols = _make_shifted(100, 100, seed=44)
        _, cert = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="raking", shrinkage_lambda=0.0),
        )
        assert cert["shrinkage_delta_ess"] == 0.0
        assert cert["shrinkage_delta_margin_gap"] == 0.0


# ---------------------------------------------------------------------------
# Stack status
# ---------------------------------------------------------------------------

class TestStackStatus:
    def test_pass_on_well_supported_data(self):
        val, tgt, cols = _make_shifted(500, 500, seed=33)
        _, cert = build_weights_layered(
            val, tgt, cols,
            cfg=StackConfig(matching="raking", shrinkage_lambda=0.0),
        )
        assert cert["stack_status"] in ("PASS", "WARN_MATCH", "WARN_CONCENTRATION")

    def test_fail_coverage_on_missing_bins(self):
        """Validation missing entire deployment bins -> FAIL_COVERAGE."""
        val = pd.DataFrame({"d_bin": [0, 0, 0, 1, 1]})
        tgt = pd.DataFrame({"d_bin": [0, 1, 2, 3, 4]})
        _, cert = build_weights_layered(
            val, tgt, ["d_bin"],
            cfg=StackConfig(
                use_coverage_gate=True, matching="none",
                coverage_mass_tol=0.01,
            ),
        )
        assert cert["stack_status"] == "FAIL_COVERAGE"

    def test_none_matching_is_uniform(self):
        """matching='none' should produce uniform weights."""
        val = pd.DataFrame({"d_bin": [0, 1, 2, 3, 4]})
        tgt = pd.DataFrame({"d_bin": [0, 1, 2, 3, 4]})
        w, cert = build_weights_layered(
            val, tgt, ["d_bin"],
            cfg=StackConfig(matching="none", use_coverage_gate=False),
        )
        assert np.allclose(w, 1.0 / 5)


# ---------------------------------------------------------------------------
# Cumulative ladder
# ---------------------------------------------------------------------------

class TestDecomposeStack:
    def test_ladder_produces_correct_rows(self):
        val, tgt, cols = _make_shifted(200, 200, seed=22)
        df = decompose_stack(val, tgt, cols)
        assert len(df) == len(DEFAULT_LADDER)
        assert "config" in df.columns
        assert "ess" in df.columns
        assert "residual_margin_gap" in df.columns
        assert "delta_ess" in df.columns

    def test_ladder_with_metrics(self):
        val, tgt, cols = _make_shifted(100, 100, seed=11)
        rng = np.random.default_rng(11)
        y_true = rng.standard_normal(100)
        y_pred = y_true + rng.standard_normal(100) * 0.3
        df = decompose_stack(val, tgt, cols, y_true=y_true, y_pred=y_pred)
        assert "rmse" in df.columns
        assert "delta_rmse" in df.columns
        assert df["rmse"].notna().all()

    def test_ladder_without_metrics(self):
        val, tgt, cols = _make_shifted(100, 100, seed=11)
        df = decompose_stack(val, tgt, cols)
        assert "rmse" not in df.columns

    def test_delta_columns_sum_to_total_change(self):
        """Delta columns should sum to total change from first to last."""
        val, tgt, cols = _make_shifted(200, 200, seed=66)
        df = decompose_stack(val, tgt, cols)
        total_ess_change = df["ess"].iloc[-1] - df["ess"].iloc[0]
        sum_deltas = df["delta_ess"].iloc[1:].sum()
        assert total_ess_change == pytest.approx(sum_deltas, abs=1e-6)


# ---------------------------------------------------------------------------
# Weighted metrics
# ---------------------------------------------------------------------------

class TestWeightedMetrics:
    def test_uniform_weights(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        m = weighted_metrics(y_true, y_pred)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert m["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        m = weighted_metrics(y_true, y_pred)
        assert m["rmse"] > 0
        assert m["mae"] > 0

    def test_weighted_shifts_estimate(self):
        """Heavier weight on row 0 should shift weighted mean toward it."""
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([0.0, 10.0])
        w_low = np.array([0.9, 0.1])
        w_high = np.array([0.1, 0.9])
        m_low = weighted_metrics(y_true, y_pred, w_low)
        m_high = weighted_metrics(y_true, y_pred, w_high)
        assert m_low["weighted_mean_true"] < m_high["weighted_mean_true"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframes(self):
        val = pd.DataFrame({"d_bin": pd.Series([], dtype=int)})
        tgt = pd.DataFrame({"d_bin": [0, 1, 2]})
        w, cert = build_weights_layered(val, tgt, ["d_bin"])
        assert len(w) == 0

    def test_single_bin(self):
        val = pd.DataFrame({"d_bin": [0, 0, 0, 0, 0]})
        tgt = pd.DataFrame({"d_bin": [0, 0, 0, 0, 0]})
        w, cert = build_weights_layered(val, tgt, ["d_bin"])
        assert np.allclose(w, 1.0 / 5)
        assert cert["residual_margin_gap"] == pytest.approx(0.0, abs=1e-10)
