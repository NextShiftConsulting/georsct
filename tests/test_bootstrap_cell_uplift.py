"""Tests for cell-level bootstrap CIs (M1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wsp" / "floodrsct" / "jobs"))
from compute_uplift_table import bootstrap_cell_uplift


def _make_cells(r0_r1_uplifts, r1_r2_uplifts=None):
    """Build minimal money_table rows from uplift lists."""
    n = max(len(r0_r1_uplifts), len(r1_r2_uplifts or []))
    rows = []
    for i in range(n):
        row = {"scenario": f"s{i}", "target": f"t{i}"}
        if i < len(r0_r1_uplifts):
            row["uplift_r0_r1_pct"] = r0_r1_uplifts[i]
        if r1_r2_uplifts and i < len(r1_r2_uplifts):
            row["uplift_r1_r2_pct"] = r1_r2_uplifts[i]
        rows.append(row)
    return rows


class TestBootstrapCellUplift:

    def test_positive_uplift_ci_above_zero(self):
        """If all cells have large positive uplift, CI should be above zero."""
        cells = _make_cells([10.0, 15.0, 12.0, 8.0, 20.0])
        result = bootstrap_cell_uplift(cells, n_boot=5000, seed=42)
        r0_r1 = result["r0_r1"]
        assert r0_r1["ci_lower_95"] > 0
        assert r0_r1["ci_upper_95"] > r0_r1["ci_lower_95"]
        assert r0_r1["n_cells"] == 5

    def test_mixed_uplift_ci_spans_zero(self):
        """If cells have mixed sign uplift, CI should span zero."""
        cells = _make_cells([10.0, -8.0, 5.0, -12.0, 3.0, -6.0])
        result = bootstrap_cell_uplift(cells, n_boot=5000, seed=42)
        r0_r1 = result["r0_r1"]
        assert r0_r1["ci_lower_95"] < 0
        assert r0_r1["ci_upper_95"] > 0

    def test_too_few_cells(self):
        """With <2 cells, result should report 'too few cells'."""
        cells = _make_cells([5.0])
        result = bootstrap_cell_uplift(cells, n_boot=100, seed=42)
        assert "note" in result["r0_r1"]
        assert result["r0_r1"]["n_cells"] == 1

    def test_zero_cells(self):
        """Empty money table returns 'too few cells' for both transitions."""
        result = bootstrap_cell_uplift([], n_boot=100, seed=42)
        assert "note" in result["r0_r1"]
        assert "note" in result["r1_r2"]

    def test_both_transitions(self):
        """Both R0->R1 and R1->R2 are computed independently."""
        cells = _make_cells(
            r0_r1_uplifts=[10.0, 15.0, 8.0],
            r1_r2_uplifts=[2.0, 3.0, 1.0],
        )
        result = bootstrap_cell_uplift(cells, n_boot=2000, seed=42)
        assert result["r0_r1"]["n_cells"] == 3
        assert result["r1_r2"]["n_cells"] == 3
        # R0->R1 should have larger mean than R1->R2
        assert result["r0_r1"]["observed_mean_uplift_pct"] > result["r1_r2"]["observed_mean_uplift_pct"]

    def test_deterministic_with_seed(self):
        """Same seed produces identical CIs."""
        cells = _make_cells([10.0, 15.0, 8.0, 20.0])
        r1 = bootstrap_cell_uplift(cells, n_boot=1000, seed=99)
        r2 = bootstrap_cell_uplift(cells, n_boot=1000, seed=99)
        assert r1["r0_r1"]["ci_lower_95"] == r2["r0_r1"]["ci_lower_95"]
        assert r1["r0_r1"]["ci_upper_95"] == r2["r0_r1"]["ci_upper_95"]

    def test_pct_positive_all_positive(self):
        """If all uplifts are positive, pct_bootstrap_positive should be ~1.0."""
        cells = _make_cells([10.0, 15.0, 20.0, 25.0, 30.0])
        result = bootstrap_cell_uplift(cells, n_boot=5000, seed=42)
        assert result["r0_r1"]["pct_bootstrap_positive"] > 0.99

    def test_output_fields(self):
        """Verify all required output fields are present."""
        cells = _make_cells([10.0, 5.0, 8.0])
        result = bootstrap_cell_uplift(cells, n_boot=100, seed=42)
        assert result["bootstrap_unit"] == "experiment_cell (scenario x target)"
        assert result["n_bootstrap"] == 100
        assert result["seed"] == 42
        assert result["confidence_level"] == 0.95
        r = result["r0_r1"]
        for field in ("n_cells", "observed_mean_uplift_pct",
                      "ci_lower_95", "ci_upper_95", "pct_bootstrap_positive"):
            assert field in r, f"Missing field: {field}"
