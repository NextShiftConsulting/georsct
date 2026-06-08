"""Layer 4: Trajectory analysis tests."""

from conftest import make_cert
from georsct.healthcheck.layers.trajectory import analyze_trajectory
from georsct.healthcheck.thresholds import GEOSPATIAL_CONUS27


P = GEOSPATIAL_CONUS27


class TestMinimumLevels:
    def test_single_level_returns_none(self):
        certs = {"r0": make_cert(alpha=0.60)}
        result = analyze_trajectory(certs, None, {}, P)
        assert result is None

    def test_two_levels_ok(self):
        certs = {
            "r0": make_cert(alpha=0.50, sigma=0.15),
            "r1": make_cert(alpha=0.60, sigma=0.12),
        }
        result = analyze_trajectory(certs, None, {"r0": "CONUS27", "r1": "CONUS27"}, P)
        assert result is not None
        assert result.levels_present == ["r0", "r1"]


class TestProvenanceGating:
    def test_mismatched_presets_refuses(self):
        certs = {
            "r0": make_cert(alpha=0.50),
            "r1": make_cert(alpha=0.60),
        }
        prov = {"r0": "CONUS27", "r1": "STRICT"}
        result = analyze_trajectory(certs, None, prov, P)
        assert result.provenance_valid is False
        assert result.alpha_trend == "unknown"
        assert any("PROVENANCE_MISMATCH" in w for w in result.warnings)

    def test_matched_presets_ok(self):
        certs = {
            "r0": make_cert(alpha=0.50, sigma=0.15),
            "r1": make_cert(alpha=0.60, sigma=0.12),
        }
        prov = {"r0": "CONUS27", "r1": "CONUS27"}
        result = analyze_trajectory(certs, None, prov, P)
        assert result.provenance_valid is True

    def test_unknown_preset_ignored(self):
        certs = {
            "r0": make_cert(alpha=0.50, sigma=0.15),
            "r1": make_cert(alpha=0.60, sigma=0.12),
        }
        prov = {"r0": "CONUS27"}  # r1 missing -> "unknown"
        result = analyze_trajectory(certs, None, prov, P)
        assert result.provenance_valid is True


class TestAlphaTrend:
    def test_improving(self):
        certs = {
            "r0": make_cert(alpha=0.40, sigma=0.15),
            "r1": make_cert(alpha=0.55, sigma=0.15),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.alpha_trend == "improving"

    def test_regressing(self):
        certs = {
            "r0": make_cert(alpha=0.60, sigma=0.15),
            "r1": make_cert(alpha=0.40, sigma=0.15),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.alpha_trend == "regressing"
        assert any("TRAJECTORY_REGRESSION" in w for w in result.warnings)

    def test_stalled(self):
        certs = {
            "r0": make_cert(alpha=0.50, sigma=0.15),
            "r1": make_cert(alpha=0.51, sigma=0.15),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.alpha_trend == "stalled"


class TestSigmaTrend:
    def test_stabilizing(self):
        certs = {
            "r0": make_cert(alpha=0.50, sigma=0.30),
            "r1": make_cert(alpha=0.50, sigma=0.20),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.sigma_trend == "stabilizing"

    def test_destabilizing(self):
        certs = {
            "r0": make_cert(alpha=0.50, sigma=0.15),
            "r1": make_cert(alpha=0.50, sigma=0.25),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.sigma_trend == "destabilizing"
        assert any("SIGMA_DESTABILIZING" in w for w in result.warnings)


class TestConvergence:
    def test_healthy_converged(self):
        certs = {
            "r0": make_cert(alpha=0.60, kappa=0.70, sigma=0.10, spatial_metric=0.65),
            "r1": make_cert(alpha=0.62, kappa=0.72, sigma=0.10, spatial_metric=0.655),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.convergence_state == "healthy_converged"

    def test_prematurely_stalled(self):
        certs = {
            "r0": make_cert(alpha=0.30, kappa=0.35, sigma=0.10, spatial_metric=0.40),
            "r1": make_cert(alpha=0.31, kappa=0.36, sigma=0.10, spatial_metric=0.402),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.convergence_state == "prematurely_stalled"
        assert any("PREMATURE_STALL" in w for w in result.warnings)

    def test_active_improvement(self):
        certs = {
            "r0": make_cert(alpha=0.40, sigma=0.15, spatial_metric=0.45),
            "r1": make_cert(alpha=0.55, sigma=0.12, spatial_metric=0.55),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.convergence_state == "active"

    def test_regressing_negative_uplift(self):
        certs = {
            "r0": make_cert(alpha=0.60, sigma=0.10, spatial_metric=0.65),
            "r1": make_cert(alpha=0.45, sigma=0.15, spatial_metric=0.50),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X"}, P)
        assert result.convergence_state == "regressing"


class TestUpliftFromMoneyTable:
    def test_money_table_takes_precedence(self):
        certs = {
            "r0": make_cert(alpha=0.50, sigma=0.15, spatial_metric=0.45),
            "r1": make_cert(alpha=0.55, sigma=0.12, spatial_metric=0.50),
        }
        money = {"uplift_r0_r1_pct": 15.5}
        result = analyze_trajectory(certs, money, {"r0": "X", "r1": "X"}, P)
        assert result.uplift_pct["r0->r1"] == 15.5


class TestThreeLevels:
    def test_trends_use_last_step_only(self):
        certs = {
            "r0": make_cert(alpha=0.40, sigma=0.30, spatial_metric=0.40),
            "r1": make_cert(alpha=0.55, sigma=0.20, spatial_metric=0.50),
            "r2": make_cert(alpha=0.45, sigma=0.25, spatial_metric=0.48),
        }
        result = analyze_trajectory(certs, None, {"r0": "X", "r1": "X", "r2": "X"}, P)
        # Last step is r1->r2: alpha went down, sigma went up
        assert result.alpha_trend == "regressing"
        assert result.sigma_trend == "destabilizing"
        assert len(result.uplift_pct) == 2
        assert "r0->r1" in result.uplift_pct
        assert "r1->r2" in result.uplift_pct
