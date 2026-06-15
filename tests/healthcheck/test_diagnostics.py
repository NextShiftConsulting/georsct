"""Layer 3: Diagnostic ratio analysis tests."""

from tests.healthcheck.conftest import make_cert
from georsct.healthcheck.layers.diagnostics import analyze_diagnostics
from georsct.healthcheck.thresholds import GEOSPATIAL_CONUS27


P = GEOSPATIAL_CONUS27


class TestNoData:
    def test_both_none(self):
        result = analyze_diagnostics(None, None, P)
        assert result is None

    def test_diag_only(self):
        diag = {"scenario": "a", "target": "b", "diag_leakage": 0.90}
        result = analyze_diagnostics(diag, None, P)
        assert result is not None
        assert result.leakage == 0.90
        assert len(result.warnings) == 1  # NO_TRANSFER

    def test_gearbox_only(self):
        gb = {"scenario": "a", "target": "b", "collapse_risk": 0.10}
        result = analyze_diagnostics(None, gb, P)
        assert result is not None
        assert result.leakage is None
        assert len(result.warnings) == 0


class TestLeakage:
    def test_above_threshold(self):
        diag = {"diag_leakage": 1.10, "diag_transfer": 0.5}
        result = analyze_diagnostics(diag, None, P)
        assert any("SPATIAL_BLEED" in w for w in result.warnings)

    def test_at_threshold(self):
        diag = {"diag_leakage": 1.05, "diag_transfer": 0.5}
        result = analyze_diagnostics(diag, None, P)
        assert not any("SPATIAL_BLEED" in w for w in result.warnings)

    def test_below_threshold(self):
        diag = {"diag_leakage": 0.95, "diag_transfer": 0.5}
        result = analyze_diagnostics(diag, None, P)
        assert not any("SPATIAL_BLEED" in w for w in result.warnings)


class TestTransfer:
    def test_null_transfer(self):
        diag = {"diag_leakage": 0.90, "diag_transfer": None}
        result = analyze_diagnostics(diag, None, P)
        assert any("NO_TRANSFER_SIGNAL" in w for w in result.warnings)

    def test_zero_transfer(self):
        diag = {"diag_leakage": 0.90, "diag_transfer": 0.0}
        result = analyze_diagnostics(diag, None, P)
        assert any("NO_TRANSFER_SIGNAL" in w for w in result.warnings)

    def test_missing_transfer_key(self):
        diag = {"diag_leakage": 0.90}
        result = analyze_diagnostics(diag, None, P)
        assert any("NO_TRANSFER_SIGNAL" in w for w in result.warnings)

    def test_positive_transfer(self):
        diag = {"diag_leakage": 0.90, "diag_transfer": 0.5}
        result = analyze_diagnostics(diag, None, P)
        assert not any("NO_TRANSFER_SIGNAL" in w for w in result.warnings)


class TestSolver:
    def test_weak_solver(self):
        diag = {"diag_solver": 0.30, "diag_transfer": 0.5}
        result = analyze_diagnostics(diag, None, P)
        assert any("WEAK_SOLVER" in w for w in result.warnings)

    def test_adequate_solver(self):
        diag = {"diag_solver": 0.60, "diag_transfer": 0.5}
        result = analyze_diagnostics(diag, None, P)
        assert not any("WEAK_SOLVER" in w for w in result.warnings)


class TestResidualSpatial:
    def test_high_residual(self):
        diag = {"diag_residual_spatial": 0.50, "diag_transfer": 0.5}
        result = analyze_diagnostics(diag, None, P)
        assert any("UNEXPLAINED_SPATIAL_PATTERN" in w for w in result.warnings)

    def test_low_residual(self):
        diag = {"diag_residual_spatial": 0.30, "diag_transfer": 0.5}
        result = analyze_diagnostics(diag, None, P)
        assert not any("UNEXPLAINED_SPATIAL_PATTERN" in w for w in result.warnings)


class TestGearbox:
    def test_collapse_risk(self):
        gb = {"collapse_risk": 0.50}
        result = analyze_diagnostics(None, gb, P)
        assert any("COLLAPSE_RISK" in w for w in result.warnings)

    def test_low_coherence(self):
        gb = {"coherence": 0.20}
        result = analyze_diagnostics(None, gb, P)
        assert any("LOW_COHERENCE" in w for w in result.warnings)

    def test_healthy_gearbox(self):
        gb = {"collapse_risk": 0.20, "coherence": 0.50}
        result = analyze_diagnostics(None, gb, P)
        assert len(result.warnings) == 0


class TestCombined:
    def test_diag_and_gearbox_warnings_accumulate(self):
        diag = {"diag_leakage": 1.20, "diag_transfer": 0.5}
        gb = {"collapse_risk": 0.50}
        result = analyze_diagnostics(diag, gb, P)
        assert any("SPATIAL_BLEED" in w for w in result.warnings)
        assert any("COLLAPSE_RISK" in w for w in result.warnings)
        assert result.leakage == 1.20
