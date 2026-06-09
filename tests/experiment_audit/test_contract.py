"""Tests for the contract parser."""
from __future__ import annotations

from pathlib import Path

import pytest

from georsct.experiment_audit.contract import parse_contract
from georsct.experiment_audit.models import CellKey

FIXTURES = Path(__file__).parent / "fixtures"
MINI_CONTRACT = FIXTURES / "mini_contract.yaml"


@pytest.fixture
def contract():
    """Parse the mini contract fixture."""
    return parse_contract(MINI_CONTRACT)


class TestParseContract:
    """Tests for parse_contract top-level fields."""

    def test_experiment_name(self, contract):
        assert contract.experiment == "test-audit"

    def test_bucket(self, contract):
        assert contract.bucket == "test-bucket"

    def test_version(self, contract):
        assert contract.version == "1.0"

    def test_scenarios(self, contract):
        assert contract.scenarios == ["houston", "nyc"]


class TestCellMatrix:
    """Tests for cell matrix resolution."""

    def test_cell_matrix_has_3_cells(self, contract):
        assert len(contract.cell_matrix) == 3

    def test_nyc_obs_has_311_not_in_matrix(self, contract):
        assert CellKey(scenario="nyc", target="obs_has_311") not in contract.cell_matrix

    def test_houston_nfip_in_matrix(self, contract):
        assert CellKey(scenario="houston", target="obs_nfip_event_claims") in contract.cell_matrix

    def test_nyc_nfip_in_matrix(self, contract):
        assert CellKey(scenario="nyc", target="obs_nfip_event_claims") in contract.cell_matrix

    def test_houston_has_311_in_matrix(self, contract):
        assert CellKey(scenario="houston", target="obs_has_311") in contract.cell_matrix


class TestPhases:
    """Tests for phase parsing."""

    def test_phase_list_contains_expected(self, contract):
        phase_ids = [p.phase_id for p in contract.phases]
        assert "geometry_kappa" in phase_ids
        assert "r0_baseline" in phase_ids
        assert "diagnostics_r0" in phase_ids

    def test_geometry_kappa_has_1_output(self, contract):
        phase = contract.get_phase("geometry_kappa")
        assert phase is not None
        assert len(phase.declared_outputs) == 1

    def test_diagnostics_r0_has_no_outputs(self, contract):
        phase = contract.get_phase("diagnostics_r0")
        assert phase is not None
        assert len(phase.declared_outputs) == 0
        assert phase.has_declared_outputs is False

    def test_depends_on_defaults_empty(self, contract):
        phase = contract.get_phase("geometry_kappa")
        assert phase is not None
        assert phase.depends_on == []


class TestResolveOutputKeys:
    """Tests for output key resolution."""

    def test_r0_baseline_resolves_4_keys(self, contract):
        phase = contract.get_phase("r0_baseline")
        assert phase is not None
        keys = phase.resolve_output_keys(contract.scenarios)
        assert len(keys) == 4

    def test_geometry_kappa_resolves_1_key(self, contract):
        phase = contract.get_phase("geometry_kappa")
        assert phase is not None
        keys = phase.resolve_output_keys(contract.scenarios)
        assert len(keys) == 1
        assert keys[0] == "results/test/geometry_kappa.json"


class TestUnresolvedTemplates:
    """Tests for unresolved template detection."""

    def test_finds_vlm_in_r4(self, contract):
        issues = contract.find_unresolved_templates()
        assert len(issues) > 0
        vlm_issues = [i for i in issues if "{vlm}" in i]
        assert len(vlm_issues) > 0
        assert any("r4_vlm_comparison" in i for i in vlm_issues)
