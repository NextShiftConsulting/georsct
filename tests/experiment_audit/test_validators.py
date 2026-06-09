"""Tests for the six audit gates in validators.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from georsct.experiment_audit.contract import parse_contract
from georsct.experiment_audit.models import ArtifactRecord, CellKey, CheckResult, Severity
from georsct.experiment_audit.validators import (
    gate_1_contract_parse,
    gate_2_declared_outputs,
    gate_3_convention_supplement,
    gate_4_cell_matrix,
    gate_5_aggregate_content,
    gate_6_ordering,
)

FIXTURES = Path(__file__).parent / "fixtures"
MINI_CONTRACT = FIXTURES / "mini_contract.yaml"


@pytest.fixture
def contract():
    """Load the mini contract fixture."""
    return parse_contract(MINI_CONTRACT)


# ---------------------------------------------------------------------------
# Gate 1: Contract Parse
# ---------------------------------------------------------------------------

class TestGate1ContractParse:
    """Gate 1 detects unresolved template variables."""

    def test_detects_unresolved_vlm(self, contract):
        """The mini contract has {vlm} in r4_vlm_comparison -- should FAIL."""
        results = gate_1_contract_parse(contract)
        fail_results = [r for r in results if r.severity == Severity.FAIL_UNRESOLVED_TEMPLATE]
        assert len(fail_results) >= 1
        assert any("{vlm}" in r.message for r in fail_results)

    def test_passes_when_clean(self, contract):
        """Remove the offending phase and verify PASS."""
        # Filter out the phase with unresolved templates
        contract.phases = [
            p for p in contract.phases if p.phase_id != "r4_vlm_comparison"
        ]
        results = gate_1_contract_parse(contract)
        assert len(results) == 1
        assert results[0].severity == Severity.PASS


# ---------------------------------------------------------------------------
# Gate 2: Declared Output Existence
# ---------------------------------------------------------------------------

class TestGate2DeclaredOutputs:
    """Gate 2 checks declared outputs exist in inventory."""

    def test_pass_when_outputs_exist(self, contract):
        """All declared outputs present -> PASS for those outputs."""
        inventory = {
            "results/test/geometry_kappa.json": ArtifactRecord(
                s3_key="results/test/geometry_kappa.json", exists=True
            ),
            "results/test/r0_houston.json": ArtifactRecord(
                s3_key="results/test/r0_houston.json", exists=True
            ),
            "results/test/r0_nyc.json": ArtifactRecord(
                s3_key="results/test/r0_nyc.json", exists=True
            ),
            "results/test/r0_houston_predictions.parquet": ArtifactRecord(
                s3_key="results/test/r0_houston_predictions.parquet", exists=True
            ),
            "results/test/r0_nyc_predictions.parquet": ArtifactRecord(
                s3_key="results/test/r0_nyc_predictions.parquet", exists=True
            ),
        }
        results = gate_2_declared_outputs(contract, inventory)
        pass_results = [r for r in results if r.severity == Severity.PASS]
        # geometry_kappa (1) + r0 per-scenario json (2) + parquet (2) = 5
        assert len(pass_results) == 5

    def test_fail_when_missing(self, contract):
        """Missing declared output -> FAIL_MISSING_OUTPUT."""
        inventory = {
            "results/test/geometry_kappa.json": ArtifactRecord(
                s3_key="results/test/geometry_kappa.json", exists=True
            ),
            # r0_baseline outputs missing
        }
        results = gate_2_declared_outputs(contract, inventory)
        fail_results = [r for r in results if r.severity == Severity.FAIL_MISSING_OUTPUT]
        assert len(fail_results) >= 1

    def test_warn_on_phase_without_outputs(self, contract):
        """Phase with no declared outputs -> WARN_CONTRACT_GAP."""
        # diagnostics_r0 and r4_vlm_comparison have no outputs
        results = gate_2_declared_outputs(contract, {})
        warn_results = [r for r in results if r.severity == Severity.WARN_CONTRACT_GAP]
        phases_warned = {r.phase_id for r in warn_results}
        assert "diagnostics_r0" in phases_warned


# ---------------------------------------------------------------------------
# Gate 3: Convention Supplement
# ---------------------------------------------------------------------------

class TestGate3ConventionSupplement:
    """Gate 3 checks convention-based outputs and undeclared artifacts."""

    def test_convention_found_warns_contract_gap(self, contract):
        """Convention file exists but phase has no declared outputs -> WARN."""
        s3_keys = ["results/s035/diagnostics_r0.json"]
        results = gate_3_convention_supplement(contract, s3_keys)
        warns = [r for r in results if r.severity == Severity.WARN_CONTRACT_GAP]
        assert any("diagnostics_r0" in r.message for r in warns)

    def test_convention_missing_fails(self, contract):
        """Convention file missing -> FAIL_MISSING_OUTPUT."""
        results = gate_3_convention_supplement(contract, [])
        fails = [r for r in results if r.severity == Severity.FAIL_MISSING_OUTPUT]
        assert len(fails) >= 1

    def test_undeclared_artifact_warns(self, contract):
        """S3 key under prefix not matching any convention -> WARN_UNDECLARED."""
        s3_keys = ["results/s035/mystery_file.json"]
        results = gate_3_convention_supplement(contract, s3_keys)
        undeclared = [r for r in results if r.severity == Severity.WARN_UNDECLARED_ARTIFACT]
        assert len(undeclared) >= 1
        assert any("mystery_file" in r.message for r in undeclared)


# ---------------------------------------------------------------------------
# Gate 4: Cell Matrix
# ---------------------------------------------------------------------------

class TestGate4CellMatrix:
    """Gate 4 compares contracted vs actual cells."""

    def test_detects_missing_cells(self, contract):
        """Contracted cell not in actuals -> FAIL_MISSING_OUTPUT."""
        # Contract has houston/obs_nfip_event_claims, nyc/obs_nfip_event_claims,
        # houston/obs_has_311
        actual = set()  # nothing present
        results = gate_4_cell_matrix(contract, actual)
        fails = [r for r in results if r.severity == Severity.FAIL_MISSING_OUTPUT]
        assert len(fails) == len(contract.cell_matrix)

    def test_detects_extra_cells(self, contract):
        """Cell in actuals but not contracted -> WARN_SCOPE_EXTRA."""
        extra = CellKey(scenario="mars", target="obs_dust_storms")
        actual = set(contract.cell_matrix) | {extra}
        results = gate_4_cell_matrix(contract, actual)
        extras = [r for r in results if r.severity == Severity.WARN_SCOPE_EXTRA]
        assert len(extras) == 1
        assert extras[0].cell == extra

    def test_all_match_passes(self, contract):
        """All contracted cells present, no extras -> all PASS."""
        actual = set(contract.cell_matrix)
        results = gate_4_cell_matrix(contract, actual)
        assert all(r.severity == Severity.PASS for r in results)
        assert len(results) == len(contract.cell_matrix)


# ---------------------------------------------------------------------------
# Gate 5: Aggregate Content
# ---------------------------------------------------------------------------

class TestGate5AggregateContent:
    """Gate 5 checks money_table and geometry_kappa content."""

    def test_money_table_missing_cell(self, contract):
        """Contracted cell not in money_table -> FAIL_CONTENT_INCOMPLETE."""
        money_content = {
            "cells": [
                {"scenario": "houston", "target": "obs_nfip_event_claims",
                 "r0_metric": 0.5},
                # nyc/obs_nfip_event_claims and houston/obs_has_311 missing
            ]
        }
        artifacts = {
            "results/test/money_table.json": ArtifactRecord(
                s3_key="results/test/money_table.json",
                exists=True,
                content=money_content,
            )
        }
        results = gate_5_aggregate_content(contract, artifacts)
        incomplete = [r for r in results if r.severity == Severity.FAIL_CONTENT_INCOMPLETE]
        assert len(incomplete) >= 1

    def test_money_table_null_metric(self, contract):
        """Null metric in money_table -> FAIL_CONTENT_DEGENERATE."""
        money_content = {
            "cells": [
                {"scenario": "houston", "target": "obs_nfip_event_claims",
                 "r0_metric": None},
                {"scenario": "nyc", "target": "obs_nfip_event_claims",
                 "r0_metric": 0.4},
                {"scenario": "houston", "target": "obs_has_311",
                 "r0_metric": 0.6},
            ]
        }
        artifacts = {
            "results/test/money_table.json": ArtifactRecord(
                s3_key="results/test/money_table.json",
                exists=True,
                content=money_content,
            )
        }
        results = gate_5_aggregate_content(contract, artifacts)
        degenerate = [r for r in results if r.severity == Severity.FAIL_CONTENT_DEGENERATE]
        assert len(degenerate) == 1

    def test_geometry_kappa_null_warns(self, contract):
        """Null kappa_geom in geometry_kappa -> WARN_CONTRACT_GAP."""
        kappa_content = {
            "cells": [
                {"scenario": "houston", "target": "obs_nfip_event_claims",
                 "kappa_geom": None},
            ]
        }
        artifacts = {
            "results/test/geometry_kappa.json": ArtifactRecord(
                s3_key="results/test/geometry_kappa.json",
                exists=True,
                content=kappa_content,
            )
        }
        results = gate_5_aggregate_content(contract, artifacts)
        warns = [r for r in results if r.severity == Severity.WARN_CONTRACT_GAP]
        assert len(warns) >= 1


# ---------------------------------------------------------------------------
# Gate 6: Ordering
# ---------------------------------------------------------------------------

class TestGate6Ordering:
    """Gate 6 checks timestamp ordering of artifacts."""

    def test_correct_ordering_passes(self):
        """Artifacts with correct internal timestamps -> PASS."""
        artifacts = {
            "results/s035/gearbox_warmup.json": ArtifactRecord(
                s3_key="results/s035/gearbox_warmup.json",
                exists=True,
                internal_timestamp="2025-01-01T00:00:00Z",
            ),
            "results/s035/certificates_r0.json": ArtifactRecord(
                s3_key="results/s035/certificates_r0.json",
                exists=True,
                internal_timestamp="2025-01-02T00:00:00Z",
            ),
        }
        results = gate_6_ordering(artifacts)
        pass_results = [r for r in results if r.severity == Severity.PASS]
        assert len(pass_results) >= 1

    def test_scenario_expansion(self):
        """Per-scenario rules expand {scenario} placeholders."""
        artifacts = {
            "results/s035/geometry_kappa.json": ArtifactRecord(
                s3_key="results/s035/geometry_kappa.json",
                exists=True,
                internal_timestamp="2025-01-01T00:00:00Z",
            ),
            "results/s035/r0_houston.json": ArtifactRecord(
                s3_key="results/s035/r0_houston.json",
                exists=True,
                internal_timestamp="2025-01-02T00:00:00Z",
            ),
        }
        results = gate_6_ordering(artifacts, scenarios=["houston"])
        # Should have a result for the geometry_kappa -> r0_houston rule
        assert any("houston" in r.name for r in results)

    def test_missing_artifact_unverifiable(self):
        """Missing artifact -> WARN_TIMESTAMP_UNVERIFIABLE."""
        results = gate_6_ordering({}, scenarios=["houston"])
        unverifiable = [
            r for r in results
            if r.severity == Severity.WARN_TIMESTAMP_UNVERIFIABLE
        ]
        assert len(unverifiable) >= 1
