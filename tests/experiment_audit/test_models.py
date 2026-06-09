"""Tests for experiment_audit.models data classes and enums."""
from __future__ import annotations

import json

import pytest

from georsct.experiment_audit.models import (
    ArtifactRecord,
    AuditResult,
    CellKey,
    CheckResult,
    Severity,
)


# ── Severity ────────────────────────────────────────────────────────

class TestSeverity:
    """Severity enum prefix methods."""

    @pytest.mark.parametrize("sev", [
        Severity.FAIL_MISSING_OUTPUT,
        Severity.FAIL_UNRESOLVED_TEMPLATE,
        Severity.FAIL_ORDERING_VIOLATION,
        Severity.FAIL_CONTENT_INCOMPLETE,
        Severity.FAIL_CONTENT_DEGENERATE,
    ])
    def test_is_fail_true(self, sev: Severity) -> None:
        assert sev.is_fail() is True
        assert sev.is_warn() is False

    @pytest.mark.parametrize("sev", [
        Severity.WARN_CONTRACT_GAP,
        Severity.WARN_UNDECLARED_ARTIFACT,
        Severity.WARN_TIMESTAMP_FALLBACK,
        Severity.WARN_TIMESTAMP_UNVERIFIABLE,
        Severity.WARN_SCOPE_EXTRA,
    ])
    def test_is_warn_true(self, sev: Severity) -> None:
        assert sev.is_warn() is True
        assert sev.is_fail() is False

    def test_pass_is_neither(self) -> None:
        assert Severity.PASS.is_fail() is False
        assert Severity.PASS.is_warn() is False


# ── CellKey ─────────────────────────────────────────────────────────

class TestCellKey:
    """CellKey creation, stringification, and parsing."""

    def test_str(self) -> None:
        ck = CellKey(scenario="houston", target="depth")
        assert str(ck) == "houston/depth"

    def test_from_string(self) -> None:
        ck = CellKey.from_string("riverside/velocity")
        assert ck.scenario == "riverside"
        assert ck.target == "velocity"

    def test_roundtrip(self) -> None:
        original = CellKey(scenario="nyc", target="extent")
        assert CellKey.from_string(str(original)) == original

    def test_frozen(self) -> None:
        ck = CellKey(scenario="a", target="b")
        with pytest.raises(AttributeError):
            ck.scenario = "c"  # type: ignore[misc]


# ── ArtifactRecord ──────────────────────────────────────────────────

class TestArtifactRecord:
    """ArtifactRecord.best_timestamp logic."""

    def test_best_timestamp_prefers_internal(self) -> None:
        ar = ArtifactRecord(
            s3_key="k",
            exists=True,
            internal_timestamp="2026-01-01T00:00:00Z",
            last_modified="2025-12-31T00:00:00Z",
        )
        assert ar.best_timestamp() == "2026-01-01T00:00:00Z"

    def test_best_timestamp_falls_back_to_last_modified(self) -> None:
        ar = ArtifactRecord(
            s3_key="k",
            exists=True,
            last_modified="2025-12-31T00:00:00Z",
        )
        assert ar.best_timestamp() == "2025-12-31T00:00:00Z"

    def test_best_timestamp_returns_none_when_both_missing(self) -> None:
        ar = ArtifactRecord(s3_key="k", exists=False)
        assert ar.best_timestamp() is None


# ── CheckResult ─────────────────────────────────────────────────────

class TestCheckResult:
    """CheckResult serialization."""

    def _make_check(self) -> CheckResult:
        return CheckResult(
            audit_stage="G1",
            name="output_exists",
            severity=Severity.FAIL_MISSING_OUTPUT,
            message="predictions.json not found",
            phase_id="phase_02",
            cell=CellKey(scenario="houston", target="depth"),
        )

    def test_to_dict_keys(self) -> None:
        d = self._make_check().to_dict()
        assert set(d.keys()) == {"audit_stage", "name", "severity", "message", "phase_id", "cell"}

    def test_to_dict_severity_is_string(self) -> None:
        d = self._make_check().to_dict()
        assert d["severity"] == "FAIL_MISSING_OUTPUT"

    def test_to_dict_cell_is_string(self) -> None:
        d = self._make_check().to_dict()
        assert d["cell"] == "houston/depth"

    def test_json_roundtrip(self) -> None:
        d = self._make_check().to_dict()
        serialized = json.dumps(d)
        restored = json.loads(serialized)
        assert restored == d

    def test_to_dict_none_cell(self) -> None:
        cr = CheckResult(
            audit_stage="G0", name="scope", severity=Severity.PASS, message="ok"
        )
        assert cr.to_dict()["cell"] is None
        assert cr.to_dict()["phase_id"] is None


# ── AuditResult ─────────────────────────────────────────────────────

class TestAuditResult:
    """AuditResult aggregation methods."""

    @staticmethod
    def _check(sev: Severity) -> CheckResult:
        return CheckResult(audit_stage="G", name="n", severity=sev, message="m")

    def test_summary_counts_mixed(self) -> None:
        ar = AuditResult(checks=[
            self._check(Severity.PASS),
            self._check(Severity.PASS),
            self._check(Severity.WARN_CONTRACT_GAP),
            self._check(Severity.FAIL_MISSING_OUTPUT),
        ])
        assert ar.summary_counts() == {"PASS": 2, "WARN": 1, "FAIL": 1}

    def test_overall_status_fail(self) -> None:
        ar = AuditResult(checks=[
            self._check(Severity.PASS),
            self._check(Severity.FAIL_CONTENT_DEGENERATE),
        ])
        assert ar.overall_status() == "FAIL"

    def test_overall_status_warn(self) -> None:
        ar = AuditResult(checks=[
            self._check(Severity.PASS),
            self._check(Severity.WARN_SCOPE_EXTRA),
        ])
        assert ar.overall_status() == "WARN"

    def test_overall_status_pass(self) -> None:
        ar = AuditResult(checks=[self._check(Severity.PASS)])
        assert ar.overall_status() == "PASS"

    def test_empty_is_pass(self) -> None:
        ar = AuditResult()
        assert ar.overall_status() == "PASS"
        assert ar.summary_counts() == {"PASS": 0, "FAIL": 0, "WARN": 0}
