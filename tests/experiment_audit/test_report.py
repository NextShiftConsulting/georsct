"""Tests for JSON and Markdown report generators."""
from __future__ import annotations

import json

from georsct.experiment_audit.models import (
    AuditResult,
    CheckResult,
    CellKey,
    Severity,
)
from georsct.experiment_audit.report import render_json, render_markdown


def _sample_result() -> AuditResult:
    """Build a sample AuditResult with mixed severities."""
    return AuditResult(
        checks=[
            CheckResult(
                audit_stage="completeness",
                name="output_exists",
                severity=Severity.PASS,
                message="All outputs present",
                phase_id="phase_1",
                cell=CellKey("houston", "depth"),
            ),
            CheckResult(
                audit_stage="completeness",
                name="output_exists",
                severity=Severity.FAIL_MISSING_OUTPUT,
                message="Missing predictions.json",
                phase_id="phase_2",
                cell=CellKey("nola", "ranking"),
            ),
            CheckResult(
                audit_stage="scope",
                name="scope_extra",
                severity=Severity.WARN_SCOPE_EXTRA,
                message="Extra artifact not in contract",
                phase_id="phase_1",
                cell=CellKey("houston", "depth"),
            ),
            CheckResult(
                audit_stage="contract",
                name="contract_gap",
                severity=Severity.WARN_CONTRACT_GAP,
                message="Field 'kappa' missing from contract",
                phase_id=None,
                cell=None,
            ),
        ],
        metadata={"experiment_name": "s035_r0", "version": "1.0"},
    )


class TestRenderJson:
    """Tests for render_json."""

    def test_parseable_json(self) -> None:
        result = _sample_result()
        output = render_json(result)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_overall_status(self) -> None:
        result = _sample_result()
        parsed = json.loads(render_json(result))
        assert parsed["overall_status"] == "FAIL"

    def test_summary_counts(self) -> None:
        result = _sample_result()
        parsed = json.loads(render_json(result))
        assert parsed["summary"] == {"PASS": 1, "FAIL": 1, "WARN": 2}

    def test_check_count(self) -> None:
        result = _sample_result()
        parsed = json.loads(render_json(result))
        assert len(parsed["checks"]) == 4

    def test_metadata_preserved(self) -> None:
        result = _sample_result()
        parsed = json.loads(render_json(result))
        assert parsed["metadata"]["experiment_name"] == "s035_r0"
        assert parsed["metadata"]["version"] == "1.0"

    def test_generated_at_present(self) -> None:
        result = _sample_result()
        parsed = json.loads(render_json(result))
        assert "generated_at" in parsed
        assert parsed["generated_at"].endswith("+00:00")


class TestRenderMarkdown:
    """Tests for render_markdown."""

    def test_has_summary_section(self) -> None:
        md = render_markdown(_sample_result())
        assert "## Summary" in md

    def test_has_failures_section(self) -> None:
        md = render_markdown(_sample_result())
        assert "## Failures" in md
        assert "FAIL_MISSING_OUTPUT" in md
        assert "Missing predictions.json" in md

    def test_has_warnings_section(self) -> None:
        md = render_markdown(_sample_result())
        assert "## Warnings" in md
        assert "WARN_SCOPE_EXTRA" in md
        assert "WARN_CONTRACT_GAP" in md

    def test_has_passed_checks_section(self) -> None:
        md = render_markdown(_sample_result())
        assert "## Passed Checks" in md
        assert "1 check(s) passed" in md

    def test_header(self) -> None:
        md = render_markdown(_sample_result())
        assert md.startswith("# Experiment Audit Report")

    def test_experiment_name_shown(self) -> None:
        md = render_markdown(_sample_result())
        assert "s035_r0" in md

    def test_overall_status_shown(self) -> None:
        md = render_markdown(_sample_result())
        assert "**Overall status:** FAIL" in md
