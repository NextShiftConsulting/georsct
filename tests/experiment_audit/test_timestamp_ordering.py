"""Tests for three-tier timestamp ordering verification."""
from __future__ import annotations

import pytest

from georsct.experiment_audit.models import ArtifactRecord, Severity
from georsct.experiment_audit.timestamp_ordering import (
    OrderingRule,
    check_ordering,
)


RULE = OrderingRule(
    before_key="results/s035/diagnostics_r0.json",
    after_key="results/s035/certificates_r0.json",
    description="diagnostics before certificates at each level",
)


def _make_record(
    key: str,
    internal_ts: str | None = None,
    s3_ts: str | None = None,
    exists: bool = True,
) -> ArtifactRecord:
    """Build an ArtifactRecord with optional timestamps."""
    return ArtifactRecord(
        s3_key=key,
        exists=exists,
        internal_timestamp=internal_ts,
        last_modified=s3_ts,
    )


class TestCheckOrdering:
    """Six-scenario test suite for check_ordering."""

    def test_pass_internal_timestamps_correct_order(self):
        """Tier 1: internal timestamps prove correct ordering."""
        before = _make_record(RULE.before_key, internal_ts="2026-01-10T08:00:00Z")
        after = _make_record(RULE.after_key, internal_ts="2026-01-10T09:00:00Z")

        result = check_ordering(RULE, before, after)

        assert result.severity is Severity.PASS
        assert "Internal timestamps confirm" in result.message

    def test_fail_ordering_violation_internal_timestamps(self):
        """Tier 1: internal timestamps prove wrong ordering."""
        before = _make_record(RULE.before_key, internal_ts="2026-01-10T12:00:00Z")
        after = _make_record(RULE.after_key, internal_ts="2026-01-10T08:00:00Z")

        result = check_ordering(RULE, before, after)

        assert result.severity is Severity.FAIL_ORDERING_VIOLATION
        assert "violate ordering" in result.message

    def test_warn_fallback_s3_correct_order(self):
        """Tier 2: S3 timestamps only, looks correct."""
        before = _make_record(RULE.before_key, s3_ts="2026-01-10T08:00:00+00:00")
        after = _make_record(RULE.after_key, s3_ts="2026-01-10T09:00:00+00:00")

        result = check_ordering(RULE, before, after)

        assert result.severity is Severity.WARN_TIMESTAMP_FALLBACK
        assert "suggest correct ordering" in result.message

    def test_warn_fallback_s3_wrong_order(self):
        """Tier 2: S3 timestamps only, looks wrong (possible rerun)."""
        before = _make_record(RULE.before_key, s3_ts="2026-01-10T12:00:00+00:00")
        after = _make_record(RULE.after_key, s3_ts="2026-01-10T08:00:00+00:00")

        result = check_ordering(RULE, before, after)

        assert result.severity is Severity.WARN_TIMESTAMP_FALLBACK
        assert "suggest wrong ordering" in result.message

    def test_warn_unverifiable_no_timestamps(self):
        """Tier 3: no timestamps at all."""
        before = _make_record(RULE.before_key)
        after = _make_record(RULE.after_key)

        result = check_ordering(RULE, before, after)

        assert result.severity is Severity.WARN_TIMESTAMP_UNVERIFIABLE
        assert "Insufficient timestamp" in result.message

    def test_warn_unverifiable_artifact_missing(self):
        """Tier 3: one artifact does not exist."""
        before = _make_record(RULE.before_key, internal_ts="2026-01-10T08:00:00Z")
        after = _make_record(RULE.after_key, exists=False)

        result = check_ordering(RULE, before, after)

        assert result.severity is Severity.WARN_TIMESTAMP_UNVERIFIABLE
        assert "Artifact missing" in result.message
        assert RULE.after_key in result.message
