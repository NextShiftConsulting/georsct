"""Core data models for experiment audit checks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    """Severity levels for audit check outcomes."""

    PASS = "PASS"
    WARN_CONTRACT_GAP = "WARN_CONTRACT_GAP"
    WARN_UNDECLARED_ARTIFACT = "WARN_UNDECLARED_ARTIFACT"
    WARN_TIMESTAMP_FALLBACK = "WARN_TIMESTAMP_FALLBACK"
    WARN_TIMESTAMP_UNVERIFIABLE = "WARN_TIMESTAMP_UNVERIFIABLE"
    WARN_SCOPE_EXTRA = "WARN_SCOPE_EXTRA"
    FAIL_MISSING_OUTPUT = "FAIL_MISSING_OUTPUT"
    FAIL_UNRESOLVED_TEMPLATE = "FAIL_UNRESOLVED_TEMPLATE"
    FAIL_ORDERING_VIOLATION = "FAIL_ORDERING_VIOLATION"
    FAIL_CONTENT_INCOMPLETE = "FAIL_CONTENT_INCOMPLETE"
    FAIL_CONTENT_DEGENERATE = "FAIL_CONTENT_DEGENERATE"

    def is_fail(self) -> bool:
        """Return True if this severity represents a failure."""
        return self.value.startswith("FAIL_")

    def is_warn(self) -> bool:
        """Return True if this severity represents a warning."""
        return self.value.startswith("WARN_")


@dataclass(frozen=True)
class CellKey:
    """Identifies a single scenario/target cell in the experiment matrix."""

    scenario: str
    target: str

    def __str__(self) -> str:
        return f"{self.scenario}/{self.target}"

    @classmethod
    def from_string(cls, s: str) -> CellKey:
        """Parse a 'scenario/target' string into a CellKey."""
        scenario, target = s.split("/", maxsplit=1)
        return cls(scenario=scenario, target=target)


@dataclass
class ArtifactRecord:
    """Metadata about a single S3 artifact."""

    s3_key: str
    exists: bool
    size_bytes: int | None = None
    last_modified: str | None = None
    internal_timestamp: str | None = None
    content: dict | None = None

    def best_timestamp(self) -> str | None:
        """Return internal_timestamp if available, else last_modified."""
        return self.internal_timestamp if self.internal_timestamp is not None else self.last_modified


@dataclass
class CheckResult:
    """A single audit check outcome."""

    audit_stage: str
    name: str
    severity: Severity
    message: str
    phase_id: str | None = None
    cell: CellKey | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        d: dict = {
            "audit_stage": self.audit_stage,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "phase_id": self.phase_id,
            "cell": str(self.cell) if self.cell is not None else None,
        }
        return d


@dataclass
class AuditResult:
    """Aggregated result from a full audit run."""

    checks: list[CheckResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary_counts(self) -> dict[str, int]:
        """Return counts of PASS, FAIL, and WARN outcomes."""
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0}
        for check in self.checks:
            if check.severity.is_fail():
                counts["FAIL"] += 1
            elif check.severity.is_warn():
                counts["WARN"] += 1
            else:
                counts["PASS"] += 1
        return counts

    def overall_status(self) -> str:
        """Return 'FAIL' if any fail, 'WARN' if any warn, else 'PASS'."""
        counts = self.summary_counts()
        if counts["FAIL"] > 0:
            return "FAIL"
        if counts["WARN"] > 0:
            return "WARN"
        return "PASS"
