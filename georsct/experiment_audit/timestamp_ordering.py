"""Three-tier timestamp ordering verification for experiment artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .models import ArtifactRecord, CheckResult, Severity


GATE = "timestamp_ordering"


@dataclass(frozen=True)
class OrderingRule:
    """Declares that *before_key* must have been produced before *after_key*."""

    before_key: str
    after_key: str
    description: str


CANONICAL_ORDERING_RULES = [
    OrderingRule(
        "results/s035/geometry_kappa.json",
        "results/s035/r0_{scenario}.json",
        "geometry_kappa computed before earliest model training",
    ),
    OrderingRule(
        "results/s035/gearbox_warmup.json",
        "results/s035/certificates_r0.json",
        "gearbox_warmup before certificates",
    ),
    OrderingRule(
        "results/s035/diagnostics_r0.json",
        "results/s035/certificates_r0.json",
        "diagnostics before certificates at each level",
    ),
    OrderingRule(
        "results/s035/diagnostics_r1.json",
        "results/s035/certificates_r1.json",
        "diagnostics before certificates at each level",
    ),
    OrderingRule(
        "results/s035/diagnostics_r2.json",
        "results/s035/certificates_r2.json",
        "diagnostics before certificates at each level",
    ),
]


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, handling the ``Z`` suffix."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def check_ordering(
    rule: OrderingRule,
    before: ArtifactRecord,
    after: ArtifactRecord,
) -> CheckResult:
    """Check that *before* was produced before *after* using three-tier logic.

    Tier 1: Both have internal JSON timestamps (authoritative).
    Tier 2: Both have S3 LastModified only (fallback, never hard-fail).
    Tier 3: One or both missing any timestamp (unverifiable).

    Args:
        rule: The ordering rule being verified.
        before: Artifact that should have been produced first.
        after: Artifact that should have been produced second.

    Returns:
        A CheckResult with the appropriate severity.
    """
    name = f"{before.s3_key} -> {after.s3_key}"

    # Tier 3: existence check
    if not before.exists or not after.exists:
        missing = before.s3_key if not before.exists else after.s3_key
        return CheckResult(
            gate=GATE,
            name=name,
            severity=Severity.WARN_TIMESTAMP_UNVERIFIABLE,
            message=f"Artifact missing: {missing}; cannot verify ordering ({rule.description})",
        )

    # Tier 1: internal timestamps
    if before.internal_timestamp is not None and after.internal_timestamp is not None:
        dt_before = _parse_iso(before.internal_timestamp)
        dt_after = _parse_iso(after.internal_timestamp)
        if dt_before <= dt_after:
            return CheckResult(
                gate=GATE,
                name=name,
                severity=Severity.PASS,
                message=f"Internal timestamps confirm ordering ({rule.description})",
            )
        return CheckResult(
            gate=GATE,
            name=name,
            severity=Severity.FAIL_ORDERING_VIOLATION,
            message=(
                f"Internal timestamps violate ordering: "
                f"{before.internal_timestamp} > {after.internal_timestamp} "
                f"({rule.description})"
            ),
        )

    # Tier 2: S3 LastModified fallback
    if before.last_modified is not None and after.last_modified is not None:
        dt_before = _parse_iso(before.last_modified)
        dt_after = _parse_iso(after.last_modified)
        if dt_before <= dt_after:
            return CheckResult(
                gate=GATE,
                name=name,
                severity=Severity.WARN_TIMESTAMP_FALLBACK,
                message=(
                    f"S3 timestamps suggest correct ordering, "
                    f"but reruns may have inverted ({rule.description})"
                ),
            )
        return CheckResult(
            gate=GATE,
            name=name,
            severity=Severity.WARN_TIMESTAMP_FALLBACK,
            message=(
                f"S3 timestamps suggest wrong ordering "
                f"(possible rerun/backfill): "
                f"{before.last_modified} > {after.last_modified} "
                f"({rule.description})"
            ),
        )

    # Tier 3: one or both missing timestamps entirely
    return CheckResult(
        gate=GATE,
        name=name,
        severity=Severity.WARN_TIMESTAMP_UNVERIFIABLE,
        message=f"Insufficient timestamp data to verify ordering ({rule.description})",
    )
