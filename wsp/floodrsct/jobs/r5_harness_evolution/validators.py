"""Patch and harness validators for R5.

Rejects patches that modify frozen fields, leak held-out data, include
target labels, or weaken schema requirements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .harness_schema import HarnessPatch, HarnessVersion


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool = True
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)


# ---------------------------------------------------------------------------
# Frozen-field guard
# ---------------------------------------------------------------------------

FROZEN_PATH_PREFIXES = (
    "frozen",
    "map_renderer",
    "legend",
    "color_scale",
    "image_size",
)


def _check_frozen_fields(patch: HarnessPatch) -> list[str]:
    """Reject any operation that touches frozen components."""
    errors = []
    for op in patch.operations:
        normalized = op.path.strip("/").lower()
        for prefix in FROZEN_PATH_PREFIXES:
            if normalized.startswith(prefix):
                errors.append(
                    f"Patch op '{op.op}' at '{op.path}' modifies frozen "
                    f"component (prefix: {prefix})"
                )
    return errors


# ---------------------------------------------------------------------------
# Allowed-component guard
# ---------------------------------------------------------------------------

ALLOWED_COMPONENTS = {
    "evidence_template", "feature_policy", "rubric", "scenario_memory",
}


def _check_allowed_components(patch: HarnessPatch) -> list[str]:
    errors = []
    for op in patch.operations:
        component = op.target_component()
        if component and component not in ALLOWED_COMPONENTS:
            errors.append(
                f"Patch op '{op.op}' at '{op.path}' targets disallowed "
                f"component '{component}'. Allowed: {ALLOWED_COMPONENTS}"
            )
    return errors


# ---------------------------------------------------------------------------
# Held-out leakage guard
# ---------------------------------------------------------------------------

def _check_heldout_leakage(
    patch: HarnessPatch,
    heldout_ids: set[str],
) -> list[str]:
    """Check that no held-out ZCTA IDs appear in patch values."""
    errors = []
    for op in patch.operations:
        if op.value is None:
            continue
        val_str = str(op.value)
        for hid in heldout_ids:
            if hid in val_str:
                errors.append(
                    f"Held-out ZCTA '{hid}' found in patch op at "
                    f"'{op.path}' -- label leakage"
                )
    return errors


# ---------------------------------------------------------------------------
# Label leakage guard
# ---------------------------------------------------------------------------

_LABEL_PATTERNS = [
    re.compile(r"ZCTA\s+\d{5}\s+is\s+(high|low|medium)\s+risk", re.I),
    re.compile(r"should\s+be\s+(classified|labeled)\s+as\s+Zone", re.I),
    re.compile(r"held.?out.*should", re.I),
]


def _check_label_leakage(patch: HarnessPatch) -> list[str]:
    errors = []
    for op in patch.operations:
        if op.value is None:
            continue
        val_str = str(op.value)
        for pat in _LABEL_PATTERNS:
            if pat.search(val_str):
                errors.append(
                    f"Possible label leakage in patch op at '{op.path}': "
                    f"matches pattern '{pat.pattern}'"
                )
    return errors


# ---------------------------------------------------------------------------
# Schema weakening guard
# ---------------------------------------------------------------------------

def _check_schema_weakening(
    patch: HarnessPatch,
    current: HarnessVersion,
) -> list[str]:
    """Reject patches that weaken output schema requirements."""
    errors = []
    for op in patch.operations:
        path = op.path.strip("/")
        # Don't allow disabling uncertainty requirement
        if "require_uncertainty" in path and op.value is False:
            errors.append(
                "Patch disables require_uncertainty -- schema weakening"
            )
        if "require_evidence_citation" in path and op.value is False:
            errors.append(
                "Patch disables require_evidence_citation -- schema weakening"
            )
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_patch(
    patch: HarnessPatch,
    current_harness: HarnessVersion,
    heldout_ids: set[str] | None = None,
) -> ValidationResult:
    """Run all validation checks on a proposed patch.

    Returns ValidationResult with passed=True if all checks pass.
    """
    result = ValidationResult()

    for msg in _check_frozen_fields(patch):
        result.fail(msg)

    for msg in _check_allowed_components(patch):
        result.fail(msg)

    if heldout_ids:
        for msg in _check_heldout_leakage(patch, heldout_ids):
            result.fail(msg)

    for msg in _check_label_leakage(patch):
        result.fail(msg)

    for msg in _check_schema_weakening(patch, current_harness):
        result.fail(msg)

    return result
