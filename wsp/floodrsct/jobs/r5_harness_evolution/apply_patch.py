"""Guarded patch application for R5 harness evolution.

Applies a validated HarnessPatch to a HarnessVersion, producing a new
HarnessVersion. Rejects any mutation of frozen fields (defense-in-depth
after validators.py).
"""

from __future__ import annotations

import copy
import dataclasses
from datetime import datetime, timezone

from .harness_schema import HarnessPatch, HarnessVersion


class PatchApplicationError(Exception):
    """Raised when a patch cannot be applied safely."""


def _resolve_pointer(obj: dict, pointer: str) -> tuple[dict, str]:
    """Walk a JSON Pointer, return (parent_dict, final_key).

    Raises PatchApplicationError if the path is invalid.
    """
    parts = pointer.strip("/").split("/")
    current = obj
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise PatchApplicationError(
                f"Path segment '{part}' not found in harness"
            )
    return current, parts[-1]


def apply_patch(
    harness: HarnessVersion,
    patch: HarnessPatch,
) -> HarnessVersion:
    """Apply a validated patch to produce a new harness version.

    The input harness is NOT mutated. A deep copy is made first.
    All operations target the editable components dict only.
    """
    h_dict = dataclasses.asdict(harness)
    editable = copy.deepcopy(h_dict["editable"])

    for op in patch.operations:
        path = op.path.strip("/")

        # Defense-in-depth: block frozen paths even if validator missed it
        if path.startswith("frozen"):
            raise PatchApplicationError(
                f"Patch op '{op.op}' at '{op.path}' targets frozen component"
            )

        try:
            parent, key = _resolve_pointer(editable, path)
        except PatchApplicationError:
            if op.op == "add":
                # For 'add', create intermediate dicts
                parts = path.split("/")
                current = editable
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = op.value
                continue
            raise

        if op.op == "add":
            parent[key] = op.value
        elif op.op == "replace":
            if key not in parent:
                raise PatchApplicationError(
                    f"Cannot replace '{key}' -- does not exist"
                )
            parent[key] = op.value
        elif op.op == "remove":
            if key not in parent:
                raise PatchApplicationError(
                    f"Cannot remove '{key}' -- does not exist"
                )
            del parent[key]
        else:
            raise PatchApplicationError(f"Unknown op: '{op.op}'")

    # Build new harness version
    new_harness = copy.deepcopy(harness)
    # Reconstruct editable from modified dict
    new_harness.editable.evidence_template.__dict__.update(
        editable.get("evidence_template", {})
    )
    new_harness.editable.feature_policy.__dict__.update(
        editable.get("feature_policy", {})
    )
    new_harness.editable.rubric.__dict__.update(
        editable.get("rubric", {})
    )
    new_harness.editable.scenario_memory.__dict__.update(
        editable.get("scenario_memory", {})
    )

    new_harness.harness_id = patch.to_harness
    new_harness.parent_harness_id = patch.from_harness
    new_harness.created_at = datetime.now(timezone.utc).isoformat()
    new_harness.change_summary = patch.evolver_rationale

    return new_harness
