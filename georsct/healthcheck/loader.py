"""Discover, parse, and validate JSON envelopes from a results folder."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import WorkflowSnapshot
from .warnings import DiagnosticWarning


def _classify_phase(phase: str) -> str | None:
    """Map a phase string to a canonical file type."""
    p = phase.lower()
    if p.startswith("certificates_r") or p.startswith("certificates "):
        return "certificates"
    if p.startswith("diagnostics_r"):
        return "diagnostics"
    if "block_admission" in p:
        return "admission"
    if "dgm_routing" in p or p == "6_dgm_routing":
        return "routing"
    if "gearbox" in p:
        return "gearbox"
    if "money_table" in p or "uplift_table" in p:
        return "money_table"
    return None


def _extract_level(phase: str) -> str | None:
    """Extract level (r0, r1, r2, ...) from phase string."""
    p = phase.lower()
    for token in p.replace("_", " ").split():
        if len(token) >= 2 and token[0] == "r" and token[1:].isdigit():
            return token
    return None


def load_folder(
    folder: str | Path,
    strict: bool = False,
) -> WorkflowSnapshot:
    """Load all JSON files from a folder into a WorkflowSnapshot.

    Args:
        folder: Path to folder containing JSON result files.
        strict: If True, raise on malformed or conflicting inputs.

    Returns:
        WorkflowSnapshot with all discovered data.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    snapshot = WorkflowSnapshot()
    seen_phases: dict[str, str] = {}

    json_files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)

    if not json_files:
        if strict:
            raise ValueError(f"No JSON files found in {folder}")
        snapshot.loader_warnings.append("No JSON files found")
        return snapshot

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            msg = f"{DiagnosticWarning.MALFORMED_JSON}: {path.name}: {e}"
            if strict:
                raise ValueError(msg) from e
            snapshot.loader_warnings.append(msg)
            continue

        if not isinstance(data, dict):
            snapshot.loader_warnings.append(
                f"{DiagnosticWarning.MALFORMED_JSON}: {path.name}: not a JSON object"
            )
            continue

        phase = data.get("phase")
        if phase is None:
            snapshot.loader_warnings.append(
                f"{DiagnosticWarning.MISSING_PHASE}: {path.name}"
            )
            continue

        file_type = _classify_phase(str(phase))
        if file_type is None:
            continue

        if phase in seen_phases:
            snapshot.loader_warnings.append(
                f"{DiagnosticWarning.DUPLICATE_PHASE}: {phase} in {path.name} "
                f"(already seen in {seen_phases[phase]}), using latest"
            )
        seen_phases[phase] = path.name

        # Extract metadata
        for key in ("experiment", "preset", "preset_id"):
            if key in data and key not in snapshot.metadata:
                snapshot.metadata[key] = data[key]

        level = _extract_level(str(phase))

        if file_type == "certificates":
            certs = data.get("certificates", [])
            if not certs:
                continue
            lvl = level or "r0"
            lvl = lvl.lower()
            snapshot.certificates[lvl] = certs
            # Extract provenance
            preset_id = data.get("preset_id") or data.get("preset")
            if preset_id:
                snapshot.provenance[lvl] = str(preset_id)

        elif file_type == "diagnostics":
            cells = data.get("cells", [])
            if not cells:
                continue
            lvl = level or "r0"
            lvl = lvl.lower()
            snapshot.diagnostics[lvl] = cells

        elif file_type == "admission":
            snapshot.admission_table = data.get("admission_table", [])
            # Also capture thresholds if present
            if "thresholds" in data:
                snapshot.metadata["admission_thresholds"] = data["thresholds"]
            if "preset" in data:
                snapshot.metadata.setdefault("preset", data["preset"])

        elif file_type == "routing":
            snapshot.routing_table = data

        elif file_type == "gearbox":
            snapshot.gearbox = data.get("cells", [])

        elif file_type == "money_table":
            snapshot.money_table = data.get("money_table", [])

    if not snapshot.certificates:
        msg = "No certificate files found"
        if strict:
            raise ValueError(msg)
        snapshot.loader_warnings.append(msg)

    return snapshot


def get_cells(snapshot: WorkflowSnapshot) -> list[tuple[str, str]]:
    """Extract unique (scenario, target) pairs from all certificates."""
    seen: set[tuple[str, str]] = set()
    for level_certs in snapshot.certificates.values():
        for cert in level_certs:
            scenario = cert.get("scenario")
            target = cert.get("target")
            if scenario and target:
                seen.add((scenario, target))
    return sorted(seen)


def get_cert_for_cell(
    snapshot: WorkflowSnapshot,
    scenario: str,
    target: str,
    level: str,
) -> dict[str, Any] | None:
    """Get certificate for a specific cell at a specific level."""
    certs = snapshot.certificates.get(level, [])
    for cert in certs:
        if cert.get("scenario") == scenario and cert.get("target") == target:
            return cert
    return None


def get_diag_for_cell(
    snapshot: WorkflowSnapshot,
    scenario: str,
    target: str,
    level: str,
) -> dict[str, Any] | None:
    """Get diagnostics for a specific cell at a specific level."""
    diags = snapshot.diagnostics.get(level, [])
    for d in diags:
        if d.get("scenario") == scenario and d.get("target") == target:
            return d
    return None


def get_gearbox_for_cell(
    snapshot: WorkflowSnapshot,
    scenario: str,
    target: str,
) -> dict[str, Any] | None:
    """Get gearbox warmup data for a specific cell."""
    if not snapshot.gearbox:
        return None
    for cell in snapshot.gearbox:
        if cell.get("scenario") == scenario and cell.get("target") == target:
            return cell
    return None


def get_admission_for_cell(
    snapshot: WorkflowSnapshot,
    scenario: str,
    target: str,
) -> list[dict[str, Any]]:
    """Get admission table rows for a specific cell."""
    if not snapshot.admission_table:
        return []
    return [
        row for row in snapshot.admission_table
        if row.get("scenario") == scenario and row.get("target") == target
    ]


def get_routing_for_cell(
    snapshot: WorkflowSnapshot,
    scenario: str,
    target: str,
) -> dict[str, Any] | None:
    """Get DGM routing entry for a specific cell."""
    if not snapshot.routing_table:
        return None
    strategies = snapshot.routing_table.get("strategies", {})
    for strategy_name, strategy_data in strategies.items():
        for row in strategy_data.get("routing_table", []):
            if row.get("scenario") == scenario and row.get("target") == target:
                return row
    return None


def get_money_for_cell(
    snapshot: WorkflowSnapshot,
    scenario: str,
    target: str,
) -> dict[str, Any] | None:
    """Get money table row for a specific cell."""
    if not snapshot.money_table:
        return None
    for row in snapshot.money_table:
        if row.get("scenario") == scenario and row.get("target") == target:
            return row
    return None
