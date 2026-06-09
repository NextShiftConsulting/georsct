"""JSON and Markdown report generators for experiment audit results."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .models import AuditResult, CheckResult


def render_json(result: AuditResult) -> str:
    """Return a JSON string summarising the audit result.

    Args:
        result: Aggregated audit result to render.

    Returns:
        JSON string with overall_status, summary counts, metadata,
        generated_at timestamp, and individual checks.
    """
    payload = {
        "overall_status": result.overall_status(),
        "summary": result.summary_counts(),
        "metadata": result.metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": [c.to_dict() for c in result.checks],
    }
    return json.dumps(payload, indent=2)


def render_markdown(result: AuditResult) -> str:
    """Return a Markdown string summarising the audit result.

    Args:
        result: Aggregated audit result to render.

    Returns:
        Markdown report with header, summary table, failures,
        warnings, and passed-check count.
    """
    counts = result.summary_counts()
    status = result.overall_status()
    generated = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("# Experiment Audit Report")
    lines.append("")

    # Experiment name from metadata (if present)
    name = result.metadata.get("experiment_name", "unknown")
    lines.append(f"**Experiment:** {name}")
    lines.append(f"**Overall status:** {status}")
    lines.append(f"**Generated:** {generated}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Outcome | Count |")
    lines.append("|---------|-------|")
    for outcome in ("PASS", "WARN", "FAIL"):
        lines.append(f"| {outcome} | {counts[outcome]} |")
    lines.append("")

    # Failures
    failures = [c for c in result.checks if c.severity.is_fail()]
    lines.append("## Failures")
    lines.append("")
    if failures:
        for c in failures:
            cell_str = str(c.cell) if c.cell is not None else "-"
            phase_str = c.phase_id if c.phase_id is not None else "-"
            lines.append(
                f"- **{c.severity.value}** | phase: {phase_str} "
                f"| cell: {cell_str} | {c.message}"
            )
    else:
        lines.append("None.")
    lines.append("")

    # Warnings
    warnings = [c for c in result.checks if c.severity.is_warn()]
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for c in warnings:
            cell_str = str(c.cell) if c.cell is not None else "-"
            phase_str = c.phase_id if c.phase_id is not None else "-"
            lines.append(
                f"- **{c.severity.value}** | phase: {phase_str} "
                f"| cell: {cell_str} | {c.message}"
            )
    else:
        lines.append("None.")
    lines.append("")

    # Passed checks
    passed = counts["PASS"]
    lines.append("## Passed Checks")
    lines.append("")
    lines.append(f"{passed} check(s) passed.")
    lines.append("")

    return "\n".join(lines)
