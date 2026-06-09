"""Six sequential audit gates for experiment contract verification."""
from __future__ import annotations

from .constants import LEVEL_PREFIXES
from .contract import ParsedContract
from .models import ArtifactRecord, CellKey, CheckResult, Severity
from .timestamp_ordering import CANONICAL_ORDERING_RULES, check_ordering
from .content_readers import MetricStatus, extract_aggregate_cells, extract_money_table_cells


# ---------------------------------------------------------------------------
# Gate 1: Contract Parse
# ---------------------------------------------------------------------------

def gate_1_contract_parse(contract: ParsedContract) -> list[CheckResult]:
    """Check that the contract has no unresolved template variables.

    Args:
        contract: A parsed experiment contract.

    Returns:
        One FAIL per unresolved template, or a single PASS.
    """
    issues = contract.find_unresolved_templates()
    if not issues:
        return [
            CheckResult(
                audit_stage="contract_parse",
                name="template_resolution",
                severity=Severity.PASS,
                message="All template variables resolved",
            )
        ]
    return [
        CheckResult(
            audit_stage="contract_parse",
            name="unresolved_template",
            severity=Severity.FAIL_UNRESOLVED_TEMPLATE,
            message=issue,
        )
        for issue in issues
    ]


# ---------------------------------------------------------------------------
# Gate 2: Declared Output Existence
# ---------------------------------------------------------------------------

def gate_2_declared_outputs(
    contract: ParsedContract,
    inventory: dict[str, ArtifactRecord],
) -> list[CheckResult]:
    """Check that every declared output exists in the artifact inventory.

    Args:
        contract: A parsed experiment contract.
        inventory: Map of S3 key to ArtifactRecord.

    Returns:
        Check results for each phase/output combination.
    """
    results: list[CheckResult] = []

    for phase in contract.phases:
        if not phase.has_declared_outputs:
            results.append(
                CheckResult(
                    audit_stage="declared_outputs",
                    name=f"phase_{phase.phase_id}_no_outputs",
                    severity=Severity.WARN_CONTRACT_GAP,
                    message=(
                        f"Phase {phase.phase_id} has no declared outputs "
                        f"-- cannot verify post-flight"
                    ),
                    phase_id=phase.phase_id,
                )
            )
            continue

        resolved_keys = phase.resolve_output_keys(contract.scenarios)
        for key in resolved_keys:
            record = inventory.get(key)
            if record is None or not record.exists:
                results.append(
                    CheckResult(
                        audit_stage="declared_outputs",
                        name=f"output_{key}",
                        severity=Severity.FAIL_MISSING_OUTPUT,
                        message=f"Declared output missing: {key}",
                        phase_id=phase.phase_id,
                    )
                )
            else:
                results.append(
                    CheckResult(
                        audit_stage="declared_outputs",
                        name=f"output_{key}",
                        severity=Severity.PASS,
                        message=f"Output exists: {key}",
                        phase_id=phase.phase_id,
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Gate 3: Convention Supplement
# ---------------------------------------------------------------------------

# Hardcoded convention patterns for phases that may lack declared outputs.
# Each entry: (phase_id, list of (key_suffix, per_scenario))
_CONVENTION_MAP: dict[str, list[tuple[str, bool]]] = {
    "diagnostics_r0": [("diagnostics_r0.json", False)],
    "diagnostics_r1": [("diagnostics_r1.json", False)],
    "diagnostics_r2": [("diagnostics_r2.json", False)],
    "certificates_r0": [
        ("certificates_r0.json", False),
        ("certificates_r0.parquet", False),
    ],
    "certificates_r1": [
        ("certificates_r1.json", False),
        ("certificates_r1.parquet", False),
    ],
    "certificates_r2": [
        ("certificates_r2.json", False),
        ("certificates_r2.parquet", False),
    ],
    "uplift_table": [("money_table.json", False)],
    "dgm_routing": [("dgm_routing.json", False)],
    "r1_hydrology": [
        ("r1_hydrology_{scenario}.json", True),
        ("r1_hydrology_{scenario}_predictions.parquet", True),
    ],
    "r2_temporal": [
        ("r2_{scenario}.json", True),
        ("r2_{scenario}_predictions.parquet", True),
    ],
}


def gate_3_convention_supplement(
    contract: ParsedContract,
    s3_keys: list[str],
    results_prefix: str = "results/s035/",
) -> list[CheckResult]:
    """Check convention-based outputs for phases without declared outputs.

    Args:
        contract: A parsed experiment contract.
        s3_keys: List of S3 keys found under the results prefix.
        results_prefix: The S3 prefix for result artifacts.

    Returns:
        Check results for convention-based outputs and undeclared artifacts.
    """
    results: list[CheckResult] = []
    s3_set = set(s3_keys)
    expected_keys: set[str] = set()

    # Collect all declared output keys so we can detect undeclared artifacts
    for phase in contract.phases:
        if phase.has_declared_outputs:
            for key in phase.resolve_output_keys(contract.scenarios):
                expected_keys.add(key)

    for phase_id, conventions in _CONVENTION_MAP.items():
        phase = contract.get_phase(phase_id)
        # Only apply conventions when the phase has no declared outputs
        has_outputs = phase is not None and phase.has_declared_outputs
        if has_outputs:
            continue

        for suffix, per_scenario in conventions:
            if per_scenario:
                for sc in contract.scenarios:
                    key = results_prefix + suffix.replace("{scenario}", sc)
                    expected_keys.add(key)
                    if key in s3_set:
                        results.append(
                            CheckResult(
                                audit_stage="convention_supplement",
                                name=f"convention_{key}",
                                severity=Severity.WARN_CONTRACT_GAP,
                                message=(
                                    f"Phase {phase_id} produced {key}, "
                                    f"but contract has no outputs declaration"
                                ),
                                phase_id=phase_id,
                            )
                        )
                    else:
                        results.append(
                            CheckResult(
                                audit_stage="convention_supplement",
                                name=f"convention_{key}",
                                severity=Severity.FAIL_MISSING_OUTPUT,
                                message=f"Convention output missing: {key}",
                                phase_id=phase_id,
                            )
                        )
            else:
                key = results_prefix + suffix
                expected_keys.add(key)
                if key in s3_set:
                    results.append(
                        CheckResult(
                            audit_stage="convention_supplement",
                            name=f"convention_{key}",
                            severity=Severity.WARN_CONTRACT_GAP,
                            message=(
                                f"Phase {phase_id} produced {key}, "
                                f"but contract has no outputs declaration"
                            ),
                            phase_id=phase_id,
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            audit_stage="convention_supplement",
                            name=f"convention_{key}",
                            severity=Severity.FAIL_MISSING_OUTPUT,
                            message=f"Convention output missing: {key}",
                            phase_id=phase_id,
                        )
                    )

    # Scan for undeclared artifacts
    for key in sorted(s3_set):
        if key.startswith(results_prefix) and key not in expected_keys:
            results.append(
                CheckResult(
                    audit_stage="convention_supplement",
                    name=f"undeclared_{key}",
                    severity=Severity.WARN_UNDECLARED_ARTIFACT,
                    message=f"Undeclared artifact on S3: {key}",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Gate 4: Cell Matrix
# ---------------------------------------------------------------------------

def gate_4_cell_matrix(
    contract: ParsedContract,
    actual_cells: set[CellKey],
) -> list[CheckResult]:
    """Compare contracted cell matrix against actual cells observed.

    Args:
        contract: A parsed experiment contract.
        actual_cells: Set of CellKeys found in actual results.

    Returns:
        PASS for matched cells, FAIL for missing, WARN for extras.
    """
    results: list[CheckResult] = []
    contracted = contract.cell_matrix

    for cell in sorted(contracted, key=str):
        if cell in actual_cells:
            results.append(
                CheckResult(
                    audit_stage="cell_matrix",
                    name=f"cell_{cell}",
                    severity=Severity.PASS,
                    message=f"Cell present: {cell}",
                    cell=cell,
                )
            )
        else:
            results.append(
                CheckResult(
                    audit_stage="cell_matrix",
                    name=f"cell_{cell}",
                    severity=Severity.FAIL_MISSING_OUTPUT,
                    message=f"Contracted cell missing from actuals: {cell}",
                    cell=cell,
                )
            )

    for cell in sorted(actual_cells, key=str):
        if cell not in contracted:
            results.append(
                CheckResult(
                    audit_stage="cell_matrix",
                    name=f"cell_{cell}",
                    severity=Severity.WARN_SCOPE_EXTRA,
                    message=f"Cell in actuals but not in contract: {cell}",
                    cell=cell,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Gate 5: Aggregate Content
# ---------------------------------------------------------------------------

def gate_5_aggregate_content(
    contract: ParsedContract,
    artifacts: dict[str, ArtifactRecord],
) -> list[CheckResult]:
    """Check aggregate content artifacts for completeness and validity.

    Inspects money_table.json for contracted cells and null metrics,
    and geometry_kappa.json for null kappa_geom values.

    Args:
        contract: A parsed experiment contract.
        artifacts: Map of S3 key to ArtifactRecord with loaded content.

    Returns:
        Check results for content completeness and degeneracy.
    """
    results: list[CheckResult] = []
    contracted = contract.cell_matrix

    # --- money_table.json ---
    money_keys = [k for k in artifacts if k.endswith("money_table.json")]
    for money_key in money_keys:
        record = artifacts[money_key]
        if record.content is None:
            results.append(
                CheckResult(
                    audit_stage="aggregate_content",
                    name="money_table_content",
                    severity=Severity.FAIL_CONTENT_INCOMPLETE,
                    message=f"Cannot read content of {money_key}",
                )
            )
            continue

        cell_metrics = extract_money_table_cells(record.content)
        found_cells = {cm.cell for cm in cell_metrics}

        for cell in sorted(contracted, key=str):
            if cell not in found_cells:
                results.append(
                    CheckResult(
                        audit_stage="aggregate_content",
                        name=f"money_table_missing_{cell}",
                        severity=Severity.FAIL_CONTENT_INCOMPLETE,
                        message=f"Contracted cell {cell} missing from money table",
                        cell=cell,
                    )
                )

        for cm in cell_metrics:
            if cm.status == MetricStatus.NULL:
                results.append(
                    CheckResult(
                        audit_stage="aggregate_content",
                        name=f"money_table_null_{cm.cell}",
                        severity=Severity.FAIL_CONTENT_DEGENERATE,
                        message=f"Null metric for cell {cm.cell} in money table",
                        cell=cm.cell,
                    )
                )

    # --- geometry_kappa.json ---
    kappa_keys = [k for k in artifacts if k.endswith("geometry_kappa.json")]
    for kappa_key in kappa_keys:
        record = artifacts[kappa_key]
        if record.content is None:
            results.append(
                CheckResult(
                    audit_stage="aggregate_content",
                    name="geometry_kappa_content",
                    severity=Severity.WARN_CONTRACT_GAP,
                    message=f"Cannot read content of {kappa_key}",
                )
            )
            continue

        cell_metrics = extract_aggregate_cells(record.content, "kappa_geom")
        for cm in cell_metrics:
            if cm.status == MetricStatus.NULL:
                results.append(
                    CheckResult(
                        audit_stage="aggregate_content",
                        name=f"geometry_kappa_null_{cm.cell}",
                        severity=Severity.WARN_CONTRACT_GAP,
                        message=f"Null kappa_geom for cell {cm.cell}",
                        cell=cm.cell,
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Gate 6: Ordering
# ---------------------------------------------------------------------------

def gate_6_ordering(
    artifacts: dict[str, ArtifactRecord],
    scenarios: list[str] | None = None,
) -> list[CheckResult]:
    """Apply canonical ordering rules and verify timestamp ordering.

    Expands per-scenario rules by replacing {scenario} placeholders
    with each scenario name.

    Args:
        artifacts: Map of S3 key to ArtifactRecord.
        scenarios: List of scenario names for per-scenario expansion.
            If None, no per-scenario expansion is performed.

    Returns:
        Check results from ordering verification.
    """
    results: list[CheckResult] = []
    expanded_scenarios = scenarios or []

    for rule in CANONICAL_ORDERING_RULES:
        has_scenario_placeholder = "{scenario}" in rule.before_key or "{scenario}" in rule.after_key

        if has_scenario_placeholder and expanded_scenarios:
            for sc in expanded_scenarios:
                before_key = rule.before_key.replace("{scenario}", sc)
                after_key = rule.after_key.replace("{scenario}", sc)
                before = artifacts.get(before_key)
                after = artifacts.get(after_key)
                if before is None:
                    before = ArtifactRecord(s3_key=before_key, exists=False)
                if after is None:
                    after = ArtifactRecord(s3_key=after_key, exists=False)
                expanded_rule = type(rule)(
                    before_key=before_key,
                    after_key=after_key,
                    description=rule.description,
                )
                results.append(check_ordering(expanded_rule, before, after))
        elif not has_scenario_placeholder:
            before = artifacts.get(rule.before_key)
            after = artifacts.get(rule.after_key)
            if before is None:
                before = ArtifactRecord(s3_key=rule.before_key, exists=False)
            if after is None:
                after = ArtifactRecord(s3_key=rule.after_key, exists=False)
            results.append(check_ordering(rule, before, after))

    return results
