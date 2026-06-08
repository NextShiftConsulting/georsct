"""Admission invariants for INVALID certificates.

These tests protect the boundary between certificate validity and
gate enforcement.

Core rule:
    INVALID certificates are never gate-evaluated.
    They are quarantined before the gate with diagnosis/evidence attached.
"""

import pytest


def assert_invalid_certificate_invariant(row: dict) -> None:
    """Assert the required admission contract for INVALID certificates."""
    assert row["certificate_status"] == "INVALID"

    # INVALID means the gate was never reached.
    assert row["enforcement_decision"] == "SKIP"
    assert row["gate_reached"] == "PRE_GATE_INVALID"

    # INVALID must be diagnosable.
    assert row.get("invalid_diagnosis") is not None
    assert row["invalid_diagnosis"].get("failure_mode")
    assert row["invalid_diagnosis"].get("remediation_action")

    # Evidence must be preserved for audit/remediation.
    assert row.get("certificate_evidence") is not None

    # INVALID rows must not masquerade as gate outcomes.
    assert row.get("gate_decision") in (None, "NOT_EVALUATED")
    assert row.get("admitted") in (None, False)


def assert_valid_gate_evaluated_invariant(row: dict) -> None:
    """Assert the required admission contract for gate-evaluated VALID certificates."""
    assert row.get("certificate_status") in ("VALID", None)
    # Gate-evaluated rows reach an actual gate, not PRE_GATE_INVALID
    assert row["gate_reached"] != "PRE_GATE_INVALID"

    # VALID gate-evaluated rows should not carry invalid diagnosis payloads.
    assert row.get("invalid_diagnosis") in (None, {})

    # Gate-evaluated rows must have an actual enforcement decision.
    assert row["enforcement_decision"] in {
        "EXECUTE", "REJECT", "BLOCK", "RE_ENCODE", "REPAIR", "WARN",
    }


@pytest.mark.parametrize(
    "failure_mode,remediation_action",
    [
        ("SOLVER_BOTH_NULL", "retrain-with-larger-sample"),
        ("STRUCTURAL_NO_FEATURES", "check-feature-registry"),
        ("SOLVER_PRIMARY_NULL", "investigate-solver-crash"),
    ],
)
def test_invalid_certificates_are_pre_gate_skips(
    failure_mode: str,
    remediation_action: str,
) -> None:
    row = {
        "certificate_status": "INVALID",
        "enforcement_decision": "SKIP",
        "gate_reached": "PRE_GATE_INVALID",
        "invalid_diagnosis": {
            "failure_mode": failure_mode,
            "remediation_action": remediation_action,
            "partial_signals": {},
        },
        "certificate_evidence": {
            "independent_signal": {"metric": None, "solver_used": None},
        },
        "gate_decision": "NOT_EVALUATED",
        "admitted": False,
    }

    assert_invalid_certificate_invariant(row)


def test_invalid_certificate_preserves_partial_signals_without_admission() -> None:
    row = {
        "certificate_status": "INVALID",
        "enforcement_decision": "SKIP",
        "gate_reached": "PRE_GATE_INVALID",
        "invalid_diagnosis": {
            "failure_mode": "SOLVER_PRIMARY_NULL",
            "remediation_action": "investigate-solver-crash",
            "partial_signals": {
                "full_R2_metric": 0.31,
                "drop_block_metric": 0.18,
                "ablation_delta": 0.13,
            },
        },
        "certificate_evidence": {
            "independent_signal": {"metric": None, "solver_used": None},
            "marginal_signal": {"delta": 0.13},
        },
        "gate_decision": "NOT_EVALUATED",
        "admitted": False,
    }

    assert_invalid_certificate_invariant(row)

    # Partial signal is audit evidence only. It is not an admission decision.
    assert row["invalid_diagnosis"]["partial_signals"]["ablation_delta"] > 0
    assert row["enforcement_decision"] == "SKIP"
    assert row["gate_reached"] == "PRE_GATE_INVALID"
    assert row["admitted"] is False


def test_invalid_certificate_is_not_gate_evaluated_even_when_metrics_would_pass() -> None:
    """Regression test against confusing INVALID with ADMIT.

    Even if surviving metrics look strong, INVALID status dominates.
    The gate must not be reached.
    """
    row = {
        "certificate_status": "INVALID",
        "enforcement_decision": "SKIP",
        "gate_reached": "PRE_GATE_INVALID",
        "invalid_diagnosis": {
            "failure_mode": "SOLVER_PRIMARY_NULL",
            "remediation_action": "investigate-solver-crash",
            "partial_signals": {
                "full_R2_metric": 0.85,
                "drop_block_metric": 0.20,
                "ablation_delta": 0.65,
            },
        },
        "certificate_evidence": {
            "independent_signal": {"metric": None},
            "marginal_signal": {"delta": 0.65},
        },
        "gate_decision": "NOT_EVALUATED",
        "admitted": False,
    }

    assert_invalid_certificate_invariant(row)


@pytest.mark.parametrize(
    "enforcement_decision",
    ["EXECUTE", "REJECT", "BLOCK", "RE_ENCODE", "REPAIR"],
)
def test_valid_gate_evaluated_certificates_have_no_invalid_diagnosis(
    enforcement_decision: str,
) -> None:
    row = {
        "certificate_status": "VALID",
        "enforcement_decision": enforcement_decision,
        "gate_reached": "ALL_PASSED" if enforcement_decision == "EXECUTE" else "GATE_1_INTEGRITY",
        "invalid_diagnosis": None,
        "certificate_evidence": {
            "independent_signal": {"metric": 0.35, "solver_used": "histgbdt"},
        },
    }

    assert_valid_gate_evaluated_invariant(row)


def test_gate_evaluated_requires_valid_certificate() -> None:
    """A row cannot be both INVALID and gate-evaluated."""
    row = {
        "certificate_status": "INVALID",
        "enforcement_decision": "SKIP",
        "gate_reached": "PRE_GATE_INVALID",
        "invalid_diagnosis": {
            "failure_mode": "SOLVER_BOTH_NULL",
            "remediation_action": "retrain-with-larger-sample",
            "partial_signals": {},
        },
        "certificate_evidence": {"independent_signal": {"metric": None}},
        "gate_decision": "NOT_EVALUATED",
        "admitted": False,
    }

    assert row["gate_reached"] != "GATE_EVALUATED"
    assert_invalid_certificate_invariant(row)


def test_invalid_certificate_requires_diagnosis() -> None:
    """INVALID without diagnosis is a contract violation."""
    row = {
        "certificate_status": "INVALID",
        "enforcement_decision": "SKIP",
        "gate_reached": "PRE_GATE_INVALID",
        "invalid_diagnosis": None,
        "certificate_evidence": {"independent_signal": {"metric": None}},
        "gate_decision": "NOT_EVALUATED",
        "admitted": False,
    }

    with pytest.raises(AssertionError):
        assert_invalid_certificate_invariant(row)


def test_invalid_certificate_requires_evidence_preservation() -> None:
    """INVALID without preserved evidence is a contract violation."""
    row = {
        "certificate_status": "INVALID",
        "enforcement_decision": "SKIP",
        "gate_reached": "PRE_GATE_INVALID",
        "invalid_diagnosis": {
            "failure_mode": "STRUCTURAL_NO_FEATURES",
            "remediation_action": "check-feature-registry",
            "partial_signals": {},
        },
        "certificate_evidence": None,
        "gate_decision": "NOT_EVALUATED",
        "admitted": False,
    }

    with pytest.raises(AssertionError):
        assert_invalid_certificate_invariant(row)
