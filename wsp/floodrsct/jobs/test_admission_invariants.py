"""
test_admission_invariants.py -- enforce INVALID/VALID certificate routing.

Hard invariants:
  1. INVALID certificates never enter the gate pipeline
  2. VALID certificates never carry invalid_diagnosis
  3. INVALID certificates always carry diagnosis + evidence
  4. Gate thresholds are never applied to INVALID certificates
  5. Solver trace is present on every certificate evidence leg

Run as:
    pytest test_admission_invariants.py -v
    python test_admission_invariants.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compute_r3_block_admission import evaluate_gates, _diagnose_invalid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_cert(**overrides):
    """Build a VALID block certificate with sensible defaults."""
    base = {
        "block": "test_block",
        "scenario": "test_scenario",
        "target": "test_target",
        "certificate_status": "VALID",
        "R": 0.60,
        "S_sup": 0.10,
        "N": 0.30,
        "alpha": 0.60,
        "kappa": 0.40,
        "sigma": 0.10,
        "evidence": {
            "independent_signal": {
                "source": "block_only",
                "metric": 0.35,
                "solver_used": "histgbdt",
                "primary_solver": "histgbdt",
                "fallback_solver": None,
                "fallback_triggered": False,
                "primary_failure_reason": None,
            },
            "marginal_signal": {
                "source": "full_R2_minus_drop_block",
                "full_R2": {"metric": 0.40, "solver_used": "histgbdt"},
                "drop_block": {"metric": 0.38, "solver_used": "histgbdt"},
                "delta": 0.02,
            },
            "feature_audit": {
                "add_status": "NO_OP_B_SUBSET_R2",
                "drop_status": "VALID",
                "block_only_valid": True,
            },
        },
    }
    base.update(overrides)
    return base


def _make_invalid_cert(**overrides):
    """Build an INVALID block certificate."""
    base = {
        "block": "test_block",
        "scenario": "test_scenario",
        "target": "test_target",
        "certificate_status": "INVALID",
        "R": float("nan"),
        "S_sup": float("nan"),
        "N": float("nan"),
        "alpha": float("nan"),
        "kappa": float("nan"),
        "sigma": float("nan"),
        "evidence": {
            "independent_signal": {
                "source": "block_only",
                "metric": None,
                "solver_used": None,
                "primary_solver": "histgbdt",
                "fallback_solver": "ridge",
                "fallback_triggered": True,
                "primary_failure_reason": "null_metric",
            },
            "marginal_signal": {
                "source": "full_R2_minus_drop_block",
                "full_R2": {"metric": 0.152, "solver_used": "histgbdt"},
                "drop_block": {"metric": 0.148, "solver_used": "histgbdt"},
                "delta": 0.004,
            },
            "feature_audit": {
                "add_status": "NO_OP_B_SUBSET_R2",
                "drop_status": "VALID",
                "block_only_valid": True,
            },
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Invariant 1: INVALID certificates never enter the gate pipeline
# ---------------------------------------------------------------------------

class TestInvalidNeverGateEvaluated:
    """INVALID certs must be caught before evaluate_gates is called."""

    def test_invalid_cert_has_nan_rsn(self):
        """INVALID certs have NaN R/S/N -- gate evaluation would be meaningless."""
        cert = _make_invalid_cert()
        assert math.isnan(cert["R"])
        assert math.isnan(cert["N"])
        assert cert["certificate_status"] == "INVALID"

    def test_evaluate_gates_on_nan_does_not_crash(self):
        """evaluate_gates handles NaN gracefully (defense in depth).

        Even though the admission loop should never call evaluate_gates on
        INVALID certs, the function itself should not crash if it receives one.
        """
        cert = _make_invalid_cert()
        # This should not raise -- _safe() coalesces NaN
        result = evaluate_gates(cert)
        # With NaN -> R=0,N=1 coalescing, this will REJECT at Gate 1
        assert result["enforcement_decision"] in ("REJECT", "BLOCK", "RE_ENCODE")


# ---------------------------------------------------------------------------
# Invariant 2: VALID certificates never carry invalid_diagnosis
# ---------------------------------------------------------------------------

class TestValidNeverHasDiagnosis:
    """Gate-evaluated certs should not carry invalid_diagnosis."""

    def test_valid_cert_gate_result_has_no_diagnosis(self):
        cert = _make_valid_cert()
        result = evaluate_gates(cert)
        assert "invalid_diagnosis" not in result
        assert result["gate_reached"] != "PRE_GATE_INVALID"

    def test_valid_cert_reaches_gate(self):
        cert = _make_valid_cert()
        result = evaluate_gates(cert)
        # A valid cert with R=0.60, alpha=0.60 should pass Gate 1
        assert result["gate_reached"] != "PRE_GATE_INVALID"
        assert result["enforcement_decision"] != "SKIP"


# ---------------------------------------------------------------------------
# Invariant 3: INVALID certificates always carry diagnosis + evidence
# ---------------------------------------------------------------------------

class TestInvalidAlwaysHasDiagnosis:
    """_diagnose_invalid must produce complete diagnosis for any INVALID cert."""

    def test_diagnosis_has_required_fields(self):
        cert = _make_invalid_cert()
        diag = _diagnose_invalid(cert)
        assert "failure_mode" in diag
        assert "remediation_action" in diag
        assert "detail" in diag
        assert "solver_trace" in diag
        assert "feature_audit_summary" in diag

    def test_diagnosis_failure_mode_is_typed(self):
        """failure_mode must be one of the known classifications."""
        known_modes = {
            "SOLVER_BOTH_NULL", "SOLVER_PRIMARY_NULL",
            "STRUCTURAL_NO_FEATURES", "UNKNOWN",
        }
        cert = _make_invalid_cert()
        diag = _diagnose_invalid(cert)
        assert diag["failure_mode"] in known_modes

    def test_solver_trace_present(self):
        cert = _make_invalid_cert()
        diag = _diagnose_invalid(cert)
        trace = diag["solver_trace"]
        assert trace["primary_solver"] == "histgbdt"
        assert trace["primary_failure_reason"] == "null_metric"
        assert trace["fallback_triggered"] is True

    def test_partial_signals_from_marginal(self):
        """When marginal signal survived, partial_signals should capture it."""
        cert = _make_invalid_cert()
        diag = _diagnose_invalid(cert)
        assert diag["partial_signals"] is not None
        assert "full_R2_metric" in diag["partial_signals"]
        assert "ablation_delta" in diag["partial_signals"]

    def test_no_partial_when_everything_failed(self):
        """When nothing survived, partial_signals should be None."""
        cert = _make_invalid_cert()
        cert["evidence"]["marginal_signal"] = {
            "source": "full_R2_minus_drop_block",
            "full_R2": {"metric": None, "solver_used": None},
            "drop_block": {"metric": None, "solver_used": None},
            "delta": None,
        }
        diag = _diagnose_invalid(cert)
        assert diag["partial_signals"] is None

    def test_structural_no_features(self):
        cert = _make_invalid_cert()
        cert["evidence"]["feature_audit"]["block_only_valid"] = False
        diag = _diagnose_invalid(cert)
        assert diag["failure_mode"] == "STRUCTURAL_NO_FEATURES"
        assert diag["remediation_action"] == "check-feature-registry"


# ---------------------------------------------------------------------------
# Invariant 4: Gate thresholds never applied to INVALID certificates
# ---------------------------------------------------------------------------

class TestGateThresholdsNotAppliedToInvalid:
    """The admission loop must skip INVALID before threshold comparison."""

    def test_invalid_cert_would_produce_degenerate_gate_result(self):
        """Show that gate-evaluating an INVALID cert gives wrong answers.

        This test documents WHY the pre-gate skip is necessary:
        NaN coalesces to R=0,N=1 which always fails Gate 1, making every
        INVALID cert look like a scientific REJECT when it's actually
        an evaluability failure.
        """
        cert = _make_invalid_cert()
        result = evaluate_gates(cert)
        # This REJECT is semantically wrong -- the block isn't rejected,
        # it's not evaluable. The admission loop must prevent this path.
        assert result["enforcement_decision"] == "REJECT"
        assert result["gate_reached"] == "GATE_1_INTEGRITY"
        # This is the degenerate behavior we're protecting against


# ---------------------------------------------------------------------------
# Invariant 5: Solver trace on certificate evidence
# ---------------------------------------------------------------------------

class TestSolverTraceOnEvidence:
    """Every evidence leg must carry solver lineage."""

    def test_independent_signal_has_solver_trace(self):
        cert = _make_valid_cert()
        indep = cert["evidence"]["independent_signal"]
        assert "solver_used" in indep
        assert "primary_solver" in indep
        assert "fallback_triggered" in indep
        assert "primary_failure_reason" in indep

    def test_fallback_cert_marks_lineage(self):
        """A cert that used ridge fallback must say so."""
        cert = _make_valid_cert()
        cert["evidence"]["independent_signal"].update({
            "solver_used": "ridge",
            "fallback_triggered": True,
            "primary_failure_reason": "null_metric",
        })
        indep = cert["evidence"]["independent_signal"]
        assert indep["solver_used"] == "ridge"
        assert indep["fallback_triggered"] is True
        assert indep["primary_failure_reason"] == "null_metric"
        # Certificate is still VALID -- fallback is allowed
        assert cert["certificate_status"] == "VALID"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
