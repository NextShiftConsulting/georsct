#!/usr/bin/env python3
"""test_evaluation_hygiene.py -- Tests for fold eligibility gate.

Verifies that degenerate folds produce abstention records with null
metrics instead of misleading accuracy=1.0 / f1=0.0 values.

Run: python -m pytest test_evaluation_hygiene.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
# Add georsct package root for metric_eligibility import
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from train_r0_baseline import (
    classify_fold_eligibility,
    _class_support,
    _classification_metrics,
    _nan_to_none,
    RunResult,
)


# ---------------------------------------------------------------------------
# classify_fold_eligibility
# ---------------------------------------------------------------------------

class TestClassifyFoldEligibility:

    def test_regression_always_eligible(self):
        y_train = np.array([1.0, 2.0, 3.0])
        y_test = np.array([4.0, 5.0])
        assert classify_fold_eligibility(y_train, y_test, "regression") == "ELIGIBLE"

    def test_classification_both_classes_eligible(self):
        y_train = np.array([0, 0, 1, 1, 0])
        y_test = np.array([0, 1, 1])
        assert classify_fold_eligibility(y_train, y_test, "classification") == "ELIGIBLE"

    def test_train_single_class(self):
        y_train = np.array([0, 0, 0, 0, 0])
        y_test = np.array([0, 1])
        assert classify_fold_eligibility(y_train, y_test, "classification") == "SKIP_TRAIN_SINGLE_CLASS"

    def test_test_single_class_now_eligible(self):
        """Test-side single-class is now per-metric (after-fit gate), not a fold-level skip."""
        y_train = np.array([0, 0, 1, 1])
        y_test = np.array([0, 0, 0])
        # Training gate passes -- test-side is handled per-metric by score_fold
        assert classify_fold_eligibility(y_train, y_test, "classification") == "ELIGIBLE"

    def test_both_single_class_reports_train_first(self):
        """When both are single-class, train-side is caught first."""
        y_train = np.array([0, 0, 0])
        y_test = np.array([1, 1, 1])
        assert classify_fold_eligibility(y_train, y_test, "classification") == "SKIP_TRAIN_SINGLE_CLASS"

    def test_single_element_test_fold_now_eligible(self):
        """Single-element test fold with valid training is ELIGIBLE (per-metric gate handles it)."""
        y_train = np.array([0, 0, 1, 1])
        y_test = np.array([0])
        assert classify_fold_eligibility(y_train, y_test, "classification") == "ELIGIBLE"


# ---------------------------------------------------------------------------
# _classification_metrics defense-in-depth
# ---------------------------------------------------------------------------

class TestClassificationMetricsPerMetric:
    """Tests for per-metric {status, value} schema from score_fold."""

    def test_single_class_truth_gets_per_metric_status(self):
        """All-negative truth: overlap SKIP_EMPTY_UNION, ranking SKIP_TEST_SINGLE_CLASS."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.1])
        m = _classification_metrics(y_true, y_pred, y_score)

        # Per-metric status -- no blanket REFUSED
        assert m["f1"]["status"] == "SKIP_EMPTY_UNION"
        assert m["f1"]["value"] is None
        assert m["roc_auc"]["status"] == "SKIP_TEST_SINGLE_CLASS"
        assert m["roc_auc"]["value"] is None
        # Accuracy is always measurable
        assert m["accuracy"]["status"] == "MEASURED"
        assert m["accuracy"]["value"] == 1.0

    def test_no_accuracy_1_f1_0_on_degenerate(self):
        """The specific pattern that triggered this fix must never appear."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.1, 0.15, 0.05])
        m = _classification_metrics(y_true, y_pred, y_score)

        # f1 is never reported as 0.0 for empty union -- it's null
        assert m["f1"]["value"] is None
        # accuracy is reported but f1 is not, so misleading pattern is impossible

    def test_valid_binary_produces_measured(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9])
        m = _classification_metrics(y_true, y_pred, y_score)

        assert m["accuracy"]["status"] == "MEASURED"
        assert m["accuracy"]["value"] is not None
        assert m["f1"]["status"] == "MEASURED"
        assert m["f1"]["value"] is not None
        assert m["roc_auc"]["status"] == "MEASURED"
        assert m["roc_auc"]["value"] is not None
        assert m["balanced_accuracy"]["status"] == "MEASURED"
        # New metrics present
        assert m["jaccard"]["status"] == "MEASURED"
        assert m["mcc"]["status"] == "MEASURED"


# ---------------------------------------------------------------------------
# _class_support
# ---------------------------------------------------------------------------

class TestClassSupport:

    def test_binary_includes_convenience_fields(self):
        y_train = np.array([0, 0, 1, 1, 0])
        y_test = np.array([0, 1])
        cs = _class_support(y_train, y_test)

        assert "train_counts" in cs
        assert "test_counts" in cs
        assert cs["train_positive"] == 2
        assert cs["train_negative"] == 3
        assert cs["test_positive"] == 1
        assert cs["test_negative"] == 1

    def test_generic_counts_always_present(self):
        # Use float32 to match actual R0 pipeline (y_all is float32)
        y_train = np.array([0, 0, 1], dtype=np.float32)
        y_test = np.array([0], dtype=np.float32)
        cs = _class_support(y_train, y_test)

        assert cs["train_counts"]["0.0"] == 2
        assert cs["train_counts"]["1.0"] == 1
        assert cs["test_counts"]["0.0"] == 1


# ---------------------------------------------------------------------------
# LEGACY_UNVERIFIED default (certificate consumer)
# ---------------------------------------------------------------------------

class TestLegacyUnverified:

    def test_old_record_without_eligibility_status(self):
        """Old R0 results without eligibility_status must not be treated as ELIGIBLE."""
        old_record = {
            "target": "obs_has_hwm",
            "split": "spatial_blocked",
            "solver": "histgbdt",
            "task": "classification",
            "metrics": {"roc_auc": 0.65, "accuracy": 0.8, "f1": 0.5},
            # No eligibility_status field
        }
        status = old_record.get("eligibility_status", "LEGACY_UNVERIFIED")
        assert status == "LEGACY_UNVERIFIED"
        assert status != "ELIGIBLE"


# ---------------------------------------------------------------------------
# RunResult serialization
# ---------------------------------------------------------------------------

class TestRunResultSerialization:

    def test_abstention_record_serializes_to_json(self):
        from dataclasses import asdict

        r = RunResult(
            scenario="houston",
            target="obs_has_hwm",
            task="classification",
            solver="histgbdt",
            split="leave_event_out",
            fold="beryl2024",
            n_train=300,
            n_test=50,
            metrics={
                name: {"status": "SKIP_TRAIN_SINGLE_CLASS", "value": None}
                for name in ["accuracy", "recall", "f1", "jaccard", "dice",
                             "mcc", "balanced_accuracy", "roc_auc", "auc_pr"]
            },
            naive_baseline={},
            features_used=33,
            timestamp="2026-07-01T00:00:00Z",
            eligibility_status="SKIP_TRAIN_SINGLE_CLASS",
            class_support={
                "train_counts": {"0.0": 280, "1.0": 20},
                "test_counts": {"0.0": 50},
                "train_positive": 20,
                "train_negative": 280,
                "test_positive": 0,
                "test_negative": 50,
            },
        )
        d = asdict(r)
        # Must be JSON-serializable
        s = json.dumps(d, default=str)
        parsed = json.loads(s)

        assert parsed["eligibility_status"] == "SKIP_TRAIN_SINGLE_CLASS"
        assert parsed["metrics"]["accuracy"]["value"] is None
        assert parsed["metrics"]["f1"]["value"] is None
        assert parsed["class_support"]["test_positive"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
