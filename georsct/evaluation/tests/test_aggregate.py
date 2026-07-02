"""Tests for status-aware aggregation (georsct.evaluation.aggregate)."""
import numpy as np

from georsct.evaluation.metric_eligibility import (
    score_fold, training_eligibility, TrainingStatus,
)
from georsct.evaluation.aggregate import aggregate_by_status


def _healthy(rng):
    n = 40
    y_true = (rng.random(n) < 0.35).astype(int)
    y_true[:2] = [0, 1]                      # guarantee both classes
    y_score = np.clip(0.4 * y_true + rng.normal(0.3, 0.2, n), 0, 1)
    y_pred = (y_score > 0.5).astype(int)
    return score_fold(y_pred, y_true, y_score,
                      fold_training_status=training_eligibility([0, 1]))


def _empty_union():
    y = np.zeros(6, int)
    return score_fold(y, y, y.astype(float),
                      fold_training_status=training_eligibility([0, 1]))


def _false_alarm():
    y_true = np.zeros(6, int)
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    return score_fold(y_pred, y_true, y_pred.astype(float),
                      fold_training_status=training_eligibility([0, 1]))


def _train_skipped():
    y_true = np.array([0, 1, 0, 1]); y_pred = np.array([0, 1, 1, 1])
    return score_fold(y_pred, y_true,
                      fold_training_status=training_eligibility(np.zeros(10, int)))


def test_detection_mean_uses_measured_folds_only():
    rng = np.random.default_rng(0)
    folds = [_healthy(rng), _healthy(rng), _healthy(rng),
             _empty_union(), _empty_union(), _false_alarm()]
    agg = aggregate_by_status(folds)
    f1 = agg["metrics"]["f1"]
    assert f1["n_measured"] == 3          # only the healthy folds
    assert f1["false_alarm_n"] == 1
    assert f1["skipped"].get("SKIP_EMPTY_UNION") == 2
    assert f1["detection_mean"] is not None


def test_false_alarm_never_enters_detection():
    rng = np.random.default_rng(1)
    # one healthy fold (f1>0) + one false-alarm fold (f1==0). If the false alarm
    # leaked into detection, the mean would be pulled toward 0.
    folds = [_healthy(rng), _false_alarm()]
    agg = aggregate_by_status(folds)
    f1 = agg["metrics"]["f1"]
    assert f1["n_measured"] == 1
    assert f1["false_alarm_n"] == 1
    # detection_mean equals the single healthy fold's f1, not an average with 0
    healthy_val = _healthy(np.random.default_rng(1))[1]["f1"]["value"]
    assert abs(f1["detection_mean"] - healthy_val) < 1e-9


def test_training_skipped_counted_not_averaged():
    rng = np.random.default_rng(2)
    folds = [_healthy(rng), _train_skipped(), _train_skipped()]
    agg = aggregate_by_status(folds)
    assert agg["n_folds"] == 3
    assert agg["n_trained"] == 1
    assert agg["training_skipped"] == 2
    assert agg["metrics"]["f1"]["n_measured"] == 1


def test_accuracy_flagged_not_comparable():
    rng = np.random.default_rng(3)
    agg = aggregate_by_status([_healthy(rng), _empty_union()])
    acc = agg["metrics"]["accuracy"]
    assert acc["comparable"] is False
    # accuracy is MEASURED on both folds (incl. the all-negative one)
    assert acc["n_measured"] == 2


def test_ranking_family_measured_only_where_both_classes_present():
    rng = np.random.default_rng(4)
    folds = [_healthy(rng), _empty_union(), _false_alarm()]
    agg = aggregate_by_status(folds)
    roc = agg["metrics"]["roc_auc"]
    assert roc["n_measured"] == 1         # only the healthy fold has both classes
    assert roc["skipped"].get("SKIP_TEST_SINGLE_CLASS") == 2


def test_accepts_bare_metric_dicts_without_training_status():
    rng = np.random.default_rng(5)
    _, metrics_only = _healthy(rng)
    agg = aggregate_by_status([metrics_only])   # bare dict, no tuple
    assert agg["n_trained"] == 1
    assert agg["training_skipped"] == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
