"""Regression guards for the metric-eligibility gate (georsct.evaluation)."""
import numpy as np
import pytest

from georsct.evaluation.metric_eligibility import (
    all_metrics, classify_fold, score_fold, training_eligibility,
    TrainingStatus, MetricStatus,
)

RNG = np.random.default_rng(0)


def _random_fold():
    n = int(RNG.integers(20, 200))
    p = float(RNG.uniform(0.0, 0.4))
    y_true = (RNG.random(n) < p).astype(int)
    y_score = np.clip(0.15 * y_true + RNG.normal(0.3, 0.25, n), 0, 1)
    y_pred = (y_score > RNG.uniform(0.3, 0.7)).astype(int)
    return y_true, y_pred, y_score


def test_jaccard_is_monotone_transform_of_f1():
    for _ in range(2000):
        y_true, y_pred, y_score = _random_fold()
        m = all_metrics(y_true, y_pred, y_score)
        if m["f1"] == m["f1"] and m["jaccard"] == m["jaccard"]:
            assert abs(m["jaccard"] - m["f1"] / (2 - m["f1"])) < 1e-9


def test_f1_and_jaccard_share_eligibility():
    for _ in range(2000):
        y_true, y_pred, y_score = _random_fold()
        elig = classify_fold(y_pred, y_true, y_score)
        assert elig["f1"] == elig["jaccard"] == elig["dice"]


def test_empty_union_skips_overlap_but_accuracy_is_measurable():
    # all-negative truth, model quiet: accuracy computable (=1.0, MEASURED) but
    # overlap must SKIP so f1 is never reported as a misleading 0.0 or 1.0.
    y_true = np.zeros(10, int)
    y_pred = np.zeros(10, int)
    elig = classify_fold(y_pred, y_true, y_true.astype(float))
    assert elig["f1"] is MetricStatus.SKIP_EMPTY_UNION
    assert elig["accuracy"] is MetricStatus.MEASURED
    _, out = score_fold(y_pred, y_true, y_true.astype(float))
    assert out["f1"]["value"] is None            # not reported
    assert out["accuracy"]["value"] == 1.0        # computed; primacy is policy


def test_false_alarm_is_flagged():
    y_true = np.zeros(5, int)
    y_pred = np.array([0, 1, 0, 1, 0])
    elig = classify_fold(y_pred, y_true, y_pred.astype(float))
    assert elig["f1"] is MetricStatus.MEASURED_FALSE_ALARM_ONLY


def test_mcc_requires_prediction_variance():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.zeros(5, int)
    elig = classify_fold(y_pred, y_true, y_true.astype(float))
    assert elig["mcc"] is MetricStatus.SKIP_NO_PREDICTION_VARIANCE
    assert elig["balanced_accuracy"] is MetricStatus.MEASURED


def test_ranking_family_stricter_than_overlap():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_score = np.array([0.1, 0.6, 0.2, 0.3])
    elig = classify_fold(y_pred, y_true, y_score)
    assert elig["roc_auc"] is MetricStatus.SKIP_TEST_SINGLE_CLASS
    assert elig["f1"] in (MetricStatus.MEASURED, MetricStatus.MEASURED_FALSE_ALARM_ONLY)


def test_training_gate_is_single_source_and_passes_through():
    # single-class training -> upstream verdict; score_fold carries it through
    ts = training_eligibility(np.zeros(20, int))
    assert ts is TrainingStatus.SKIP_TRAIN_SINGLE_CLASS
    y_true = np.array([0, 1, 0, 1]); y_pred = np.array([0, 1, 1, 1])
    carried, _ = score_fold(y_pred, y_true, fold_training_status=ts)
    assert carried == TrainingStatus.SKIP_TRAIN_SINGLE_CLASS.value


def test_score_fold_does_not_recompute_training():
    # score_fold has no y_train argument -> cannot re-decide training eligibility
    import inspect
    params = inspect.signature(score_fold).parameters
    assert "y_train" not in params


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
