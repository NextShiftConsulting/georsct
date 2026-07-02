#!/usr/bin/env python3
"""
metric_eligibility.py  (GeoRSCT / georsct.evaluation)

Metric-specific eligibility gate. Answers ONE question: which metrics can this
fold report HONESTLY? Metric *selection* policy (which metric is primary) is a
certificate-level decision and deliberately does NOT live here.

Two eligibility FAMILIES -- the real "changes what is usable" axis (NOT F1 vs
Jaccard, which are monotone: J = F1/(2-F1)):

  OVERLAP family    (f1, jaccard, dice, recall): eligible iff union TP+FP+FN>0.
  RANKING/TN family (roc_auc, auc_pr, mcc, balanced_accuracy): eligible iff BOTH
                    classes present in test truth; MCC also needs prediction
                    variance. Stricter -> drops more folds.

Gate composition (single decision authority per axis, cf. ADR-062):
  BEFORE fit :  training_eligibility(y_train)  -> can the solver learn at all?
                Call this ONCE upstream (train_r0_baseline). Its verdict is
                carried through, never recomputed downstream.
  AFTER fit  :  classify_fold(y_pred, y_true, ...) -> which metrics are reportable?
                Needs y_pred to separate false-alarm (Case 3) from empty-union
                (Case 4). Does NOT re-decide training eligibility.

Reason codes are a typed Enum (ADR-025 / ADR-034), not free-text.
Dependency surface: numpy + scipy. (pandas only under __main__.)
"""
from __future__ import annotations
import numpy as np
from enum import Enum
from scipy.stats import rankdata

__all__ = [
    "TrainingStatus", "MetricStatus",
    "training_eligibility", "classify_fold", "score_fold",
    "all_metrics", "confusion", "roc_auc", "auc_pr",
]


class TrainingStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    SKIP_TRAIN_SINGLE_CLASS = "SKIP_TRAIN_SINGLE_CLASS"


class MetricStatus(str, Enum):
    MEASURED = "MEASURED"                            # computable and honest
    SKIP_TEST_SINGLE_CLASS = "SKIP_TEST_SINGLE_CLASS"  # ranking/TN family
    SKIP_NO_PREDICTION_VARIANCE = "SKIP_NO_PREDICTION_VARIANCE"  # MCC extra requirement
    SKIP_EMPTY_UNION = "SKIP_EMPTY_UNION"            # overlap family, no truth & no pred
    MEASURED_FALSE_ALARM_ONLY = "MEASURED_FALSE_ALARM_ONLY"  # truth all-neg, model fires


_MEASURED = (MetricStatus.MEASURED, MetricStatus.MEASURED_FALSE_ALARM_ONLY)


# --------------------------------------------------------------------------- #
# Metric values (NaN on degeneracy).                                          #
# --------------------------------------------------------------------------- #
def confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(y_score)
    return float((r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_pr(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-np.asarray(y_score, float))
    yt = y_true[order]
    tp = np.cumsum(yt); fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return float(trap(precision, recall))


def all_metrics(y_true, y_pred, y_score=None):
    tp, fp, tn, fn = confusion(y_true, y_pred)
    f1_den = 2 * tp + fp + fn
    jac_den = tp + fp + fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(
        accuracy=(tp + tn) / max(tp + fp + tn + fn, 1),
        recall=rec,
        f1=(2 * tp / f1_den) if f1_den else np.nan,
        jaccard=(tp / jac_den) if jac_den else np.nan,   # == f1/(2-f1); IoU
        dice=(2 * tp / f1_den) if f1_den else np.nan,     # == f1
        mcc=((tp * tn - fp * fn) / mcc_den) if mcc_den else np.nan,
        balanced_accuracy=float(np.nanmean([rec, tnr])),
        roc_auc=roc_auc(y_true, y_score) if y_score is not None else np.nan,
        auc_pr=auc_pr(y_true, y_score) if y_score is not None else np.nan,
    )


# --------------------------------------------------------------------------- #
# BEFORE-fit gate: single source of truth. Decided once, never recomputed.     #
# --------------------------------------------------------------------------- #
def training_eligibility(y_train) -> TrainingStatus:
    """Can the solver learn a classifier on this fold? Call ONCE upstream."""
    y_train = np.asarray(y_train).astype(int)
    return (TrainingStatus.ELIGIBLE if len(np.unique(y_train)) == 2
            else TrainingStatus.SKIP_TRAIN_SINGLE_CLASS)


# --------------------------------------------------------------------------- #
# AFTER-fit gate: which metrics are reportable? (Does not touch training.)     #
# --------------------------------------------------------------------------- #
def classify_fold(y_pred, y_true, y_score=None) -> dict:
    """Per-metric eligibility for a fitted fold. Returns {metric: MetricStatus}."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp, fp, tn, fn = confusion(y_true, y_pred)
    truth_both = (tp + fn > 0) and (tn + fp > 0)
    pred_both = (tp + fp > 0) and (tn + fn > 0)
    union = tp + fp + fn
    truth_has_pos = (tp + fn) > 0

    def overlap():
        if union == 0:
            return MetricStatus.SKIP_EMPTY_UNION
        if not truth_has_pos:
            return MetricStatus.MEASURED_FALSE_ALARM_ONLY
        return MetricStatus.MEASURED

    ranking = MetricStatus.MEASURED if truth_both else MetricStatus.SKIP_TEST_SINGLE_CLASS
    mcc = (MetricStatus.MEASURED if (truth_both and pred_both)
           else (MetricStatus.SKIP_TEST_SINGLE_CLASS if not truth_both
                 else MetricStatus.SKIP_NO_PREDICTION_VARIANCE))
    has_score = y_score is not None

    return {
        # accuracy is always computable; whether it is PRIMARY is a certificate
        # policy decision, not an eligibility gate -> MEASURED here.
        "accuracy": MetricStatus.MEASURED,
        "f1": overlap(),
        "jaccard": overlap(),
        "dice": overlap(),
        "recall": overlap(),
        "mcc": mcc,
        "balanced_accuracy": ranking,
        "roc_auc": ranking if has_score else MetricStatus.SKIP_TEST_SINGLE_CLASS,
        "auc_pr": (MetricStatus.MEASURED if (truth_has_pos and has_score)
                   else MetricStatus.SKIP_TEST_SINGLE_CLASS),
    }


def score_fold(y_pred, y_true, y_score=None, fold_training_status=None, ndigits=4):
    """Report eligible metrics with per-metric status.

    `fold_training_status` is the upstream training_eligibility() verdict, carried
    through unchanged (single decision authority). It is NOT recomputed here.
    Returns (fold_training_status_value_or_None, {metric: {status, value}}).
    """
    elig = classify_fold(y_pred, y_true, y_score)
    vals = all_metrics(y_true, y_pred, y_score)
    out = {}
    for name, status in elig.items():
        v = vals[name]
        measured = status in _MEASURED and v == v      # v==v filters NaN
        out[name] = dict(status=status.value,
                         value=round(float(v), ndigits) if measured else None)
    ts = getattr(fold_training_status, "value", fold_training_status)
    return ts, out


# --------------------------------------------------------------------------- #
# Demo (pandas only here).                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 140)

    def _fold(y_true, y_pred, y_score):
        return dict(y_pred=np.array(y_pred), y_true=np.array(y_true),
                    y_score=np.array(y_score, float),
                    fold_training_status=training_eligibility([0, 1]))

    folds = {
        "healthy (both classes, good model)": _fold(
            [0,0,0,1,1,0,1,0,0,1], [0,0,1,1,1,0,1,0,0,0],
            [.1,.2,.6,.8,.7,.2,.9,.3,.1,.4]),
        "all-neg truth, model quiet (Case 4)": _fold(
            [0,0,0,0,0], [0,0,0,0,0], [.1,.2,.1,.3,.2]),
        "all-neg truth, model fires (Case 3)": _fold(
            [0,0,0,0,0], [0,1,0,1,0], [.1,.6,.2,.7,.3]),
        "positives present, model misses all": _fold(
            [0,0,1,1,0], [0,0,0,0,0], [.1,.2,.4,.3,.2]),
    }
    for name, f in folds.items():
        ts, out = score_fold(**f)
        print("=" * 92)
        print(f"{name}   [training={ts}]")
        row = {k: (v["value"] if v["value"] is not None else v["status"])
               for k, v in out.items()}
        print(pd.Series(row).to_string())
