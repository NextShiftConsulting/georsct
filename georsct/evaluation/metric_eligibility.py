#!/usr/bin/env python3
"""
metric_eligibility.py  (GeoRSCT)

Corrected metric-specific eligibility gate.

Adopts the RIGHT part of the review (eligibility is per-metric; split training
status from metric status; never let empty-union score 1.0) and fixes the WRONG
part (that Jaccard changes usability relative to F1 -- it does not).

Two eligibility FAMILIES, which is the real "changes what is usable" axis:

  OVERLAP family   (F1, Jaccard/IoU, Dice, recall) :
      eligible iff union TP+FP+FN > 0.  Identical across the family.
      Jaccard == monotone(F1); swapping them never changes a ranking.

  RANKING/TN family (ROC-AUC, AUC-PR, MCC, balanced acc, specificity) :
      eligible iff BOTH classes present in test truth (MCC also needs
      prediction variance). STRICTER -> drops more folds.

So the metric that would actually change your picture is NOT Jaccard-vs-F1.
It is overlap-family vs ranking-family. The ranking family asks a different
(often better for rare events, e.g. AUC-PR) question but costs you folds in
New Orleans / Riverside -- the opposite of rescuing them.

Reason codes are a typed Enum (per your ADR-025 / ADR-034 discipline), not
free-text strings.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from enum import Enum
from scipy.stats import rankdata


class TrainingStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    SKIP_TRAIN_SINGLE_CLASS = "SKIP_TRAIN_SINGLE_CLASS"


class MetricStatus(str, Enum):
    MEASURED = "MEASURED"
    NOT_PRIMARY = "NOT_PRIMARY"                       # accuracy: defined but TN-inflated
    SKIP_TEST_SINGLE_CLASS = "SKIP_TEST_SINGLE_CLASS"  # ranking/TN family
    SKIP_NO_PREDICTION_VARIANCE = "SKIP_NO_PREDICTION_VARIANCE"  # MCC extra requirement
    SKIP_EMPTY_UNION = "SKIP_EMPTY_UNION"             # overlap family, Case 4
    MEASURED_FALSE_ALARM_ONLY = "MEASURED_FALSE_ALARM_ONLY"  # Case 3: truth all-neg, pred+


# --------------------------------------------------------------------------- #
# Metric values (NaN on degeneracy).                                          #
# --------------------------------------------------------------------------- #
def _confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(y_score)
    return (r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc_pr(y_true, y_score):
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-np.asarray(y_score))
    yt = y_true[order]
    tp = np.cumsum(yt); fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(precision, recall))


def all_metrics(y_true, y_pred, y_score=None):
    tp, fp, tn, fn = _confusion(y_true, y_pred)
    f1_den = 2 * tp + fp + fn
    jac_den = tp + fp + fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(
        accuracy=(tp + tn) / max(tp + fp + tn + fn, 1),
        recall=rec,
        f1=(2 * tp / f1_den) if f1_den else np.nan,
        jaccard=(tp / jac_den) if jac_den else np.nan,
        dice=(2 * tp / f1_den) if f1_den else np.nan,          # == F1
        mcc=((tp * tn - fp * fn) / mcc_den) if mcc_den else np.nan,
        balanced_accuracy=np.nanmean([rec, tnr]),
        roc_auc=roc_auc(y_true, y_score) if y_score is not None else np.nan,
        auc_pr=auc_pr(y_true, y_score) if y_score is not None else np.nan,
    )


# --------------------------------------------------------------------------- #
# The gate: fold_training_status + per-metric eligibility with typed codes.    #
# --------------------------------------------------------------------------- #
def classify_fold(y_train, y_pred, y_true, y_score=None):
    y_train = np.asarray(y_train).astype(int)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    train_status = (TrainingStatus.ELIGIBLE
                    if len(np.unique(y_train)) == 2
                    else TrainingStatus.SKIP_TRAIN_SINGLE_CLASS)

    tp, fp, tn, fn = _confusion(y_true, y_pred)
    truth_both = (tp + fn > 0) and (tn + fp > 0)
    pred_both = (tp + fp > 0) and (tn + fn > 0)
    union = tp + fp + fn
    truth_has_pos = (tp + fn) > 0

    def overlap_status():
        if union == 0:
            return MetricStatus.SKIP_EMPTY_UNION            # Case 4
        if not truth_has_pos:
            return MetricStatus.MEASURED_FALSE_ALARM_ONLY   # Case 3: score is 0, but flag it
        return MetricStatus.MEASURED

    ranking_status = (MetricStatus.MEASURED if truth_both
                      else MetricStatus.SKIP_TEST_SINGLE_CLASS)

    mcc_status = (MetricStatus.MEASURED if (truth_both and pred_both)
                  else (MetricStatus.SKIP_TEST_SINGLE_CLASS if not truth_both
                        else MetricStatus.SKIP_NO_PREDICTION_VARIANCE))

    elig = {
        "accuracy": MetricStatus.NOT_PRIMARY,
        "f1": overlap_status(),
        "jaccard": overlap_status(),      # identical to f1 by construction
        "dice": overlap_status(),
        "recall": overlap_status(),
        "mcc": mcc_status,
        "balanced_accuracy": ranking_status,
        "roc_auc": ranking_status if y_score is not None else MetricStatus.SKIP_TEST_SINGLE_CLASS,
        "auc_pr": (MetricStatus.MEASURED if truth_has_pos and y_score is not None
                   else MetricStatus.SKIP_TEST_SINGLE_CLASS),
    }
    return train_status, elig


def score_fold(y_train, y_pred, y_true, y_score=None):
    """Return only the metrics this fold is eligible to report, with statuses."""
    train_status, elig = classify_fold(y_train, y_pred, y_true, y_score)
    vals = all_metrics(y_true, y_pred, y_score)
    out = {}
    for name, status in elig.items():
        measured = status in (MetricStatus.MEASURED,
                              MetricStatus.MEASURED_FALSE_ALARM_ONLY,
                              MetricStatus.NOT_PRIMARY)
        out[name] = dict(status=status.value,
                         value=(round(float(vals[name]), 4)
                                if measured and vals[name] == vals[name] else None))
    return train_status.value, out


# --------------------------------------------------------------------------- #
# Demonstration + equivalence checks.                                         #
# --------------------------------------------------------------------------- #
def _fold(y_true, y_pred, y_score, y_train=(0, 1)):
    return dict(y_train=np.array(y_train), y_true=np.array(y_true),
                y_pred=np.array(y_pred), y_score=np.array(y_score, float))


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    rng = np.random.default_rng(0)

    # ---- equivalence: F1 and Jaccard get IDENTICAL statuses over random folds
    same_status = True
    mono_ok = True
    for _ in range(3000):
        n = rng.integers(20, 200)
        p = rng.uniform(0.0, 0.4)
        yt = (rng.random(n) < p).astype(int)
        ys = np.clip(0.15 * yt + rng.normal(0.3, 0.25, n), 0, 1)
        yp = (ys > rng.uniform(0.3, 0.7)).astype(int)
        _, e = classify_fold([0, 1], yp, yt, ys)
        if e["f1"] != e["jaccard"]:
            same_status = False
        m = all_metrics(yt, yp, ys)
        if m["f1"] == m["f1"] and m["jaccard"] == m["jaccard"]:
            if abs(m["jaccard"] - m["f1"] / (2 - m["f1"])) > 1e-9:
                mono_ok = False
    print("EQUIVALENCE CHECK (3000 random folds)")
    print(f"  F1 and Jaccard identical eligibility status : {same_status}")
    print(f"  Jaccard == F1/(2-F1) wherever both defined   : {mono_ok}")
    print("  => within the overlap family, metric choice cannot change a ranking.\n")

    # ---- the four canonical folds, run through the gate
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
        row = {k: (f"{v['value']}" if v["value"] is not None else v["status"])
               for k, v in out.items()}
        print(pd.Series(row).to_string())
