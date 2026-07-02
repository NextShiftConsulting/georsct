#!/usr/bin/env python3
"""
aggregate.py  (georsct.evaluation)

Status-aware aggregation of per-fold score_fold() outputs.

The per-metric statuses make two different questions answerable, and folding
them together is exactly the dilution the gate was built to prevent:

  detection    : mean of each metric over folds where THAT metric is MEASURED.
                 (New Orleans F1 is a mean over its 11 measured folds, not 46.)
  false_alarm  : the MEASURED_FALSE_ALARM_ONLY folds, reported as their own
                 channel and NEVER folded into detection.
  skipped      : per-status counts of folds a metric could not measure.

Folds whose training was skipped upstream (SKIP_TRAIN_SINGLE_CLASS) are counted,
never averaged. Status/vocabulary come from metric_eligibility -- this module
adds no new decision logic and defines no new status codes.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

from .metric_eligibility import MetricStatus, TrainingStatus

__all__ = ["aggregate_by_status"]

_MEASURED = MetricStatus.MEASURED.value
_FALSE_ALARM = MetricStatus.MEASURED_FALSE_ALARM_ONLY.value
_SKIP_TRAIN = TrainingStatus.SKIP_TRAIN_SINGLE_CLASS.value

# accuracy is MEASURED on every trained fold, including non-detection folds
# (all-negative truth -> accuracy=1.0). Its cross-fold mean is therefore not a
# comparison surface; metric primacy is a certificate-level policy, not the
# aggregator's decision. Flagged, still reported, never silently comparable.
_NOT_COMPARABLE = ("accuracy",)


def _normalize(fold_outputs):
    """Accept score_fold tuples (training_status, metrics) or bare metric dicts."""
    norm = []
    for item in fold_outputs:
        if (isinstance(item, tuple) and len(item) == 2
                and isinstance(item[1], dict)):
            train_status, metrics = item
        else:
            train_status, metrics = None, item
        norm.append((getattr(train_status, "value", train_status), metrics))
    return norm


def aggregate_by_status(fold_outputs, ndigits=4):
    """Aggregate a list of score_fold() outputs into per-metric channels.

    Returns:
        {
          "n_folds": int, "n_trained": int, "training_skipped": int,
          "metrics": {
            metric: {
              "detection_mean": float|None,   # mean over MEASURED folds only
              "detection_std":  float|None,   # sample std (ddof=1) if n>1
              "n_measured":     int,
              "false_alarm_n":  int,          # MEASURED_FALSE_ALARM_ONLY folds
              "skipped":        {status: count},
              # accuracy also carries: "comparable": False, "note": ...
            }, ...
          }
        }
    """
    folds = _normalize(fold_outputs)
    n_total = len(folds)
    trained = [(ts, m) for ts, m in folds if ts != _SKIP_TRAIN]

    names = []
    for _, m in trained:
        for k in m:
            if k not in names:
                names.append(k)

    metrics = {}
    for name in names:
        det, fa, skips = [], [], Counter()
        for _, m in trained:
            cell = m.get(name)
            if not cell:
                continue
            status, value = cell.get("status"), cell.get("value")
            if status == _MEASURED and value is not None:
                det.append(float(value))
            elif status == _FALSE_ALARM:
                fa.append(0.0 if value is None else float(value))
            elif status not in (_MEASURED, _FALSE_ALARM):
                skips[status] += 1
        entry = {
            "detection_mean": round(float(np.mean(det)), ndigits) if det else None,
            "detection_std": (round(float(np.std(det, ddof=1)), ndigits)
                              if len(det) > 1 else None),
            "n_measured": len(det),
            "false_alarm_n": len(fa),
            "skipped": dict(skips),
        }
        if name in _NOT_COMPARABLE:
            entry["comparable"] = False
            entry["note"] = ("MEASURED on every trained fold incl. non-detection "
                             "folds; not a comparison metric (primacy is policy)")
        metrics[name] = entry

    return {
        "n_folds": n_total,
        "n_trained": len(trained),
        "training_skipped": n_total - len(trained),
        "metrics": metrics,
    }
