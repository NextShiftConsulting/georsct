"""GeoRSCT evaluation: scoring (process/outcome/GeoCert) + metric eligibility.

`scoring` is imported lazily (it pulls georsct.contracts / provenance); the
metric-eligibility gate has no such dependency and is exported directly.
"""
from .metric_eligibility import (
    TrainingStatus, MetricStatus,
    training_eligibility, classify_fold, score_fold,
    all_metrics, confusion, roc_auc, auc_pr,
)
from .aggregate import aggregate_by_status

__all__ = [
    "TrainingStatus", "MetricStatus",
    "training_eligibility", "classify_fold", "score_fold",
    "all_metrics", "confusion", "roc_auc", "auc_pr",
    "aggregate_by_status",
]
