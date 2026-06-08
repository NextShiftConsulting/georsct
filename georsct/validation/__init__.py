"""Deployment-aligned spatial validation for RSCT.

Implements task descriptors, deployment-domain definitions, TWCV/IWCV
weighting, and alignment certificates following the framework of
Brenning & Suesse (2026).
"""

from georsct.validation.task_descriptors import (
    SENTINEL_BIN,
    TaskDescriptorConfig,
    add_nearest_training_distance,
    apply_bins,
    build_task_descriptors,
    build_target_descriptors,
    fit_quantile_edges,
    haversine_km,
)
from georsct.validation.deployment_alignment import (
    AlignmentGates,
    alignment_gaps,
    alignment_summary,
    compute_iwcv_weights,
    effective_sample_size,
    js_divergence,
    js_null_threshold,
    marginal_ratio_weights,
    normalize_weights,
    rake_weights,
)
from georsct.validation.deployment_domains import (
    EligibilityRule,
    EmptyDeploymentDomain,
    MissingHazardLayer,
    RegimeDomain,
)
from georsct.validation.deployment_target import (
    build_deployment_target_descriptors,
    nearest_distance_to_reference,
)

__all__ = [
    # task_descriptors
    "SENTINEL_BIN",
    "TaskDescriptorConfig",
    "add_nearest_training_distance",
    "apply_bins",
    "build_task_descriptors",
    "build_target_descriptors",
    "fit_quantile_edges",
    "haversine_km",
    # deployment_alignment
    "AlignmentGates",
    "alignment_gaps",
    "alignment_summary",
    "compute_iwcv_weights",
    "effective_sample_size",
    "js_divergence",
    "js_null_threshold",
    "marginal_ratio_weights",
    "normalize_weights",
    "rake_weights",
    # deployment_domains
    "EligibilityRule",
    "EmptyDeploymentDomain",
    "MissingHazardLayer",
    "RegimeDomain",
    # deployment_target
    "build_deployment_target_descriptors",
    "nearest_distance_to_reference",
]
