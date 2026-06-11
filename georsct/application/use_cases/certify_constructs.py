"""Use case: Compute five-construct divergence for a geography.

Certifies the same geography under each of five flood constructs,
then computes pairwise certificate distance to produce a 5x5
divergence matrix.

MEASUREMENT layer (P4).  No decisions emitted -- only scores.
Orchestration only: imports from domain + ports, no I/O, no pandas.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import sparse

from georsct.domain.construct_certificate import (
    CONSTRUCT_TARGET_COLUMNS,
    ConstructCertificate,
    ConstructLabel,
    compute_kappa_spatial,
)
from georsct.domain.construct_divergence_matrix import (
    CONSTRUCT_ORDER,
    DivergenceMatrix,
    build_divergence_matrix,
)
from georsct.domain.kappa_reconstruct import compute_kappa_reconstruct
from georsct.ports.construct_data_source import ConstructDataSource
from georsct.ports.model_fitter import ModelFitter

log = logging.getLogger(__name__)


def certify_single_construct(
    construct: ConstructLabel,
    features: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
    region_ids: np.ndarray,
    region_order: tuple[str, ...],
    coords2d: np.ndarray,
    W_geo: sparse.csr_matrix,
    model_fitter: ModelFitter,
    task_type: str = "regression",
    n_baseline_trials: int = 20,
    n_mantel_perms: int = 99,
) -> ConstructCertificate:
    """Certify one geography under one construct.

    Delegates fitting to the ModelFitter port, kappa computation
    to domain functions.

    Args:
        construct: Which flood construct.
        features: (n_obs, n_features) feature matrix.
        target: (n_obs,) target values for this construct.
        fold_ids: (n_obs,) integer fold assignment.
        region_ids: (n_obs,) string region labels.
        region_order: Canonical ordering for region embeddings.
        coords2d: (n_regions, 2) true spatial coordinates.
        W_geo: (n_regions, n_regions) row-normalized adjacency.
        model_fitter: Concrete ModelFitter implementation.
        task_type: "regression" or "binary_classification".
        n_baseline_trials: Null-graph trials for kappa_reconstruct.
        n_mantel_perms: Mantel permutations for corroboration.

    Returns:
        ConstructCertificate with forward_score, kappa_spatial,
        kappa_reconstruct, and provenance.
    """
    target_col = CONSTRUCT_TARGET_COLUMNS.get(construct, construct.value)

    n_finite = int(np.isfinite(target).sum())
    if n_finite < 30:
        return ConstructCertificate.missing(
            construct,
            f"insufficient finite targets (n={n_finite}, need 30)",
        )

    # 1. Fit model via port -> forward score
    fp = model_fitter.fit_predict(features, target, fold_ids, task_type)

    # 2. Compute residuals -> kappa_spatial via domain function
    residuals = target - fp.predictions
    valid_mask = np.isfinite(residuals)

    # Aggregate residuals to region level
    region_resid = _aggregate_to_regions(
        residuals, region_ids, region_order, valid_mask,
    )
    kappa_s, morans_i = compute_kappa_spatial(region_resid, W_geo)

    # 3. Compute region embeddings -> kappa_reconstruct via domain function
    embed = model_fitter.aggregate_embeddings(features, region_ids, region_order)
    sr = compute_kappa_reconstruct(
        embed.embeddings,
        coords2d,
        n_baseline_trials=n_baseline_trials,
        n_mantel_perms=n_mantel_perms,
    )

    return ConstructCertificate.from_scores(
        construct=construct,
        target_column=target_col,
        forward_score=float(fp.forward_score),
        kappa_spatial=float(kappa_s),
        kappa_reconstruct=float(sr.kappa_reconstruct),
        morans_i=float(morans_i),
        n_regions=len(region_order),
        n_observations=len(target),
        n_finite_targets=n_finite,
    )


def compute_five_construct_divergence(
    scenario_id: str,
    data_source: ConstructDataSource,
    model_fitter: ModelFitter,
    features: np.ndarray,
    fold_ids: np.ndarray,
    region_ids: np.ndarray,
    region_order: tuple[str, ...],
    coords2d: np.ndarray,
    W_geo: sparse.csr_matrix,
    event_id: Optional[str] = None,
) -> DivergenceMatrix:
    """Certify all five constructs and compute divergence matrix.

    Handles missing constructs gracefully per ADR-020 D8.

    Args:
        scenario_id: Scenario key (e.g., "houston").
        data_source: Concrete ConstructDataSource implementation.
        model_fitter: Concrete ModelFitter implementation.
        features: (n_obs, n_features) shared feature matrix.
        fold_ids: (n_obs,) integer fold assignment.
        region_ids: (n_obs,) string region labels.
        region_order: Canonical region ordering.
        coords2d: (n_regions, 2) true coordinates.
        W_geo: (n_regions, n_regions) row-normalized adjacency.
        event_id: Optional event filter for event-specific constructs.

    Returns:
        DivergenceMatrix with 5x5 pairwise distances.
    """
    certificates: list[ConstructCertificate] = []

    for construct in CONSTRUCT_ORDER:
        log.info(
            "Certifying construct: %s (%s)",
            construct.name, construct.value,
        )

        cd = data_source.load_construct_target(
            construct, scenario_id, event_id,
        )

        if not cd.available:
            log.warning(
                "Construct %s unavailable: %s", construct.name, cd.reason,
            )
            certificates.append(
                ConstructCertificate.missing(construct, cd.reason)
            )
            continue

        cert = certify_single_construct(
            construct=construct,
            features=features,
            target=cd.target_values,
            fold_ids=fold_ids,
            region_ids=region_ids,
            region_order=region_order,
            coords2d=coords2d,
            W_geo=W_geo,
            model_fitter=model_fitter,
        )
        certificates.append(cert)

        log.info(
            "  %s: forward=%.3f  kappa_spatial=%.3f  kappa_reconstruct=%.3f",
            construct.name,
            cert.forward_score,
            cert.kappa_spatial,
            cert.kappa_reconstruct,
        )

    geo_label = f"{scenario_id}/{event_id}" if event_id else scenario_id
    return build_divergence_matrix(certificates, geography_id=geo_label)


# ---------------------------------------------------------------------------
# Internal helper -- region-level aggregation (no pandas)
# ---------------------------------------------------------------------------

def _aggregate_to_regions(
    values: np.ndarray,
    region_ids: np.ndarray,
    region_order: tuple[str, ...],
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Mean of values per region, aligned to region_order.

    Returns (n_regions,) array with NaN for regions with no valid data.
    """
    n = len(region_order)
    sums = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    idx_map = {r: i for i, r in enumerate(region_order)}

    for k in range(len(values)):
        if not valid_mask[k]:
            continue
        r = str(region_ids[k])
        i = idx_map.get(r)
        if i is not None:
            sums[i] += values[k]
            counts[i] += 1

    result = np.full(n, np.nan, dtype=float)
    nonzero = counts > 0
    result[nonzero] = sums[nonzero] / counts[nonzero]
    return result
