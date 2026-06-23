"""
Gate 4 Grounding Certificate for GeoRSCT / FloodRSCT

Frozen July design:

Gate 4 = Gate 4A + Gate 4B

Gate 4A:
    Intrinsic non-collapse guard.
    Does NOT use permutation nulls.

Gate 4B:
    Physical-proxy checkability via permute-before-probe null.
    The probe (leave-one-block-out ridge) is refitted under each
    permutation so the null tests the entire pipeline (probe + metric),
    not just the metric.  This fixes the 11.5% false-positive rate
    found in the pre-fix version where the probe was fitted once and
    only the proxy was shuffled.

    Uses spatially restricted block-permutation nulls derived from the
    existing spatial CV partition.

Explicit exclusions:
    - No delta_solver_score inside Gate 4.
    - No R/S/N adjustment inside Gate 4.
    - No temporal-order test.
    - No residual-Moran physics channel.
    - No claim of physical correctness or world modeling.
    - No external y_pred accepted (leakage-proof by construction).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Literal, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

Direction = Literal["higher_is_better", "lower_is_better"]
Verdict = Literal["PASS", "WARN", "FAIL", "INSUFFICIENT"]


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Gate4Config:
    """Predeclared constants for Gate 4.

    These should be recorded in every emitted certificate.
    """

    alpha: float = 0.05
    n_perm: int = 999
    base_seed: int = 42

    # Spatial-null construction
    min_block_size: int = 3
    min_admissible_units: int = 10
    min_distinct_null: Optional[int] = None
    distinct_round_decimals: int = 12

    # Coverage requirements
    min_noncollapse_coverage: float = 0.80
    min_proxy_coverage: float = 0.80

    # Non-collapse guard
    min_effective_rank: float = 2.0
    warn_effective_rank_floor: float = 3.0
    min_total_variance: float = 1e-8
    min_pairwise_dispersion: float = 1e-8

    # Ridge probe regularization
    ridge_alpha: float = 1.0

    # Diagnostic policy: per-channel p-values uncorrected; PASS/WARN/FAIL
    # aggregation is handled by required-proxy logic, not by a hidden
    # multiple-comparison rule.
    p_value_correction: Literal["none"] = "none"


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class NonCollapseMetrics:
    effective_rank: float
    total_variance: float
    pairwise_dispersion: float
    n_units: int
    n_dims: int
    coverage: float


@dataclass(frozen=True)
class NullResult:
    observed: float
    p_value: float
    n_perm: int
    n_distinct_null: int
    coverage: float
    resolvable: bool
    direction: Direction
    reason: Optional[str] = None


@dataclass(frozen=True)
class ProxyCheckSpec:
    """One exogenous physical-proxy recovery check.

    The probe prediction is computed internally by a leave-one-block-out
    ridge regressor.  No external y_pred is accepted -- this is the
    leakage-proof design.

    Example names:
        precipitation
        hand_elevation
        water_adjacency
        impervious_surface
        soil_moisture
    """

    name: str
    y_true_proxy: np.ndarray
    direction: Direction = "higher_is_better"
    required: bool = True


@dataclass(frozen=True)
class Gate4Result:
    verdict: Verdict
    reason: str
    noncollapse: Dict
    checkability: Dict[str, Dict]
    alpha: float
    n_perm: int
    base_seed: int
    null_mode: str
    block_source: str
    required_proxies: Sequence[str]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _derive_seed(base_seed: int, name: str) -> int:
    """Deterministic per-channel seed.

    Python's built-in hash() is salted per process, so use a stable
    SHA-256 digest.  This makes parallel execution order-independent.
    """
    digest = hashlib.sha256(f"{base_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def empirical_p_value(
    observed: float,
    null_scores: np.ndarray,
    direction: Direction,
) -> float:
    """Add-one corrected empirical p-value.

    p = (1 + # null scores at least as extreme as observed) / (B + 1)
    """
    null_scores = np.asarray(null_scores, dtype=float)

    if direction == "higher_is_better":
        extreme = int(np.sum(null_scores >= observed))
    elif direction == "lower_is_better":
        extreme = int(np.sum(null_scores <= observed))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return float((1 + extreme) / (len(null_scores) + 1))


# ---------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------

def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation (scale-free, directly readable).

    Default metric for Gate 4B.  Higher is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 3 or len(y_true) != len(y_pred):
        return float("nan")

    if np.std(y_true) <= 0 or np.std(y_pred) <= 0:
        return float("nan")

    corr = np.corrcoef(y_true, y_pred)[0, 1]

    if not np.isfinite(corr):
        return float("nan")

    return float(corr)


def recovery_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Skill score against a mean-only baseline.  Higher is better.

    Available as a secondary metric; pearson_correlation is preferred
    because its observed value is directly interpretable and scale-free.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return float("nan")

    denom = np.mean((y_true - np.mean(y_true)) ** 2)

    if not np.isfinite(denom) or denom <= 0:
        return float("nan")

    mse = np.mean((y_true - y_pred) ** 2)

    if not np.isfinite(mse):
        return float("nan")

    return float(1.0 - mse / denom)


# ---------------------------------------------------------------------
# OOF Ridge probe
# ---------------------------------------------------------------------

def _fit_oof_ridge_predictions(
    Z: np.ndarray,
    y: np.ndarray,
    block_id: np.ndarray,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Leave-one-block-out ridge predictions.

    For each block, fit ridge on all other blocks, predict on the held-out
    block.  The normal matrix A = Z^T Z + lambda*I depends only on
    features, but we recompute per fold (each fold removes one block's
    rows) to keep the design honest.

    Returns an array of OOF predictions aligned with the input rows.
    """
    n, d = Z.shape
    y_pred = np.full(n, np.nan)
    unique_blocks = np.unique(block_id)

    for b in unique_blocks:
        test_mask = block_id == b
        train_mask = ~test_mask

        Z_train = Z[train_mask]
        y_train = y[train_mask]
        Z_test = Z[test_mask]

        if Z_train.shape[0] < d + 1:
            # Not enough training samples; leave predictions as NaN
            continue

        # Closed-form ridge: w = (Z^T Z + alpha*I)^{-1} Z^T y
        A = Z_train.T @ Z_train + ridge_alpha * np.eye(d)
        try:
            w = np.linalg.solve(A, Z_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue

        y_pred[test_mask] = Z_test @ w

    return y_pred


def _fit_oof_ridge_predictions_multi_rhs(
    Z: np.ndarray,
    Y_matrix: np.ndarray,
    block_id: np.ndarray,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Leave-one-block-out ridge predictions for multiple RHS targets.

    Y_matrix shape: (n, B) where B is the number of permuted targets.
    Returns (n, B) OOF prediction matrix.

    The key optimization: A = Z_train^T Z_train + alpha*I is factored
    once per fold and solved for all B right-hand sides simultaneously.
    This makes permute-before-probe ~3.5x serial cost, not 999x.
    """
    n, d = Z.shape
    B = Y_matrix.shape[1]
    Y_pred = np.full((n, B), np.nan)
    unique_blocks = np.unique(block_id)

    for b in unique_blocks:
        test_mask = block_id == b
        train_mask = ~test_mask

        Z_train = Z[train_mask]
        Z_test = Z[test_mask]
        Y_train = Y_matrix[train_mask]  # shape (n_train, B)

        if Z_train.shape[0] < d + 1:
            continue

        A = Z_train.T @ Z_train + ridge_alpha * np.eye(d)
        try:
            # Multi-RHS solve: W shape (d, B)
            W = np.linalg.solve(A, Z_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue

        Y_pred[test_mask] = Z_test @ W

    return Y_pred


# ---------------------------------------------------------------------
# Non-collapse metrics
# ---------------------------------------------------------------------

def compute_noncollapse_metrics(
    Z: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> NonCollapseMetrics:
    """Compute intrinsic non-collapse diagnostics for embedding matrix Z.

    Z shape: n_units x n_dims.  This is not a permutation-null test.
    """
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 2:
        raise ValueError("Z must be a 2-D array with shape n_units x n_dims.")

    n_total, n_dims = Z.shape
    finite_rows = np.all(np.isfinite(Z), axis=1)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if len(valid_mask) != n_total:
            raise ValueError("valid_mask must have one entry per row of Z.")
        finite_rows = finite_rows & valid_mask

    n_valid = int(finite_rows.sum())
    coverage = n_valid / n_total if n_total else 0.0

    if n_valid == 0 or n_dims == 0:
        return NonCollapseMetrics(
            effective_rank=float("nan"),
            total_variance=float("nan"),
            pairwise_dispersion=float("nan"),
            n_units=n_valid,
            n_dims=n_dims,
            coverage=coverage,
        )

    Z_valid = Z[finite_rows]
    Z_centered = Z_valid - np.mean(Z_valid, axis=0, keepdims=True)

    total_variance = float(np.sum(np.var(Z_centered, axis=0)))
    pairwise_dispersion = float(np.mean(np.sum(Z_centered ** 2, axis=1)))

    if n_valid < 2:
        effective_rank = float("nan")
    else:
        singular_values = np.linalg.svd(Z_centered, compute_uv=False)
        eigen_mass = singular_values ** 2
        total_mass = float(np.sum(eigen_mass))

        if total_mass <= eps or not np.isfinite(total_mass):
            effective_rank = 0.0
        else:
            p = eigen_mass / total_mass
            p = p[p > eps]
            entropy = -float(np.sum(p * np.log(p)))
            effective_rank = float(np.exp(entropy))

    return NonCollapseMetrics(
        effective_rank=effective_rank,
        total_variance=total_variance,
        pairwise_dispersion=pairwise_dispersion,
        n_units=n_valid,
        n_dims=n_dims,
        coverage=coverage,
    )


def evaluate_noncollapse_guard(
    metrics: NonCollapseMetrics,
    config: Gate4Config,
) -> Dict:
    """Gate 4A: intrinsic non-collapse guard."""
    payload = asdict(metrics)

    if metrics.coverage < config.min_noncollapse_coverage:
        return {"verdict": "INSUFFICIENT", "reason": "INSUFFICIENT_NONCOLLAPSE_COVERAGE", "metrics": payload}

    if metrics.n_units < 3:
        return {"verdict": "INSUFFICIENT", "reason": "INSUFFICIENT_LATENT_UNITS", "metrics": payload}

    if metrics.n_dims < 2:
        return {"verdict": "INSUFFICIENT", "reason": "INSUFFICIENT_LATENT_DIMENSIONS", "metrics": payload}

    if not np.isfinite(metrics.effective_rank):
        return {"verdict": "INSUFFICIENT", "reason": "INSUFFICIENT_EFFECTIVE_RANK", "metrics": payload}

    if not np.isfinite(metrics.total_variance):
        return {"verdict": "INSUFFICIENT", "reason": "INSUFFICIENT_LATENT_VARIANCE", "metrics": payload}

    if not np.isfinite(metrics.pairwise_dispersion):
        return {"verdict": "INSUFFICIENT", "reason": "INSUFFICIENT_PAIRWISE_DISPERSION", "metrics": payload}

    if metrics.total_variance <= config.min_total_variance:
        return {"verdict": "FAIL", "reason": "FAIL_ZERO_LATENT_VARIANCE", "metrics": payload}

    if metrics.pairwise_dispersion <= config.min_pairwise_dispersion:
        return {"verdict": "FAIL", "reason": "FAIL_ZERO_PAIRWISE_DISPERSION", "metrics": payload}

    if metrics.effective_rank < config.min_effective_rank:
        return {"verdict": "FAIL", "reason": "FAIL_COLLAPSE", "metrics": payload}

    warn_floor = max(config.warn_effective_rank_floor, 0.05 * metrics.n_dims)

    if metrics.effective_rank < warn_floor:
        return {"verdict": "WARN", "reason": "WARN_LOW_EFFECTIVE_RANK", "metrics": payload}

    return {"verdict": "PASS", "reason": "PASS_NON_COLLAPSE", "metrics": payload}


# ---------------------------------------------------------------------
# Spatial block permutation null
# ---------------------------------------------------------------------

class SpatialBlockPermutationNull:
    """Spatially restricted block permutation null.

    Permutes proxy values within pre-existing spatial CV blocks only,
    never across them.  Sub-minimum blocks are excluded from both the
    observed statistic and the null via admissible_mask().
    """

    def __init__(self, block_id: Sequence, *, min_block_size: int = 3):
        self.block_id = np.asarray(block_id)
        self.min_block_size = int(min_block_size)

        if self.block_id.ndim != 1:
            raise ValueError("block_id must be a 1-D per-unit label array.")

        if self.min_block_size < 2:
            raise ValueError("min_block_size must be >= 2.")

    def admissible_mask(self) -> np.ndarray:
        """Boolean mask of units whose block meets the size floor."""
        mask = np.zeros(len(self.block_id), dtype=bool)
        for b in np.unique(self.block_id):
            idx = np.where(self.block_id == b)[0]
            if len(idx) >= self.min_block_size:
                mask[idx] = True
        return mask

    def permute(
        self,
        values: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Permute within each block."""
        values = np.asarray(values)
        if len(values) != len(self.block_id):
            raise ValueError("values and block_id must have the same length.")

        permuted = values.copy()
        for b in np.unique(self.block_id):
            idx = np.where(self.block_id == b)[0]
            permuted[idx] = rng.permutation(permuted[idx])
        return permuted

    def permute_matrix(
        self,
        values: np.ndarray,
        rng: np.random.Generator,
        n_perm: int,
    ) -> np.ndarray:
        """Generate n_perm within-block permutations as columns.

        Returns shape (n_units, n_perm).  Each column is an independent
        within-block permutation of `values`.
        """
        n = len(values)
        result = np.empty((n, n_perm), dtype=values.dtype)
        for j in range(n_perm):
            result[:, j] = self.permute(values, rng)
        return result


# ---------------------------------------------------------------------
# Gate 4B: permute-before-probe checkability evaluator
# ---------------------------------------------------------------------

def evaluate_checkability_permute_before_probe(
    *,
    name: str,
    Z: np.ndarray,
    y_true_proxy: np.ndarray,
    block_id: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    direction: Direction = "higher_is_better",
    alpha: float = 0.05,
    n_perm: int = 999,
    base_seed: int = 42,
    ridge_alpha: float = 1.0,
    min_block_size: int = 3,
    min_admissible_units: int = 10,
    min_distinct_null: Optional[int] = None,
    distinct_round_decimals: int = 12,
) -> NullResult:
    """Permute-before-probe checkability test.

    For the observed statistic:
        1. Fit leave-one-block-out ridge: Z -> y_true_proxy
        2. Score OOF predictions against y_true_proxy

    For each null draw:
        1. Shuffle y_true_proxy within spatial blocks
        2. Refit leave-one-block-out ridge: Z -> shuffled_proxy
        3. Score OOF predictions against shuffled_proxy

    This tests the full pipeline (probe + metric) under H0, so any
    between-block structure the probe manufactures is also present in
    the null and cancels.  Fixes the anticonservative null that caused
    11.5% false-positive rate in the permute-after design.

    The ridge normal matrix is factored once per fold via multi-RHS
    solve, making the cost ~3.5x serial (not 999x).
    """
    Z = np.asarray(Z, dtype=float)
    y_true = np.asarray(y_true_proxy, dtype=float)
    block_id = np.asarray(block_id)
    n_total = len(y_true)

    if min_distinct_null is None:
        min_distinct_null = int(np.ceil(1.0 / alpha))

    def _insufficient(reason, observed=np.nan, coverage=0.0, n_distinct=0):
        return NullResult(
            observed=observed, p_value=np.nan, n_perm=0,
            n_distinct_null=n_distinct, coverage=coverage,
            resolvable=False, direction=direction, reason=reason,
        )

    if n_total == 0:
        return _insufficient("INSUFFICIENT_EMPTY_PROXY")

    # 1. Finite mask
    finite = np.isfinite(y_true) & np.all(np.isfinite(Z), axis=1)
    if finite.sum() == 0:
        return _insufficient("INSUFFICIENT_NO_FINITE_PROXY")

    # 2. Block-size admissibility after finite masking
    finite_idx = np.where(finite)[0]
    finite_blocks = block_id[finite_idx]
    sampler = SpatialBlockPermutationNull(finite_blocks, min_block_size=min_block_size)
    admissible_local = sampler.admissible_mask()
    admissible_idx = finite_idx[admissible_local]

    n_admissible = len(admissible_idx)
    coverage = n_admissible / n_total if n_total else 0.0

    if n_admissible < min_admissible_units:
        return _insufficient("INSUFFICIENT_ADMISSIBLE_UNITS", coverage=coverage)

    # Restrict to admissible units
    Z_a = Z[admissible_idx]
    y_a = y_true[admissible_idx]
    blocks_a = block_id[admissible_idx]
    sampler_a = SpatialBlockPermutationNull(blocks_a, min_block_size=min_block_size)

    # 3. Observed: fit OOF ridge on real proxy, score
    y_pred_obs = _fit_oof_ridge_predictions(Z_a, y_a, blocks_a, ridge_alpha)
    valid_obs = np.isfinite(y_pred_obs)

    if valid_obs.sum() < min_admissible_units:
        return _insufficient("INSUFFICIENT_OOF_PREDICTIONS", coverage=coverage)

    observed = metric_fn(y_a[valid_obs], y_pred_obs[valid_obs])
    if not np.isfinite(observed):
        return _insufficient("INSUFFICIENT_NONFINITE_OBSERVED", observed=observed, coverage=coverage)

    # 4. Null: permute proxy within blocks, refit OOF ridge, score
    rng = np.random.default_rng(_derive_seed(base_seed, name))

    # Build permuted proxy matrix: (n_admissible, n_perm)
    Y_perm = sampler_a.permute_matrix(y_a, rng, n_perm)

    # Multi-RHS OOF ridge for all permutations at once
    Y_pred_perm = _fit_oof_ridge_predictions_multi_rhs(
        Z_a, Y_perm, blocks_a, ridge_alpha,
    )

    # Score each permutation
    null_scores = np.full(n_perm, np.nan)
    for j in range(n_perm):
        y_perm_j = Y_perm[:, j]
        y_pred_j = Y_pred_perm[:, j]
        valid_j = np.isfinite(y_pred_j) & np.isfinite(y_perm_j)
        if valid_j.sum() >= min_admissible_units:
            null_scores[j] = metric_fn(y_perm_j[valid_j], y_pred_j[valid_j])

    finite_null = np.isfinite(null_scores)
    if finite_null.sum() < n_perm * 0.5:
        return _insufficient(
            "INSUFFICIENT_NONFINITE_NULL_STATISTICS",
            observed=observed, coverage=coverage,
            n_distinct=int(finite_null.sum()),
        )

    null_scores_clean = null_scores[finite_null]

    # 5. Resolvability
    rounded = np.round(null_scores_clean, distinct_round_decimals)
    n_distinct = int(len(np.unique(rounded)))

    if n_distinct < min_distinct_null:
        return _insufficient(
            "INSUFFICIENT_NULL_RESOLUTION",
            observed=observed, coverage=coverage, n_distinct=n_distinct,
        )

    # 6. p-value
    p_value = empirical_p_value(observed, null_scores_clean, direction)

    return NullResult(
        observed=observed, p_value=p_value,
        n_perm=int(finite_null.sum()),
        n_distinct_null=n_distinct,
        coverage=coverage, resolvable=True,
        direction=direction, reason=None,
    )


def evaluate_proxy_checks(
    *,
    Z: np.ndarray,
    specs: Sequence[ProxyCheckSpec],
    block_id: Sequence,
    config: Gate4Config,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = pearson_correlation,
) -> Dict[str, NullResult]:
    """Evaluate all physical-proxy checkability channels.

    The metric_fn is applied uniformly across all channels for
    comparability.  Default is pearson_correlation (scale-free).
    """
    Z = np.asarray(Z, dtype=float)
    block_id = np.asarray(block_id)

    results: Dict[str, NullResult] = {}
    for spec in specs:
        if spec.name in results:
            raise ValueError(f"Duplicate proxy check name: {spec.name}")

        results[spec.name] = evaluate_checkability_permute_before_probe(
            name=spec.name,
            Z=Z,
            y_true_proxy=spec.y_true_proxy,
            block_id=block_id,
            metric_fn=metric_fn,
            direction=spec.direction,
            alpha=config.alpha,
            n_perm=config.n_perm,
            base_seed=config.base_seed,
            ridge_alpha=config.ridge_alpha,
            min_block_size=config.min_block_size,
            min_admissible_units=config.min_admissible_units,
            min_distinct_null=config.min_distinct_null,
            distinct_round_decimals=config.distinct_round_decimals,
        )

    return results


# ---------------------------------------------------------------------
# Gate 4 verdict
# ---------------------------------------------------------------------

def _result_to_payload(
    result: NullResult,
    *,
    alpha: float,
    min_proxy_coverage: float,
) -> Dict:
    """Convert a NullResult into a machine-readable channel payload."""
    payload = {
        "observed": result.observed,
        "p_value": result.p_value,
        "n_perm": result.n_perm,
        "n_distinct_null": result.n_distinct_null,
        "coverage": result.coverage,
        "resolvable": result.resolvable,
        "direction": result.direction,
        "reason": result.reason,
        "verdict": None,
    }

    if result.coverage < min_proxy_coverage:
        payload["verdict"] = "INSUFFICIENT"
        payload["reason"] = "INSUFFICIENT_PROXY_COVERAGE"
        return payload

    if not result.resolvable or not np.isfinite(result.p_value):
        payload["verdict"] = "INSUFFICIENT"
        payload["reason"] = result.reason or "INSUFFICIENT_PROXY_NULL"
        return payload

    if result.p_value < alpha:
        payload["verdict"] = "PASS"
    else:
        payload["verdict"] = "FAIL"

    return payload


def gate4_grounding_verdict(
    *,
    noncollapse_metrics: NonCollapseMetrics,
    checkability_results: Dict[str, NullResult],
    required_proxy_names: Sequence[str],
    config: Gate4Config,
    block_source: str = "existing_spatial_cv_partition",
) -> Gate4Result:
    """Compute the Gate 4 grounding verdict.

    Verdict logic:
        PASS:         non-collapse passes, all required proxies pass.
        WARN:         non-collapse warns but required proxies pass, OR
                      at least one required proxy passes and one fails.
        FAIL:         non-collapse fails, OR no required proxy passes.
        INSUFFICIENT: non-collapse insufficient, OR required proxy
                      missing/unresolvable.
    """
    if not required_proxy_names:
        raise ValueError("required_proxy_names must contain at least one proxy name.")

    noncollapse = evaluate_noncollapse_guard(noncollapse_metrics, config)

    checkability_payload = {
        name: _result_to_payload(
            result, alpha=config.alpha,
            min_proxy_coverage=config.min_proxy_coverage,
        )
        for name, result in checkability_results.items()
    }

    def _result(verdict, reason):
        return Gate4Result(
            verdict=verdict, reason=reason,
            noncollapse=noncollapse,
            checkability=checkability_payload,
            alpha=config.alpha, n_perm=config.n_perm,
            base_seed=config.base_seed,
            null_mode="permute_before_probe",
            block_source=block_source,
            required_proxies=list(required_proxy_names),
        )

    # Missing required proxies
    missing = [n for n in required_proxy_names if n not in checkability_payload]
    if missing:
        return _result("INSUFFICIENT", f"INSUFFICIENT_MISSING_REQUIRED_PROXY:{','.join(missing)}")

    required_payloads = {n: checkability_payload[n] for n in required_proxy_names}

    # Non-collapse gates
    if noncollapse["verdict"] == "INSUFFICIENT":
        return _result("INSUFFICIENT", noncollapse["reason"])

    # Unresolvable required proxies
    unresolved = [n for n, p in required_payloads.items() if p["verdict"] == "INSUFFICIENT"]
    if unresolved:
        reasons = [f"{n}:{required_payloads[n].get('reason')}" for n in unresolved]
        return _result("INSUFFICIENT", "INSUFFICIENT_REQUIRED_PROXY:" + "|".join(reasons))

    if noncollapse["verdict"] == "FAIL":
        return _result("FAIL", noncollapse["reason"])

    passed = [n for n, p in required_payloads.items() if p["verdict"] == "PASS"]
    failed = [n for n, p in required_payloads.items() if p["verdict"] == "FAIL"]

    if len(passed) == 0:
        return _result("FAIL", "FAIL_PROXY_CHECKABILITY")

    warn_reasons = []
    if noncollapse["verdict"] == "WARN":
        warn_reasons.append(noncollapse["reason"])
    if failed:
        warn_reasons.append(
            f"WARN_PARTIAL_PROXY_CHECKABILITY:passed={','.join(passed)};failed={','.join(failed)}"
        )

    if warn_reasons:
        return _result("WARN", "|".join(warn_reasons))

    return _result("PASS", "PASS_GROUNDED_PROXY_CHECKABLE")


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------

def evaluate_gate4(
    *,
    Z: np.ndarray,
    proxy_specs: Sequence[ProxyCheckSpec],
    block_id: Sequence,
    required_proxy_names: Sequence[str],
    config: Optional[Gate4Config] = None,
    latent_valid_mask: Optional[np.ndarray] = None,
    block_source: str = "existing_spatial_cv_partition",
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = pearson_correlation,
) -> Gate4Result:
    """End-to-end Gate 4 evaluation.

    Inputs:
        Z:                     n_units x n_dims representation matrix.
        proxy_specs:           Exogenous physical-proxy recovery checks.
        block_id:              Per-unit spatial CV block assignment.
        required_proxy_names:  Ex ante required proxies for the scenario.
        config:                Predeclared Gate 4 constants.
        metric_fn:             Metric for checkability (default: pearson_correlation).
    """
    if config is None:
        config = Gate4Config()

    Z = np.asarray(Z, dtype=float)
    block_id = np.asarray(block_id)

    if Z.ndim != 2:
        raise ValueError("Z must be 2-D.")
    if len(block_id) != Z.shape[0]:
        raise ValueError("block_id must have one label per row of Z.")

    noncollapse_metrics = compute_noncollapse_metrics(Z, valid_mask=latent_valid_mask)

    checkability_results = evaluate_proxy_checks(
        Z=Z, specs=proxy_specs, block_id=block_id,
        config=config, metric_fn=metric_fn,
    )

    return gate4_grounding_verdict(
        noncollapse_metrics=noncollapse_metrics,
        checkability_results=checkability_results,
        required_proxy_names=required_proxy_names,
        config=config, block_source=block_source,
    )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------

def gate4_result_to_dict(result: Gate4Result) -> Dict:
    """Convert Gate4Result to a JSON-serializable dictionary."""
    return asdict(result)


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import json

    rng = np.random.default_rng(42)
    n, d = 120, 16

    Z = rng.normal(size=(n, d))
    block_id = np.repeat(np.arange(12), 10)

    # Physical proxies (exogenous)
    precipitation = Z @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    hand_elevation = 0.3 * (Z @ rng.normal(size=d)) + rng.normal(size=n)
    water_adjacency = rng.normal(size=n)  # genuinely independent of Z

    config = Gate4Config(alpha=0.05, n_perm=999, base_seed=42)

    # NOTE: No external y_pred. The OOF ridge probe is fitted internally.
    # Leakage is structurally impossible.
    proxy_specs = [
        ProxyCheckSpec(name="precipitation", y_true_proxy=precipitation, required=True),
        ProxyCheckSpec(name="hand_elevation", y_true_proxy=hand_elevation, required=True),
        ProxyCheckSpec(name="water_adjacency", y_true_proxy=water_adjacency, required=True),
    ]

    result = evaluate_gate4(
        Z=Z,
        proxy_specs=proxy_specs,
        block_id=block_id,
        required_proxy_names=["precipitation", "hand_elevation", "water_adjacency"],
        config=config,
    )

    print(json.dumps(gate4_result_to_dict(result), indent=2, default=str))
    print()
    print(f"Verdict: {result.verdict}")
    print(f"Reason:  {result.reason}")
    print(f"Null:    {result.null_mode}")
    for name, ch in result.checkability.items():
        print(f"  {name}: observed={ch['observed']:.3f}  p={ch['p_value']:.4f}  -> {ch['verdict']}")
