"""
kappa_reconstruct: spatial recoverability of learned representations
====================================================================

Measures whether a representation's implied neighbor structure is
*topologically coherent* when realized on the true spatial support.

The instrument is a Gabriel graph built in embedding space, drawn on
true region centroids, with crossings counted as "flyover" violations.
A crossing means the representation asserts two regions are neighbors
via a link that passes over intervening territory.

Orthogonality to kappa_spatial (Moran's I on residuals):
  - kappa_spatial: do model *errors* cluster spatially?
  - kappa_reconstruct: is the *representation's implied geometry* planar?
  Different objects. High residual autocorrelation with a perfectly planar
  implied graph (smooth, structured errors) and low autocorrelation with
  a non-planar representation can coexist.

Lineage: S018U backward recoverability concept, instantiated for
geospatial topology.

Reference: rsct-governance/physics/ (future P-008 candidate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import Delaunay


# =========================================================================
# Gabriel graph construction
# =========================================================================

def gabriel_graph(points: np.ndarray) -> list[tuple[int, int]]:
    """Build Gabriel graph from points. Parameter-free.

    An edge (i, j) is in the Gabriel graph iff no other point k lies
    inside the open diametral ball of segment (i, j):
        ||p_k - midpoint|| < ||p_i - p_j|| / 2  for any k => remove edge

    Computed via Delaunay triangulation (Gabriel is a subgraph of Delaunay)
    then filtering edges by the diametral ball criterion.

    Args:
        points: (n, d) array of point coordinates in any dimension.

    Returns:
        List of (i, j) edges with i < j.
    """
    n = points.shape[0]
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    tri = Delaunay(points)
    candidate_edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a in range(len(simplex)):
            for b in range(a + 1, len(simplex)):
                i, j = int(simplex[a]), int(simplex[b])
                candidate_edges.add((min(i, j), max(i, j)))

    # Filter: keep edge (i,j) only if no point k falls inside the
    # open diametral ball (midpoint = center, radius = half edge length)
    gabriel_edges = []
    for i, j in candidate_edges:
        mid = (points[i] + points[j]) / 2.0
        radius_sq = np.sum((points[i] - points[j]) ** 2) / 4.0
        dists_sq = np.sum((points - mid) ** 2, axis=1)
        dists_sq[i] = radius_sq + 1.0
        dists_sq[j] = radius_sq + 1.0
        if np.all(dists_sq >= radius_sq):
            gabriel_edges.append((i, j))

    return gabriel_edges


# =========================================================================
# Edge crossing detection (the "flyover" test)
# =========================================================================

def _segments_cross_2d(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> bool:
    """Test whether segment (p1,p2) properly crosses segment (p3,p4) in 2D.

    Proper crossing: segments intersect at a single interior point of both.
    Shared endpoints or collinear overlap do not count.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return False
    d3 = p3 - p1
    t = (d3[0] * d2[1] - d3[1] * d2[0]) / denom
    u = (d3[0] * d1[1] - d3[1] * d1[0]) / denom
    return (0.0 < t < 1.0) and (0.0 < u < 1.0)


def count_crossings(
    edges: list[tuple[int, int]],
    coords2d: np.ndarray,
) -> int:
    """Count pairs of edges that cross when drawn on 2D coordinates.

    Args:
        edges: list of (i, j) edges (from Gabriel graph in embedding space).
        coords2d: (n, 2) true spatial coordinates (centroids).

    Returns:
        Number of crossing pairs.
    """
    n_crossings = 0
    n_edges = len(edges)
    for a in range(n_edges):
        i, j = edges[a]
        for b in range(a + 1, n_edges):
            k, el = edges[b]
            if i == k or i == el or j == k or j == el:
                continue
            if _segments_cross_2d(
                coords2d[i], coords2d[j],
                coords2d[k], coords2d[el],
            ):
                n_crossings += 1
    return n_crossings


# =========================================================================
# Baseline: expected crossings under degree-matched random graph
# =========================================================================

def _random_crossing_baseline(
    n_nodes: int,
    n_edges: int,
    coords2d: np.ndarray,
    n_trials: int = 20,
    rng: np.random.Generator | None = None,
) -> float:
    """Expected crossing count for a degree-matched random graph.

    Generates random edge sets of the same size and counts crossings.
    Returns mean crossing count across trials.
    """
    if n_edges < 2 or n_nodes < 4:
        return 0.0
    if rng is None:
        rng = np.random.default_rng(42)
    counts = []
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    if n_edges > len(all_pairs):
        return 0.0
    for _ in range(n_trials):
        idx = rng.choice(len(all_pairs), size=n_edges, replace=False)
        random_edges = [all_pairs[k] for k in idx]
        counts.append(count_crossings(random_edges, coords2d))
    return float(np.mean(counts))


# =========================================================================
# Mantel test (labeled context, never gated)
# =========================================================================

def mantel_correlation(
    embedding_dists: np.ndarray,
    geo_dists: np.ndarray,
    n_perms: int = 999,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Mantel test: Pearson correlation between distance matrices.

    Labeled context for the certificate, NEVER gated. Measures
    distance-correlation (Tobler signal), not topological recoverability.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = embedding_dists.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    x = embedding_dists[triu_idx]
    y = geo_dists[triu_idx]

    obs_r = float(np.corrcoef(x, y)[0, 1])

    n_ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        perm_geo = geo_dists[np.ix_(perm, perm)]
        perm_y = perm_geo[triu_idx]
        perm_r = float(np.corrcoef(x, perm_y)[0, 1])
        if perm_r >= obs_r:
            n_ge += 1
    p_value = (n_ge + 1) / (n_perms + 1)
    return obs_r, p_value


# =========================================================================
# MDS stress (corroboration, not gated)
# =========================================================================

def mds_stress(
    embedding_dists: np.ndarray,
    coords2d: np.ndarray,
) -> float:
    """Kruskal stress-1: how well do 2D coordinates reproduce embedding distances?

    stress = sqrt(sum((d_emb - d_2d)^2) / sum(d_emb^2))

    Reported as corroboration. Not gated.
    """
    n = coords2d.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    d_emb = embedding_dists[triu_idx]
    d_2d = np.sqrt(
        np.sum((coords2d[:, None, :] - coords2d[None, :, :]) ** 2, axis=-1)
    )[triu_idx]
    if np.sum(d_emb ** 2) < 1e-12:
        return 1.0
    return float(np.sqrt(np.sum((d_emb - d_2d) ** 2) / np.sum(d_emb ** 2)))


# =========================================================================
# Coordinate lift guard (SatCLIP/PDFM future-proofing)
# =========================================================================

def coordinate_lift(
    embedding_dists: np.ndarray,
    geo_dists: np.ndarray,
    kappa_raw: float,
) -> float:
    """How much of kappa_reconstruct is explained by coordinate proximity?

    For feature-based substrates (R0-R5), lift is ~0 because features
    do not encode coordinates directly. For location-conditioned encoders
    (SatCLIP, PDFM), lift saturates near kappa_raw.

    When lift > 0.8 * kappa_raw, flag: kappa_reconstruct may be measuring
    coordinate encoding, not task-relevant topology.
    """
    n = embedding_dists.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    r = float(np.corrcoef(
        embedding_dists[triu_idx],
        geo_dists[triu_idx],
    )[0, 1])
    return float(np.clip(abs(r) * kappa_raw, 0.0, 1.0))


# =========================================================================
# Main result type and computation
# =========================================================================

@dataclass(frozen=True)
class SpatialRecoverabilityResult:
    """Full result from spatial recoverability computation.

    Headline: kappa_reconstruct (baseline-corrected crossing-violation rate).
    Corroboration (not gated): stress, mantel_r, coordinate_lift_score.
    """

    kappa_reconstruct: float
    n_gabriel_edges: int
    n_crossings: int
    n_crossings_baseline: float
    stress: float
    mantel_r: float
    mantel_p: float
    coordinate_lift_score: float
    n_regions: int


def compute_kappa_reconstruct(
    embeddings: np.ndarray,
    coords2d: np.ndarray,
    geo_dists: Optional[np.ndarray] = None,
    n_baseline_trials: int = 20,
    n_mantel_perms: int = 999,
) -> SpatialRecoverabilityResult:
    """Compute spatial recoverability score.

    We construct a parameter-free Gabriel implied-neighbor graph in
    representation space and test whether that graph remains planar
    when realized on the true spatial support.

    Args:
        embeddings: (n, d) region embeddings from the substrate.
        coords2d: (n, 2) true spatial coordinates (centroids in
            projected CRS). Used to draw the implied graph and detect
            crossings.
        geo_dists: (n, n) pairwise geographic distances. If None,
            computed from coords2d as Euclidean.
        n_baseline_trials: random-graph trials for baseline correction.
        n_mantel_perms: permutations for Mantel test.

    Returns:
        SpatialRecoverabilityResult.
    """
    n = embeddings.shape[0]
    coords2d = np.asarray(coords2d, dtype=np.float64)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    rng = np.random.default_rng(42)

    # 1. Gabriel graph in embedding space
    edges = gabriel_graph(embeddings)
    n_edges = len(edges)

    # 2. Count crossings on true coordinates
    n_crossings = count_crossings(edges, coords2d)

    # 3. Baseline correction
    baseline = _random_crossing_baseline(
        n, n_edges, coords2d, n_trials=n_baseline_trials, rng=rng,
    )

    # 4. Score: 1 - excess crossings / max possible
    max_possible = max(n_edges * (n_edges - 1) // 2, 1)
    excess = max(n_crossings - baseline, 0.0)
    kappa = float(np.clip(1.0 - excess / max_possible, 0.0, 1.0))

    # 5. Distance matrices for corroboration
    emb_dists = np.sqrt(
        np.sum((embeddings[:, None, :] - embeddings[None, :, :]) ** 2, axis=-1)
    )
    if geo_dists is None:
        geo_dists = np.sqrt(
            np.sum((coords2d[:, None, :] - coords2d[None, :, :]) ** 2, axis=-1)
        )

    # 6. Corroboration
    stress_val = mds_stress(emb_dists, coords2d)
    m_r, m_p = mantel_correlation(
        emb_dists, geo_dists, n_perms=n_mantel_perms, rng=rng,
    )
    lift = coordinate_lift(emb_dists, geo_dists, kappa)

    return SpatialRecoverabilityResult(
        kappa_reconstruct=kappa,
        n_gabriel_edges=n_edges,
        n_crossings=n_crossings,
        n_crossings_baseline=baseline,
        stress=stress_val,
        mantel_r=m_r,
        mantel_p=m_p,
        coordinate_lift_score=lift,
        n_regions=n,
    )


# =========================================================================
# Gate 3B decision -- MOVED to application layer (P4 separation)
# =========================================================================
# Canonical location: georsct.application.use_cases.gate_3b_decision
# Re-exported here for backwards compatibility with existing callers.

def gate_3b_decision(
    forward_score: float,
    kappa_reconstruct: float,
    forward_floor: float = 0.0,
    reconstruct_floor: float = 0.3,
) -> str:
    """Gate 3B: spatial recoverability.

    .. deprecated::
        Import from ``georsct.application.use_cases.gate_3b_decision``
        instead.  This re-export exists for backwards compatibility only.
    """
    from georsct.application.use_cases.gate_3b_decision import (
        gate_3b_decision as _canonical,
    )
    return _canonical(
        forward_score, kappa_reconstruct, forward_floor, reconstruct_floor,
    )


# =========================================================================
# Adversarial geography permutation (the "not kappa_spatial twice" proof)
# =========================================================================

@dataclass(frozen=True)
class AdversarialPermutationResult:
    """Result of adversarial W-matrix permutation test.

    Residualizes delta_kappa_reconstruct on delta_kappa_spatial across
    permutation trials. If the residual mean is significantly negative,
    kappa_reconstruct responds to topology through a channel
    kappa_spatial does not cover.

    EFT failure condition: if the residual is not significantly negative,
    kappa_reconstruct stays a diagnostic, not a gate.
    """

    delta_reconstruct_mean: float
    delta_spatial_mean: float
    residual_mean: float
    residual_std: float
    residual_t: float
    residual_p: float
    n_trials: int
    earns_gate: bool


def adversarial_geography_permutation(
    refit_and_embed_fn,
    baseline_reconstruct: float,
    baseline_spatial: float,
    coords2d: np.ndarray,
    n_trials: int = 10,
    geo_dists: Optional[np.ndarray] = None,
    significance: float = 0.05,
) -> AdversarialPermutationResult:
    """Run adversarial W-matrix permutation and test over-and-above kappa_spatial.

    For each trial:
      1. Permute the W-matrix (scramble spatial structure)
      2. Call refit_and_embed_fn(W_perm) -> (embeddings, forward_score, kappa_spatial)
      3. Compute kappa_reconstruct on the permuted embeddings
      4. Record deltas from baseline

    Then: residualize delta_reconstruct on delta_spatial. If the residual
    is significantly negative, kappa_reconstruct captures topology that
    kappa_spatial misses.

    Args:
        refit_and_embed_fn: callable(adjacency_dict) ->
            (embeddings: ndarray, forward_score: float, kappa_spatial: float)
            Must retrain the model with permuted W-matrix and return the
            result. This is the adapter the caller must provide.
        baseline_reconstruct: kappa_reconstruct from the real W-matrix.
        baseline_spatial: kappa_spatial from the real W-matrix.
        coords2d: (n, 2) true spatial coordinates.
        n_trials: number of permutation trials.
        geo_dists: (n, n) geographic distances (optional).
        significance: p-value threshold for earns_gate decision.

    Returns:
        AdversarialPermutationResult.
    """
    from scipy import stats

    delta_reconstructs = []
    delta_spatials = []

    for trial in range(n_trials):
        embeddings, fwd, ks = refit_and_embed_fn(trial)
        result = compute_kappa_reconstruct(
            embeddings, coords2d, geo_dists=geo_dists,
            n_baseline_trials=10, n_mantel_perms=99,
        )
        delta_reconstructs.append(result.kappa_reconstruct - baseline_reconstruct)
        delta_spatials.append(ks - baseline_spatial)

    dr = np.array(delta_reconstructs)
    ds = np.array(delta_spatials)

    # Residualize: regress delta_reconstruct on delta_spatial, take residuals
    if np.std(ds) > 1e-8:
        slope = np.cov(dr, ds)[0, 1] / np.var(ds)
        residuals = dr - slope * ds
    else:
        residuals = dr

    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    if res_std > 1e-8 and len(residuals) > 1:
        t_stat = res_mean / (res_std / np.sqrt(len(residuals)))
        p_val = float(stats.t.cdf(t_stat, df=len(residuals) - 1))
    else:
        t_stat = 0.0
        p_val = 1.0

    return AdversarialPermutationResult(
        delta_reconstruct_mean=float(np.mean(dr)),
        delta_spatial_mean=float(np.mean(ds)),
        residual_mean=res_mean,
        residual_std=res_std,
        residual_t=t_stat,
        residual_p=p_val,
        n_trials=n_trials,
        earns_gate=(p_val < significance and res_mean < 0),
    )


# =========================================================================
# KAPPA REGISTRY ENTRY
# =========================================================================

KAPPA_REGISTRY_ENTRY = {
    "name": "kappa_reconstruct",
    "full_name": "Spatial Recoverability Score",
    "domain": "spatial_topology",
    "formula": "1 - (crossings - baseline) / max_possible_crossings",
    "inputs": ["embeddings (n, d)", "coords2d (n, 2)"],
    "range": "[0, 1]",
    "monotonicity": "higher = more planar-consistent implied topology",
    "gate": "Gate 3B (spatial recoverability)",
    "gate_condition": (
        "forward_score >= floor AND kappa_reconstruct < reconstruct_floor "
        "=> RE_ENCODE"
    ),
    "orthogonality": (
        "kappa_spatial measures residual autocorrelation (Moran's I). "
        "kappa_reconstruct measures planarity of the implied neighbor graph. "
        "Different objects: clustered errors (high kappa_spatial) can coexist "
        "with a perfectly planar representation, and vice versa."
    ),
    "lineage": (
        "S018U backward recoverability concept -> "
        "geospatial topology instantiation"
    ),
    "status": "EXPERIMENTAL",
    "corroboration": [
        "stress (MDS)",
        "mantel_r (distance correlation)",
        "coordinate_lift (location guard)",
    ],
    "admitted": "2026-06-11",
}
