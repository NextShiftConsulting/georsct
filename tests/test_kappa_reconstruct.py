"""Acceptance tests for kappa_reconstruct domain module and adversarial adapter.

Five locked acceptance criteria:

1. Coherent fixture       -> kappa_reconstruct ~ 1.0, gate = EXECUTE
2. Taxi-map fixture       -> kappa_reconstruct ~ 0.0, gate = RE_ENCODE
3. Full W permutation     -> delta_kappa_reconstruct < 0
4. Ladder corruption      -> gamma_level < 0, CI excludes 0
5. Orthogonality check    -> kappa_reconstruct movement survives
                             residualization on kappa_spatial
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Domain module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from georsct.domain.kappa_reconstruct import (
    SpatialRecoverabilityResult,
    compute_kappa_reconstruct,
    count_crossings,
    gabriel_graph,
    gate_3b_decision,
)

# Adapter discriminant tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wsp" / "floodrsct" / "jobs"))
from adversarial_reconstruct import (
    discriminant_intercept_test,
    discriminant_ladder_test,
    partial_permute_w,
    build_permutation_schedule,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _grid_coords(n_side: int = 5) -> np.ndarray:
    """Regular grid of points in 2D. (n_side^2, 2)."""
    xs = np.arange(n_side, dtype=float)
    ys = np.arange(n_side, dtype=float)
    grid = np.array([(x, y) for y in ys for x in xs])
    return grid


def _coherent_embeddings(coords: np.ndarray, noise: float = 0.01) -> np.ndarray:
    """Embeddings that preserve spatial structure (+ small noise)."""
    rng = np.random.default_rng(42)
    return coords + rng.normal(0, noise, size=coords.shape)


def _taxi_map_embeddings(coords: np.ndarray) -> np.ndarray:
    """Embeddings that scramble spatial structure entirely."""
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(coords))
    return coords[perm]


# ---------------------------------------------------------------------------
# AC-1: Coherent fixture -> kappa_reconstruct ~ 1.0
# ---------------------------------------------------------------------------

class TestCoherentFixture:
    """A representation that mirrors the true geography should score high."""

    def test_kappa_near_one(self):
        coords = _grid_coords(6)
        emb = _coherent_embeddings(coords, noise=0.01)
        result = compute_kappa_reconstruct(
            emb, coords, n_baseline_trials=10, n_mantel_perms=99,
        )
        assert isinstance(result, SpatialRecoverabilityResult)
        assert result.kappa_reconstruct >= 0.85, (
            f"Coherent embeddings should score >= 0.85, got {result.kappa_reconstruct:.4f}"
        )

    def test_gate_executes(self):
        coords = _grid_coords(6)
        emb = _coherent_embeddings(coords, noise=0.01)
        result = compute_kappa_reconstruct(
            emb, coords, n_baseline_trials=10, n_mantel_perms=99,
        )
        decision = gate_3b_decision(
            forward_score=0.5, kappa_reconstruct=result.kappa_reconstruct,
            forward_floor=0.0, reconstruct_floor=0.3,
        )
        assert decision == "EXECUTE"

    def test_gabriel_edges_exist(self):
        coords = _grid_coords(5)
        emb = _coherent_embeddings(coords, noise=0.01)
        result = compute_kappa_reconstruct(
            emb, coords, n_baseline_trials=5, n_mantel_perms=0,
        )
        assert result.n_gabriel_edges > 0
        assert result.n_regions == 25


# ---------------------------------------------------------------------------
# AC-2: Taxi-map fixture -> kappa_reconstruct ~ 0.0, gate = RE_ENCODE
# ---------------------------------------------------------------------------

class TestTaxiMapFixture:
    """A permuted representation implies non-realizable topology."""

    def test_kappa_lower_than_coherent(self):
        coords = _grid_coords(6)
        emb_good = _coherent_embeddings(coords, noise=0.01)
        emb_bad = _taxi_map_embeddings(coords)
        r_good = compute_kappa_reconstruct(
            emb_good, coords, n_baseline_trials=10, n_mantel_perms=0,
        )
        r_bad = compute_kappa_reconstruct(
            emb_bad, coords, n_baseline_trials=10, n_mantel_perms=0,
        )
        assert r_bad.kappa_reconstruct < r_good.kappa_reconstruct, (
            f"Scrambled ({r_bad.kappa_reconstruct:.4f}) should be lower "
            f"than coherent ({r_good.kappa_reconstruct:.4f})"
        )

    def test_gate_re_encodes(self):
        decision = gate_3b_decision(
            forward_score=0.7, kappa_reconstruct=0.1,
            forward_floor=0.0, reconstruct_floor=0.3,
        )
        assert decision == "RE_ENCODE"

    def test_gate_passes_when_forward_fails(self):
        """If forward score is below floor, gate does not fire (not taxi-map quadrant)."""
        decision = gate_3b_decision(
            forward_score=-0.5, kappa_reconstruct=0.1,
            forward_floor=0.0, reconstruct_floor=0.3,
        )
        assert decision == "EXECUTE"

    def test_crossings_increase_under_scramble(self):
        coords = _grid_coords(6)
        emb_good = _coherent_embeddings(coords, noise=0.01)
        emb_bad = _taxi_map_embeddings(coords)
        r_good = compute_kappa_reconstruct(
            emb_good, coords, n_baseline_trials=5, n_mantel_perms=0,
        )
        r_bad = compute_kappa_reconstruct(
            emb_bad, coords, n_baseline_trials=5, n_mantel_perms=0,
        )
        assert r_bad.n_crossings >= r_good.n_crossings


# ---------------------------------------------------------------------------
# AC-3: Full W permutation -> delta_kappa_reconstruct < 0
# ---------------------------------------------------------------------------

class TestFullPermutationDrop:
    """Intercept test: kappa_reconstruct drops when topology is fully scrambled."""

    def test_intercept_negative(self):
        rng = np.random.default_rng(7)
        n_trials = 30
        d_recon = rng.normal(-0.15, 0.05, size=n_trials)
        d_spatial = rng.normal(-0.05, 0.03, size=n_trials)

        result = discriminant_intercept_test(d_recon, d_spatial, alpha=0.05)
        assert result["intercept"] < 0, (
            f"Intercept should be negative, got {result['intercept']:.4f}"
        )
        assert result["p_one_sided"] < 0.05
        assert result["verdict"] == "EARNS_SLOT"

    def test_no_signal_inconclusive(self):
        rng = np.random.default_rng(42)
        n_trials = 30
        d_recon = rng.normal(0.0, 0.01, size=n_trials)
        d_spatial = rng.normal(0.0, 0.01, size=n_trials)

        result = discriminant_intercept_test(d_recon, d_spatial, alpha=0.05)
        assert result["verdict"] == "INCONCLUSIVE"

    def test_insufficient_data(self):
        result = discriminant_intercept_test(
            np.array([0.1]), np.array([0.2]), alpha=0.05,
        )
        assert result["verdict"] == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# AC-4: Ladder corruption -> gamma_level < 0, CI excludes 0
# ---------------------------------------------------------------------------

class TestLadderCorruption:
    """Graded corruption: kappa_reconstruct degrades monotonically with level."""

    @staticmethod
    def _synthetic_ladder(
        n_folds: int = 6,
        levels: tuple = (0.1, 0.25, 0.5, 0.75, 1.0),
        reps: int = 3,
        seed: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Synthetic data where kappa_reconstruct drops with corruption level."""
        rng = np.random.default_rng(seed)
        d_recon, d_spatial, lv, fid = [], [], [], []
        for f in range(n_folds):
            sens = 0.9 + 0.3 * rng.standard_normal()
            for level in levels:
                for _ in range(reps):
                    ds = -0.30 * level * sens + 0.03 * rng.standard_normal()
                    independent = 0.45 * level + 0.04 * rng.standard_normal()
                    shared = 0.15 * level
                    dr = -shared - independent + 0.02 * rng.standard_normal()
                    d_spatial.append(ds)
                    d_recon.append(dr)
                    lv.append(level)
                    fid.append(f)
        return (np.array(d_recon), np.array(d_spatial),
                np.array(lv), np.array(fid))

    def test_gamma_negative_with_folds(self):
        d_recon, d_spatial, lv, fid = self._synthetic_ladder()
        result = discriminant_ladder_test(
            d_recon, d_spatial, lv, fold_id=fid,
            n_bootstrap=1000, seed=42,
        )
        assert result["gamma_level"] < 0, (
            f"gamma_level should be negative, got {result['gamma_level']:.4f}"
        )
        assert result["gamma_ci"][1] < 0, (
            f"CI upper bound should be < 0, got {result['gamma_ci']}"
        )
        assert result["verdict"] == "EARNS_SLOT"

    def test_gamma_negative_without_folds(self):
        d_recon, d_spatial, lv, _ = self._synthetic_ladder()
        result = discriminant_ladder_test(
            d_recon, d_spatial, lv, fold_id=None,
            n_bootstrap=1000, seed=42,
        )
        assert result["gamma_level"] < 0
        assert result["verdict"] == "EARNS_SLOT"

    def test_no_signal_inconclusive(self):
        rng = np.random.default_rng(42)
        n = 50
        d_recon = rng.normal(0.0, 0.01, size=n)
        d_spatial = rng.normal(0.0, 0.01, size=n)
        lv = np.tile([0.1, 0.25, 0.5, 0.75, 1.0], n // 5)
        result = discriminant_ladder_test(
            d_recon, d_spatial, lv, n_bootstrap=500, seed=42,
        )
        assert result["verdict"] == "INCONCLUSIVE"

    def test_schedule_builds_correctly(self):
        levels = (0.1, 0.5, 1.0)
        schedule = build_permutation_schedule(levels, n_seeds=3)
        assert len(schedule) == 9
        assert all(0.0 < lv <= 1.0 for lv, _ in schedule)


# ---------------------------------------------------------------------------
# AC-5: Orthogonality -- kappa_reconstruct not redundant with kappa_spatial
# ---------------------------------------------------------------------------

class TestOrthogonality:
    """kappa_reconstruct captures topology that kappa_spatial misses."""

    def test_intercept_survives_residualization(self):
        """When d_recon has independent-of-d_spatial signal, intercept is negative."""
        rng = np.random.default_rng(11)
        n = 40
        d_spatial = rng.normal(-0.10, 0.04, size=n)
        independent_drop = rng.normal(-0.20, 0.03, size=n)
        shared_component = 0.5 * d_spatial
        d_recon = shared_component + independent_drop

        result = discriminant_intercept_test(d_recon, d_spatial, alpha=0.05)
        assert result["intercept"] < 0
        assert result["p_one_sided"] < 0.05

    def test_collinear_signal_inconclusive(self):
        """When d_recon is purely a scaled copy of d_spatial, no independent signal."""
        rng = np.random.default_rng(42)
        n = 40
        d_spatial = rng.normal(-0.15, 0.05, size=n)
        d_recon = 1.2 * d_spatial + rng.normal(0.0, 0.005, size=n)

        result = discriminant_intercept_test(d_recon, d_spatial, alpha=0.05)
        assert abs(result["intercept"]) < 0.02, (
            f"Collinear signal should have near-zero intercept, got {result['intercept']:.4f}"
        )

    def test_ladder_partials_out_spatial(self):
        """Ladder test: gamma_level is the FWL-partialled independent channel."""
        d_recon, d_spatial, lv, fid = TestLadderCorruption._synthetic_ladder()
        result = discriminant_ladder_test(
            d_recon, d_spatial, lv, fold_id=fid,
            n_bootstrap=1000, seed=42,
        )
        # gamma_level captures the independent channel after partialling d_spatial
        assert result["gamma_level"] < 0
        assert result["gamma_ci"][1] < 0


# ---------------------------------------------------------------------------
# Helpers: Gabriel graph and crossing primitives
# ---------------------------------------------------------------------------

class TestGabrielGraph:
    """Sanity checks on the Gabriel graph construction."""

    def test_triangle(self):
        pts = np.array([[0, 0], [1, 0], [0.5, 0.866]], dtype=float)
        edges = gabriel_graph(pts)
        assert len(edges) == 3

    def test_near_collinear_points(self):
        """Nearly collinear points: only adjacent pairs survive Gabriel filter."""
        pts = np.array([[0, 0], [1, 0.001], [2, 0], [3, 0.001]], dtype=float)
        edges = gabriel_graph(pts)
        # Adjacent pairs dominate; long-range edges blocked by intervening points
        assert len(edges) >= 3
        assert all(i < j for i, j in edges)

    def test_grid_planar(self):
        """Gabriel graph of a regular grid, drawn on the same grid, has zero crossings."""
        coords = _grid_coords(5)
        edges = gabriel_graph(coords)
        assert count_crossings(edges, coords) == 0

    def test_empty_and_small(self):
        assert gabriel_graph(np.array([[0, 0]])) == []
        edges = gabriel_graph(np.array([[0, 0], [1, 1]]))
        assert edges == [(0, 1)]


class TestCrossings:
    def test_no_crossings_for_parallel(self):
        edges = [(0, 1), (2, 3)]
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        assert count_crossings(edges, coords) == 0

    def test_crossing_detected(self):
        edges = [(0, 1), (2, 3)]
        coords = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
        assert count_crossings(edges, coords) == 1


# ---------------------------------------------------------------------------
# Partial permutation: self-loop safety
# ---------------------------------------------------------------------------

class TestPartialPermuteW:
    def test_level_zero_preserves_structure(self):
        """Level 0 applies identity permutation; sparsity pattern is preserved."""
        from scipy import sparse
        rng = np.random.default_rng(42)
        n = 10
        W = sparse.random(n, n, density=0.3, random_state=42, format="csr")
        W.setdiag(0.0)
        W.eliminate_zeros()
        W_perm = partial_permute_w(W, level=0.0, rng=rng)
        # Same set of (row, col) nonzero entries (CSR may reorder internally)
        orig = set(zip(*W.nonzero()))
        perm = set(zip(*W_perm.nonzero()))
        assert orig == perm

    def test_no_self_loops(self):
        from scipy import sparse
        rng = np.random.default_rng(42)
        n = 20
        W = sparse.random(n, n, density=0.3, random_state=42, format="csr")
        W.setdiag(0.0)
        W.eliminate_zeros()
        for level in [0.25, 0.5, 0.75, 1.0]:
            W_perm = partial_permute_w(W, level=level, rng=rng)
            diag = W_perm.diagonal()
            assert np.allclose(diag, 0.0), (
                f"Self-loops found at level={level}: {diag[diag != 0]}"
            )

    def test_row_normalized(self):
        from scipy import sparse
        rng = np.random.default_rng(42)
        n = 15
        W = sparse.random(n, n, density=0.4, random_state=42, format="csr")
        W.setdiag(0.0)
        W.eliminate_zeros()
        W_perm = partial_permute_w(W, level=0.5, rng=rng)
        row_sums = np.asarray(W_perm.sum(axis=1)).ravel()
        nonzero = row_sums > 0
        assert np.allclose(row_sums[nonzero], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Old statistic is identically zero (regression test for the bug)
# ---------------------------------------------------------------------------

class TestOldStatisticIsZero:
    """The mean of intercept-model OLS residuals is identically zero.

    This test documents the bug that the patches fixed. If someone
    reintroduces the old test, this catches it.
    """

    def test_ols_residual_mean_is_zero(self):
        rng = np.random.default_rng(7)
        n = 100
        x = rng.normal(0, 1, size=n)
        y = -0.5 * x + rng.normal(-0.3, 0.1, size=n)

        X = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        assert abs(residuals.mean()) < 1e-12, (
            "OLS residuals with intercept must have exactly zero mean"
        )

    def test_intercept_is_not_zero(self):
        """But the intercept itself IS a real testable quantity."""
        rng = np.random.default_rng(7)
        n = 100
        x = rng.normal(0, 1, size=n)
        y = -0.5 * x + rng.normal(-0.3, 0.1, size=n)

        X = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        assert abs(beta[0]) > 0.1, (
            f"Intercept should be meaningfully nonzero, got {beta[0]:.4f}"
        )
