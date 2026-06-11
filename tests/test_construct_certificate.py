"""Tests for georsct.domain.construct_certificate.

Acceptance criteria:
  AC-1: ConstructLabel has exactly 5 members.
  AC-2: ConstructCertificate.missing() returns NaN scores, target_available=False.
  AC-3: compute_kappa_spatial on known Moran's I cases.
  AC-4: compute_kappa_spatial returns (NaN, NaN) with < 3 regions.
"""

import math

import numpy as np
import pytest
from scipy import sparse

from georsct.domain.construct_certificate import (
    CONSTRUCT_TARGET_COLUMNS,
    ConstructCertificate,
    ConstructLabel,
    compute_kappa_spatial,
)


# =========================================================================
# AC-1: ConstructLabel has exactly 5 members
# =========================================================================

class TestConstructLabel:

    def test_exactly_five_members(self):
        assert len(ConstructLabel) == 5

    def test_member_names(self):
        names = {m.name for m in ConstructLabel}
        assert names == {"JRC", "DELTARES", "FEMA", "NFIP", "FAST"}

    def test_values_are_descriptive_strings(self):
        for member in ConstructLabel:
            assert isinstance(member.value, str)
            assert len(member.value) > 5

    def test_target_column_mapping_covers_all(self):
        for member in ConstructLabel:
            assert member in CONSTRUCT_TARGET_COLUMNS


# =========================================================================
# AC-2: ConstructCertificate.missing() -- ADR-020 D8
# =========================================================================

class TestMissingCertificate:

    def test_scores_are_nan(self):
        cert = ConstructCertificate.missing(ConstructLabel.JRC, "data not fetched")
        assert math.isnan(cert.forward_score)
        assert math.isnan(cert.kappa_spatial)
        assert math.isnan(cert.kappa_reconstruct)
        assert math.isnan(cert.morans_i)

    def test_not_available(self):
        cert = ConstructCertificate.missing(ConstructLabel.FEMA, "pending")
        assert cert.target_available is False
        assert cert.n_regions == 0
        assert cert.n_observations == 0
        assert cert.n_finite_targets == 0

    def test_provenance_is_none(self):
        cert = ConstructCertificate.missing(ConstructLabel.DELTARES, "no data")
        assert cert.kappa_reconstruct_source is None
        assert cert.kappa_reconstruct_formula is None
        assert cert.kappa_spatial_source is None
        assert cert.kappa_spatial_formula is None

    def test_warning_contains_reason(self):
        reason = "S3 parquet not found"
        cert = ConstructCertificate.missing(ConstructLabel.FAST, reason)
        assert reason in cert.warnings

    def test_scores_never_zero(self):
        """ADR-020 D8: missing kappa = NaN, never 0.0."""
        cert = ConstructCertificate.missing(ConstructLabel.NFIP, "missing")
        assert cert.forward_score != 0.0  # NaN != 0.0 is True
        assert cert.kappa_spatial != 0.0
        assert cert.kappa_reconstruct != 0.0


# =========================================================================
# AC-3: compute_kappa_spatial on known Moran's I cases
# =========================================================================

def _make_rook_w(n: int) -> sparse.csr_matrix:
    """Row-normalized rook contiguity for a 1D chain of n regions."""
    rows, cols, data = [], [], []
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        w = 1.0 / len(neighbors) if neighbors else 0.0
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            data.append(w)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


class TestKappaSpatial:

    def test_checkerboard_negative_i(self):
        """Alternating high-low pattern: Moran's I is negative (dissimilar neighbors)."""
        n = 20
        W = _make_rook_w(n)
        residuals = np.array([(-1.0) ** i for i in range(n)])
        kappa, I = compute_kappa_spatial(residuals, W)
        assert I < -0.5, f"Expected strong negative I, got {I}"
        assert 0.0 <= kappa <= 1.0

    def test_clustered_positive_i(self):
        """Block pattern: first half high, second half low. Strong positive I."""
        n = 20
        W = _make_rook_w(n)
        residuals = np.array([1.0] * (n // 2) + [-1.0] * (n // 2))
        kappa, I = compute_kappa_spatial(residuals, W)
        assert I > 0.5, f"Expected strong positive I, got {I}"
        assert 0.0 <= kappa <= 1.0

    def test_kappa_bounds(self):
        """kappa_spatial is always in [0, 1]."""
        n = 10
        W = _make_rook_w(n)
        rng = np.random.default_rng(42)
        for _ in range(20):
            residuals = rng.standard_normal(n)
            kappa, _ = compute_kappa_spatial(residuals, W)
            if math.isfinite(kappa):
                assert 0.0 <= kappa <= 1.0


# =========================================================================
# AC-4: compute_kappa_spatial returns (NaN, NaN) with < 3 regions
# =========================================================================

class TestKappaSpatialEdgeCases:

    def test_two_regions(self):
        W = _make_rook_w(2)
        residuals = np.array([1.0, -1.0])
        kappa, I = compute_kappa_spatial(residuals, W)
        assert math.isnan(kappa)
        assert math.isnan(I)

    def test_one_region(self):
        W = sparse.csr_matrix((1, 1))
        residuals = np.array([0.5])
        kappa, I = compute_kappa_spatial(residuals, W)
        assert math.isnan(kappa)
        assert math.isnan(I)

    def test_all_nan_values(self):
        W = _make_rook_w(5)
        residuals = np.full(5, np.nan)
        kappa, I = compute_kappa_spatial(residuals, W)
        assert math.isnan(kappa)
        assert math.isnan(I)

    def test_constant_residuals(self):
        """All-same residuals: variance is zero -> NaN."""
        W = _make_rook_w(10)
        residuals = np.ones(10) * 3.0
        kappa, I = compute_kappa_spatial(residuals, W)
        assert math.isnan(kappa)
        assert math.isnan(I)

    def test_morans_i_exceeding_one_is_clamped(self):
        """Binary (non-row-normalized) W can produce |I| > 1; kappa must stay [0,1].

        The theoretical Moran's I range for a non-row-normalized binary W
        is [lambda_min, lambda_max] of (n/s0)*M@W@M, which can exceed [-1, 1].
        We use the eigenvector for the minimum eigenvalue to achieve this.
        """
        g = 10
        n = g * g
        rows, cols, data = [], [], []
        for r in range(g):
            for c in range(g):
                idx = r * g + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < g and 0 <= nc < g:
                        rows.append(idx)
                        cols.append(nr * g + nc)
                        data.append(1.0)  # binary, not row-normalized
        W_binary = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

        # Find residual pattern that achieves |I| > 1 via eigendecomposition
        W_dense = W_binary.toarray()
        M = np.eye(n) - np.ones((n, n)) / n
        B = M @ W_dense @ M
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Eigenvector for minimum eigenvalue is already mean-zero (in range of M)
        residuals = eigenvectors[:, 0]

        kappa, I = compute_kappa_spatial(residuals, W_binary)

        # Raw I should exceed |1| for this non-row-normalized W
        assert abs(I) > 1.0, f"Expected |I| > 1, got I={I}"
        # kappa must be clamped to [0, 1]
        assert 0.0 <= kappa <= 1.0, f"Clamp failed: kappa={kappa}"


# =========================================================================
# from_scores factory
# =========================================================================

class TestFromScores:

    def test_provenance_populated(self):
        cert = ConstructCertificate.from_scores(
            construct=ConstructLabel.NFIP,
            target_column="obs_nfip_event_claims",
            forward_score=0.65,
            kappa_spatial=0.80,
            kappa_reconstruct=0.70,
            morans_i=-0.20,
            n_regions=100,
            n_observations=500,
            n_finite_targets=480,
        )
        assert cert.target_available is True
        assert cert.kappa_reconstruct_source is not None
        assert cert.kappa_spatial_source is not None
        assert "kappa_reconstruct" in cert.kappa_reconstruct_source
        assert "construct_certificate" in cert.kappa_spatial_source

    def test_frozen(self):
        cert = ConstructCertificate.from_scores(
            construct=ConstructLabel.JRC,
            target_column="jrc_occurrence_mean",
            forward_score=0.5,
            kappa_spatial=0.5,
            kappa_reconstruct=0.5,
            morans_i=0.0,
            n_regions=50,
            n_observations=200,
            n_finite_targets=200,
        )
        with pytest.raises(AttributeError):
            cert.forward_score = 0.99  # type: ignore[misc]
