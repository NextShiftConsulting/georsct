"""Tests for georsct.domain.construct_divergence_matrix.

Acceptance criteria:
  AC-1: Identical certificates -> distance = 0.
  AC-2: Maximally different certificates -> distance = sqrt(3).
  AC-3: Missing construct -> distance is NaN.
  AC-4: Matrix is (5, 5), symmetric, zero diagonal.
  AC-5: ADR-020 D8 provenance fields non-empty on valid certificates.
  AC-6: summarize_divergence() -> JSON roundtrip preserves all fields.
"""

import json
import math

import numpy as np
import pytest

from georsct.domain.construct_certificate import (
    ConstructCertificate,
    ConstructLabel,
)
from georsct.domain.construct_divergence_matrix import (
    CONSTRUCT_ORDER,
    DivergenceMatrix,
    build_divergence_matrix,
    compute_certificate_distance,
    summarize_divergence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cert(
    construct: ConstructLabel,
    forward: float = 0.5,
    ks: float = 0.5,
    kr: float = 0.5,
    available: bool = True,
) -> ConstructCertificate:
    if not available:
        return ConstructCertificate.missing(construct, "test missing")
    return ConstructCertificate.from_scores(
        construct=construct,
        target_column=f"target_{construct.name.lower()}",
        forward_score=forward,
        kappa_spatial=ks,
        kappa_reconstruct=kr,
        morans_i=0.0,
        n_regions=50,
        n_observations=200,
        n_finite_targets=200,
    )


def _make_all_certs(**overrides) -> list[ConstructCertificate]:
    """One cert per construct, all with the same scores unless overridden."""
    defaults = dict(forward=0.5, ks=0.5, kr=0.5, available=True)
    defaults.update(overrides)
    return [_make_cert(c, **defaults) for c in CONSTRUCT_ORDER]


# =========================================================================
# AC-1: Identical certificates -> distance = 0
# =========================================================================

class TestIdenticalDistance:

    def test_same_scores(self):
        a = _make_cert(ConstructLabel.JRC, 0.7, 0.8, 0.6)
        b = _make_cert(ConstructLabel.DELTARES, 0.7, 0.8, 0.6)
        pw = compute_certificate_distance(a, b)
        assert pw.euclidean_distance == pytest.approx(0.0, abs=1e-12)
        assert pw.forward_delta == pytest.approx(0.0, abs=1e-12)
        assert pw.both_available is True


# =========================================================================
# AC-2: Maximally different -> distance = sqrt(3)
# =========================================================================

class TestMaximalDistance:

    def test_all_ones_vs_all_zeros(self):
        a = _make_cert(ConstructLabel.JRC, 1.0, 1.0, 1.0)
        b = _make_cert(ConstructLabel.NFIP, 0.0, 0.0, 0.0)
        pw = compute_certificate_distance(a, b)
        assert pw.euclidean_distance == pytest.approx(math.sqrt(3), rel=1e-10)

    def test_deltas_signed(self):
        a = _make_cert(ConstructLabel.JRC, 0.9, 0.2, 0.7)
        b = _make_cert(ConstructLabel.FEMA, 0.3, 0.8, 0.1)
        pw = compute_certificate_distance(a, b)
        assert pw.forward_delta == pytest.approx(0.6, abs=1e-10)
        assert pw.kappa_spatial_delta == pytest.approx(-0.6, abs=1e-10)
        assert pw.kappa_reconstruct_delta == pytest.approx(0.6, abs=1e-10)


# =========================================================================
# AC-3: Missing construct -> distance is NaN
# =========================================================================

class TestMissingDistance:

    def test_one_missing(self):
        a = _make_cert(ConstructLabel.JRC, 0.7, 0.8, 0.6)
        b = ConstructCertificate.missing(ConstructLabel.FAST, "no data")
        pw = compute_certificate_distance(a, b)
        assert math.isnan(pw.euclidean_distance)
        assert pw.both_available is False

    def test_both_missing(self):
        a = ConstructCertificate.missing(ConstructLabel.JRC, "missing")
        b = ConstructCertificate.missing(ConstructLabel.DELTARES, "missing")
        pw = compute_certificate_distance(a, b)
        assert math.isnan(pw.euclidean_distance)
        assert pw.both_available is False


# =========================================================================
# AC-4: Matrix is (5, 5), symmetric, zero diagonal
# =========================================================================

class TestDivergenceMatrix:

    def test_shape(self):
        certs = _make_all_certs()
        dm = build_divergence_matrix(certs, "test_geo")
        assert dm.matrix.shape == (5, 5)

    def test_symmetric(self):
        certs = [
            _make_cert(ConstructLabel.JRC, 0.9, 0.8, 0.7),
            _make_cert(ConstructLabel.DELTARES, 0.6, 0.5, 0.4),
            _make_cert(ConstructLabel.FEMA, 0.3, 0.2, 0.1),
            _make_cert(ConstructLabel.FAST, 0.8, 0.7, 0.6),
            _make_cert(ConstructLabel.NFIP, 0.5, 0.4, 0.3),
        ]
        dm = build_divergence_matrix(certs, "test")
        np.testing.assert_array_almost_equal(dm.matrix, dm.matrix.T)

    def test_zero_diagonal(self):
        certs = _make_all_certs()
        dm = build_divergence_matrix(certs, "test")
        np.testing.assert_array_equal(np.diag(dm.matrix), 0.0)

    def test_n_available_all(self):
        certs = _make_all_certs()
        dm = build_divergence_matrix(certs, "test")
        assert dm.n_available == 5

    def test_n_available_partial(self):
        certs = [
            _make_cert(ConstructLabel.JRC, available=True),
            _make_cert(ConstructLabel.DELTARES, available=False),
            _make_cert(ConstructLabel.FEMA, available=True),
            _make_cert(ConstructLabel.FAST, available=False),
            _make_cert(ConstructLabel.NFIP, available=True),
        ]
        dm = build_divergence_matrix(certs, "test")
        assert dm.n_available == 3

    def test_pairwise_count(self):
        certs = _make_all_certs()
        dm = build_divergence_matrix(certs, "test")
        assert len(dm.pairwise) == 10  # C(5, 2) = 10

    def test_geography_id_preserved(self):
        certs = _make_all_certs()
        dm = build_divergence_matrix(certs, "houston")
        assert dm.geography_id == "houston"

    def test_max_pair_identified(self):
        certs = [
            _make_cert(ConstructLabel.JRC, 1.0, 1.0, 1.0),
            _make_cert(ConstructLabel.DELTARES, 0.5, 0.5, 0.5),
            _make_cert(ConstructLabel.FEMA, 0.0, 0.0, 0.0),
            _make_cert(ConstructLabel.FAST, 0.5, 0.5, 0.5),
            _make_cert(ConstructLabel.NFIP, 0.5, 0.5, 0.5),
        ]
        dm = build_divergence_matrix(certs, "test")
        assert set(dm.max_pair) == {ConstructLabel.JRC, ConstructLabel.FEMA}

    def test_duplicate_construct_raises(self):
        certs = [_make_cert(ConstructLabel.JRC)] * 5
        with pytest.raises(ValueError, match="Duplicate"):
            build_divergence_matrix(certs)

    def test_wrong_count_raises(self):
        certs = [_make_cert(ConstructLabel.JRC)]
        with pytest.raises(ValueError, match="Expected 5"):
            build_divergence_matrix(certs)

    def test_all_identical_scores_zero_matrix(self):
        certs = _make_all_certs(forward=0.6, ks=0.7, kr=0.8)
        dm = build_divergence_matrix(certs, "test")
        assert dm.mean_distance == pytest.approx(0.0, abs=1e-12)
        assert dm.max_distance == pytest.approx(0.0, abs=1e-12)


# =========================================================================
# AC-5: ADR-020 D8 provenance on valid certificates
# =========================================================================

class TestProvenanceInMatrix:

    def test_valid_certificates_have_provenance(self):
        certs = _make_all_certs()
        dm = build_divergence_matrix(certs, "test")
        for cert in dm.certificates:
            assert cert.kappa_reconstruct_source is not None
            assert len(cert.kappa_reconstruct_source) > 0
            assert cert.kappa_spatial_source is not None
            assert len(cert.kappa_spatial_source) > 0
            assert cert.kappa_reconstruct_authority is not None
            assert cert.kappa_spatial_authority is not None

    def test_missing_certificates_have_none_provenance(self):
        certs = [
            _make_cert(ConstructLabel.JRC, available=True),
            _make_cert(ConstructLabel.DELTARES, available=False),
            _make_cert(ConstructLabel.FEMA, available=True),
            _make_cert(ConstructLabel.FAST, available=False),
            _make_cert(ConstructLabel.NFIP, available=True),
        ]
        dm = build_divergence_matrix(certs, "test")
        for cert in dm.certificates:
            if not cert.target_available:
                assert cert.kappa_reconstruct_source is None
                assert cert.kappa_spatial_source is None


# =========================================================================
# AC-6: summarize_divergence() -> JSON roundtrip
# =========================================================================

class TestSummarizeDivergence:

    def test_json_roundtrip(self):
        certs = [
            _make_cert(ConstructLabel.JRC, 0.9, 0.8, 0.7),
            _make_cert(ConstructLabel.DELTARES, 0.6, 0.5, 0.4),
            _make_cert(ConstructLabel.FEMA, 0.3, 0.2, 0.1),
            _make_cert(ConstructLabel.FAST, 0.8, 0.7, 0.6),
            _make_cert(ConstructLabel.NFIP, 0.5, 0.4, 0.3),
        ]
        dm = build_divergence_matrix(certs, "houston")
        summary = summarize_divergence(dm)

        # Must be JSON-serializable
        text = json.dumps(summary)
        parsed = json.loads(text)

        assert parsed["geography_id"] == "houston"
        assert parsed["n_available"] == 5
        assert len(parsed["per_construct"]) == 5
        assert len(parsed["pairwise"]) == 10
        assert len(parsed["distance_matrix"]) == 5
        assert len(parsed["construct_order"]) == 5

    def test_nan_becomes_none(self):
        certs = [
            _make_cert(ConstructLabel.JRC, available=True),
            _make_cert(ConstructLabel.DELTARES, available=False),
            _make_cert(ConstructLabel.FEMA, available=True),
            _make_cert(ConstructLabel.FAST, available=False),
            _make_cert(ConstructLabel.NFIP, available=True),
        ]
        dm = build_divergence_matrix(certs, "test")
        summary = summarize_divergence(dm)
        text = json.dumps(summary)  # NaN would crash JSON

        # Find the missing cert entry
        for entry in summary["per_construct"]:
            if not entry["target_available"]:
                assert entry["forward_score"] is None
                assert entry["kappa_spatial"] is None

    def test_construct_order_matches(self):
        certs = _make_all_certs()
        dm = build_divergence_matrix(certs, "test")
        summary = summarize_divergence(dm)
        order = summary["construct_order"]
        assert order == [c.value for c in CONSTRUCT_ORDER]
