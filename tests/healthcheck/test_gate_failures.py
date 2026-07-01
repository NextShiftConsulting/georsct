"""Layer 1: Gate failure tests.

Tests verify decision parity with canonical SequentialGatekeeper from
yrsn-controlplane, using controlplane vocabulary for gate_evidence keys
and failure_reason (mapped to sub_signal) strings.

Key semantics:
  - Gate 1: OR logic (noise_admissibility_ok OR alpha_ok)
  - Gate 1 input: noise_admissibility from evidence, fallback to raw N
  - Gate 2: respects gate_2_require_coherence (False for GEOSPATIAL_CONUS27)
  - Oobleck: lambda_turbulence=0.0 for geo -> flat kappa_req=0.22
"""

from tests.healthcheck.conftest import make_cert
from georsct.healthcheck.layers.gate_triage import evaluate_gates
from georsct.healthcheck.thresholds import GEOSPATIAL_CONUS27, UNIVERSAL


P = GEOSPATIAL_CONUS27


class TestGate1Integrity:
    """Gate 1: OR logic — pass if noise_admissibility < N_thr OR alpha >= alpha_min."""

    def test_reject_both_noise_and_alpha_fail(self):
        """Both checks fail -> REJECT."""
        cert = make_cert(N=0.55, alpha=0.25)
        result = evaluate_gates(cert, P)
        assert result.decision == "REJECT"
        assert result.gate_reached == "GATE_1_INTEGRITY"
        assert result.sub_signal == "gate_1_noise_above_threshold"

    def test_pass_high_noise_but_good_alpha(self):
        """N > N_thr but alpha >= alpha_min -> OR passes."""
        cert = make_cert(N=0.60, alpha=0.45)
        result = evaluate_gates(cert, P)
        ev = result.gate_evidence["gate_1_integrity"]
        assert ev["n_check_passed"] is False
        assert ev["alpha_check_passed"] is True
        assert ev["status"] == "passed"

    def test_pass_low_alpha_but_low_noise(self):
        """alpha < alpha_min but N < N_thr -> OR passes."""
        cert = make_cert(N=0.20, alpha=0.25)
        result = evaluate_gates(cert, P)
        ev = result.gate_evidence["gate_1_integrity"]
        assert ev["n_check_passed"] is True
        assert ev["alpha_check_passed"] is False
        assert ev["status"] == "passed"

    def test_pass_both_ok(self):
        cert = make_cert(N=0.30, alpha=0.50)
        result = evaluate_gates(cert, P)
        assert result.gate_evidence["gate_1_integrity"]["status"] == "passed"

    def test_boundary_n_thr_is_050(self):
        """N_thr is 0.50 from canonical preset, not 0.70."""
        assert P.N_thr == 0.50
        # N=0.49 passes noise check
        cert = make_cert(N=0.49, alpha=0.25)
        result = evaluate_gates(cert, P)
        assert result.gate_evidence["gate_1_integrity"]["n_check_passed"] is True
        assert result.gate_evidence["gate_1_integrity"]["status"] == "passed"
        # N=0.51 fails noise check but still needs alpha check
        cert2 = make_cert(N=0.51, alpha=0.25)
        result2 = evaluate_gates(cert2, P)
        assert result2.gate_evidence["gate_1_integrity"]["n_check_passed"] is False
        assert result2.gate_evidence["gate_1_integrity"]["alpha_check_passed"] is False
        assert result2.decision == "REJECT"


class TestGate1NoiseAdmissibility:
    """Gate 1 uses noise_admissibility from evidence, falls back to raw N."""

    def test_uses_noise_admissibility_when_present(self):
        cert = make_cert(N=0.80)
        cert["evidence"] = {"noise_admissibility": 0.30}
        result = evaluate_gates(cert, P)
        ev = result.gate_evidence["gate_1_integrity"]
        assert ev["noise_admissibility"] == 0.30
        assert ev["n_check_passed"] is True

    def test_fallback_to_raw_n_when_absent(self):
        cert = make_cert(N=0.55, alpha=0.25)
        result = evaluate_gates(cert, P)
        ev = result.gate_evidence["gate_1_integrity"]
        assert ev["noise_admissibility"] == 0.55

    def test_noise_admissibility_overrides_raw_n(self):
        """noise_admissibility=PASS but raw N=FAIL -> PASS."""
        cert = make_cert(N=0.80, alpha=0.20)
        cert["evidence"] = {"noise_admissibility": 0.40}
        result = evaluate_gates(cert, P)
        ev = result.gate_evidence["gate_1_integrity"]
        assert ev["n_check_passed"] is True
        assert ev["status"] == "passed"


class TestGate2Consensus:
    """Gate 2 respects gate_2_require_coherence (False for GEOSPATIAL_CONUS27)."""

    def test_coherence_present_still_checked_for_geo(self):
        """When coherence IS present, it's always checked even if require=False."""
        assert P.gate_2_require_coherence is False
        cert = make_cert(alpha=0.50, N=0.30, coherence=0.10)
        result = evaluate_gates(cert, P)
        assert result.decision == "BLOCK"
        assert result.gate_reached == "GATE_2_CONSENSUS"

    def test_coherence_absent_legacy_pass_for_geo(self):
        """When coherence is absent and require=False, pass (legacy path)."""
        assert P.gate_2_require_coherence is False
        cert = make_cert(alpha=0.50, N=0.30)
        result = evaluate_gates(cert, P)
        ev = result.gate_evidence["gate_2_consensus"]
        assert ev["coherence"] is None
        assert ev["failure_path"] == "coherence_absent_legacy_pass"

    def test_coherence_enforced_for_universal(self):
        """UNIVERSAL has gate_2_require_coherence=True."""
        u = UNIVERSAL
        assert u.gate_2_require_coherence is True
        cert = make_cert(alpha=0.50, N=0.30, coherence=0.10)
        result = evaluate_gates(cert, u)
        assert result.decision == "BLOCK"
        assert result.gate_reached == "GATE_2_CONSENSUS"

    def test_coherence_absent_blocks_for_universal(self):
        """Fail closed when coherence required but absent."""
        u = UNIVERSAL
        cert = make_cert(alpha=0.50, N=0.30)
        result = evaluate_gates(cert, u)
        assert result.decision == "BLOCK"
        assert result.sub_signal == "gate_2_coherence_below_threshold"


class TestGate3Admissibility:
    def test_oobleck_geo_is_flat(self):
        """GEOSPATIAL_CONUS27 has lambda_turbulence=0.0 -> kappa_req constant."""
        assert P.lambda_turbulence == 0.0
        assert P.kappa_req(0.0) == P.kappa_base
        assert P.kappa_req(0.5) == P.kappa_base
        assert P.kappa_req(1.0) == P.kappa_base
        assert P.kappa_base == 0.22

    def test_oobleck_universal_is_sigmoidal(self):
        """UNIVERSAL has lambda_turbulence=0.4 -> kappa_req varies."""
        u = UNIVERSAL
        assert u.lambda_turbulence == 0.4
        low = u.kappa_req(0.0)
        high = u.kappa_req(1.0)
        assert high > low + 0.1, f"Expected sigmoid variation, got {low:.3f} -> {high:.3f}"

    def test_re_encode_kappa_below_oobleck(self):
        cert = make_cert(sigma=0.20, kappa_compat=0.20, alpha=0.50, N=0.30)
        result = evaluate_gates(cert, P)
        assert result.decision == "RE_ENCODE"
        assert result.gate_reached == "GATE_3_ADMISSIBILITY"

    def test_sigmoid_math_matches_pinned_formula(self):
        import math
        sigma = 0.35
        x = P.oobleck_steepness * (sigma - P.oobleck_sigma_c)
        expected = P.kappa_base + P.lambda_turbulence * (1.0 / (1.0 + math.exp(-x)))
        assert abs(P.kappa_req(sigma) - expected) < 1e-10

    def test_landauer_gray_zone_low_sigma_forgives(self):
        """Gray zone with sigma below tiebreaker -> PASS (canonical behavior).

        SequentialGatekeeper uses sigma tiebreaker in the gray zone:
        sigma <= landauer_sigma_tiebreaker -> forgive, PASS.
        """
        sigma = 0.20  # below landauer_sigma_tiebreaker=0.40
        kappa_req = P.kappa_req(sigma)
        cert = make_cert(
            sigma=sigma,
            kappa_compat=kappa_req - P.epsilon_L * 0.5,
            alpha=0.50,
            N=0.30,
        )
        result = evaluate_gates(cert, P)
        # Gray zone forgiven because sigma is low
        assert result.gate_evidence["gate_3_admissibility"]["status"] == "passed"

    def test_landauer_gray_zone_high_sigma_fails(self):
        """Gray zone with sigma above tiebreaker -> RE_ENCODE."""
        kappa_req = P.kappa_req(0.45)
        cert = make_cert(
            sigma=0.45,  # above landauer_sigma_tiebreaker=0.40
            kappa_compat=kappa_req - P.epsilon_L * 0.5,
            alpha=0.50,
            N=0.30,
        )
        result = evaluate_gates(cert, P)
        assert result.decision == "RE_ENCODE"
        assert result.sub_signal == "gate_3_landauer_gray_zone_sigma_tiebreaker"

    def test_landauer_above_kappa_req_passes(self):
        sigma = 0.20
        kappa_req = P.kappa_req(sigma)
        cert = make_cert(
            sigma=sigma,
            kappa_compat=kappa_req + P.epsilon_L * 0.5,
            alpha=0.50,
            N=0.30,
        )
        result = evaluate_gates(cert, P)
        assert result.gate_evidence["gate_3_admissibility"]["status"] == "passed"


class TestGate4Grounding:
    def test_repair_low_grounding(self):
        cert = make_cert(kappa_L=0.10, alpha=0.50, N=0.30, sigma=0.10, kappa_compat=0.80)
        result = evaluate_gates(cert, P)
        assert result.decision == "REPAIR"
        assert result.gate_reached == "GATE_4_GROUNDING"
        assert result.sub_signal == "gate_4_low_grounding_below_threshold"


class TestAllPass:
    def test_execute(self):
        cert = make_cert(alpha=0.70, kappa_compat=0.80, sigma=0.10, N=0.20)
        result = evaluate_gates(cert, P)
        assert result.decision == "EXECUTE"
        assert result.gate_reached == "ALL_PASSED"
        assert result.sub_signal is None


class TestPresetParity:
    """Prove rsct-healthcheck uses the same GEOSPATIAL_CONUS27 values as canonical."""

    def test_geospatial_conus27_matches_controlplane(self):
        from yrsn_controlplane import PRESET_GEOSPATIAL_CONUS27 as cfg
        assert P.N_thr == cfg.N_thr
        assert P.alpha_min == cfg.alpha_min
        assert P.c_min == cfg.c_min
        assert P.gate_2_require_coherence == cfg.gate_2_require_coherence
        assert P.sigma_thr == cfg.sigma_thr
        assert P.kappa_base == cfg.kappa_base
        assert P.lambda_turbulence == cfg.lambda_turbulence
        assert P.epsilon_L == cfg.epsilon_L
        assert P.oobleck_steepness == cfg.steepness
        assert P.oobleck_sigma_c == cfg.sigma_c
        assert P.landauer_sigma_tiebreaker == cfg.landauer_sigma_tiebreaker
        assert P.enable_gate_3b == cfg.enable_gate_3b
        assert P.r_bar_min == cfg.r_bar_min
        assert P.kappa_L_min == cfg.kappa_L_min

    def test_oobleck_agrees_with_controlplane(self):
        """kappa_req() output must match canonical _oobleck_kappa_req()."""
        from yrsn_controlplane.gatekeeper import _oobleck_kappa_req
        from yrsn_controlplane import PRESET_GEOSPATIAL_CONUS27 as cfg
        for sigma in [0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0]:
            local = P.kappa_req(sigma)
            canon = _oobleck_kappa_req(
                sigma, cfg.kappa_base, cfg.lambda_turbulence,
                cfg.steepness, cfg.sigma_c,
            )
            assert abs(local - canon) < 1e-12, (
                f"sigma={sigma}: local={local}, canonical={canon}"
            )

    def test_previous_66_reject_case_no_longer_mass_rejects(self):
        """With N_thr=0.50+OR, a typical geo cert should not mass-reject."""
        cert = make_cert(N=0.525, alpha=0.475, kappa_compat=0.22, sigma=0.12)
        result = evaluate_gates(cert, P)
        ev = result.gate_evidence["gate_1_integrity"]
        assert ev["n_check_passed"] is False
        assert ev["alpha_check_passed"] is True
        assert ev["status"] == "passed"
        assert result.gate_reached != "GATE_1_INTEGRITY"


class TestKappaUnavailable:
    """kappa_compat=None is handled at the bridging layer."""

    def test_block_when_kappa_none(self):
        cert = make_cert(alpha=0.50, N=0.30)
        cert["kappa_compat"] = None
        result = evaluate_gates(cert, P)
        assert result.decision == "BLOCK"
        assert result.gate_reached == "GATE_3_ADMISSIBILITY"
        assert result.sub_signal == "kappa_unavailable"
