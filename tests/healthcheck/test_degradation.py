"""Layer 2: Degradation pattern tests."""

from conftest import make_cert
from georsct.healthcheck.layers.degradation import classify_degradation


class TestDegradationGrid:
    def test_healthy(self):
        cert = make_cert(alpha=0.80, kappa_compat=0.85)
        result = classify_degradation(cert, "EXECUTE")
        assert result.degradation_type == "HEALTHY"
        assert result.alpha_level == "HIGH"
        assert result.kappa_level == "HIGH"

    def test_noise_dominant(self):
        cert = make_cert(R=0.30, N=0.50, alpha=0.40, kappa_compat=0.70)
        result = classify_degradation(cert, "REJECT")
        assert result.degradation_type == "NOISE_DOMINANT"

    def test_noise_collapsed(self):
        cert = make_cert(R=0.15, N=0.70, alpha=0.20, kappa_compat=0.50)
        result = classify_degradation(cert, "REJECT")
        assert result.degradation_type == "NOISE_COLLAPSED"

    def test_unstable(self):
        cert = make_cert(sigma=0.35, alpha=0.50, kappa_compat=0.70)
        result = classify_degradation(cert, "RE_ENCODE")
        assert result.degradation_type == "UNSTABLE"

    def test_geometry_mismatch(self):
        cert = make_cert(alpha=0.60, kappa_compat=0.25, sigma=0.10, R=0.50, N=0.30)
        result = classify_degradation(cert, "REPAIR")
        assert result.degradation_type == "GEOMETRY_MISMATCH"

    def test_content_degradation(self):
        cert = make_cert(alpha=0.25, kappa_compat=0.70, sigma=0.10, R=0.50, N=0.30)
        result = classify_degradation(cert, "REJECT")
        assert result.degradation_type == "CONTENT_DEGRADATION"


class TestDiagnosisLabelParsing:
    def test_parse_structured_dict(self):
        cert = make_cert(
            diagnosis_label={
                "degradation_type": "MILD_CONTENT_ISSUE",
                "confidence": 0.33,
                "explanation": "test explanation",
                "alpha_level": "MEDIUM",
                "kappa_level": "HIGH",
            }
        )
        result = classify_degradation(cert, "EXECUTE")
        assert result.degradation_type == "MILD_CONTENT_ISSUE"
        assert result.confidence == 0.33

    def test_parse_stringified_repr(self):
        label = (
            "DiagnosisResult(degradation_type=<DiagnosisDegradationType.UNSTABLE: 5>, "
            "confidence=0.9, recommended_action='Increase samples', "
            "explanation='sigma too high', "
            "alpha_level=<QualityLevel.MEDIUM: 'medium'>, "
            "kappa_level=<QualityLevel.HIGH: 'high'>)"
        )
        cert = make_cert(diagnosis_label=label)
        result = classify_degradation(cert, "RE_ENCODE")
        assert result.degradation_type == "UNSTABLE"
        assert result.confidence == 0.9
        assert result.alpha_level == "MEDIUM"
        assert result.kappa_level == "HIGH"

    def test_malformed_label_falls_back(self):
        cert = make_cert(diagnosis_label="garbage string", alpha=0.80, kappa_compat=0.85)
        result = classify_degradation(cert, "EXECUTE")
        # Falls back to grid classification
        assert result.degradation_type == "HEALTHY"
