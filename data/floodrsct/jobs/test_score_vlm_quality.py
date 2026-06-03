#!/usr/bin/env python3
"""Fixture test for score_vlm_quality.py programmatic grader.

Verifies that:
  1. Claim extraction pulls the right atomic claims from known responses
  2. Known-fabricated claims are labeled N_claim
  3. Known-correct claims are labeled verified
  4. Tier 1 metrics compute to hand-checkable values
  5. Extraction itself is validated (missed claims, spurious splits)

Run: python test_score_vlm_quality.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score_vlm_quality import (
    extract_claims,
    verify_claim_programmatic,
    compute_tier1_metrics,
    _classify_claim,
)


# ---------------------------------------------------------------------------
# Fixture: source feature row for ZCTA "77001" (Houston-like)
# ---------------------------------------------------------------------------

SOURCE_ROW = pd.Series({
    "zcta_id": "77001",
    "flood_pct_zone_a": 42.3,
    "flood_pct_zone_x": 35.1,
    "flood_pct_zone_x500": 22.6,
    "population": 45000,
    "acs_median_hh_income": 52000,
    "svi_overall": 0.72,
    "obs_nfip_event_claims": 87,
    "obs_has_311": 1,
})

EVIDENCE_TEXT = """ZCTA 77001 in Harris County, Texas.

FEMA Flood Zones:
- 42.3% in Zone A (1% annual chance floodplain)
- 35.1% in Zone X (minimal flood hazard)
- 22.6% in Zone X500 (0.2% annual chance floodplain)

Demographics (ACS):
- Population: 45,000, Median income: $52,000
- SVI overall: 0.72 (High)

Historical Events:
- NFIP claims: 87 events
- 311 flood reports: Yes
"""


# ---------------------------------------------------------------------------
# Test 1: Claim extraction from a known response
# ---------------------------------------------------------------------------

def test_claim_extraction():
    """Verify extraction pulls the right claims from a known response."""
    response = {
        "zone_interpretation": (
            "42.3% of this ZCTA is in Zone A, indicating significant "
            "flood risk. 35.1% is in Zone X with minimal hazard."
        ),
        "vulnerability_factors": [
            "High SVI score of 0.72 indicates social vulnerability",
            "Large population of 45,000 in flood-prone area",
            "History of 87 NFIP claims shows recurring flood events",
        ],
        "spatial_reasoning": (
            "Neighboring ZCTAs also show significant Zone A coverage, "
            "suggesting regional flood exposure."
        ),
        "evidence_used": [
            "42.3% Zone A from FEMA flood zones",
            "Population of 45,000 from ACS data",
            "87 NFIP claims from historical events",
        ],
    }

    claims = extract_claims(response)

    # Should have claims from all four fields
    fields = {c["field"] for c in claims}
    assert "zone_interpretation" in fields, "missing zone_interpretation claims"
    assert "vulnerability_factors" in fields, "missing vulnerability_factors claims"
    assert "spatial_reasoning" in fields, "missing spatial_reasoning claims"
    assert "evidence_used" in fields, "missing evidence_used claims"

    # Should have at least 8 claims (2 zone + 3 vuln + 1 spatial + 3 citation)
    assert len(claims) >= 8, f"expected >= 8 claims, got {len(claims)}"

    # Citation claims should be typed correctly
    citation_claims = [c for c in claims if c["field"] == "evidence_used"]
    assert all(c["claim_type"] == "citation" for c in citation_claims), \
        "evidence_used claims should be typed as citation"

    print(f"  PASS: extracted {len(claims)} claims from 4 fields")
    return claims


# ---------------------------------------------------------------------------
# Test 2: Known-correct numeric claim -> verified
# ---------------------------------------------------------------------------

def test_correct_numeric_verified():
    """A claim with correct numbers should be labeled 'verified'."""
    claim = {
        "text": "42.3% of this ZCTA is in Zone A, the 1% annual chance floodplain",
        "field": "zone_interpretation",
        "claim_type": "numeric",
    }
    result = verify_claim_programmatic(claim, SOURCE_ROW, EVIDENCE_TEXT)
    assert result["tier1_label"] == "verified", \
        f"expected verified, got {result['tier1_label']}: {result['tier1_reason']}"
    print(f"  PASS: correct zone A percentage -> verified")


# ---------------------------------------------------------------------------
# Test 3: Fabricated numeric claim -> N_claim
# ---------------------------------------------------------------------------

def test_fabricated_numeric_n_claim():
    """A claim with wrong numbers should be labeled 'N_claim'."""
    claim = {
        "text": "78.5% of this ZCTA is in Zone A, the 1% annual chance floodplain",
        "field": "zone_interpretation",
        "claim_type": "numeric",
    }
    result = verify_claim_programmatic(claim, SOURCE_ROW, EVIDENCE_TEXT)
    assert result["tier1_label"] == "N_claim", \
        f"expected N_claim, got {result['tier1_label']}: {result['tier1_reason']}"
    print(f"  PASS: fabricated zone A percentage (78.5 vs 42.3) -> N_claim")


# ---------------------------------------------------------------------------
# Test 4: Citation referencing real evidence -> verified
# ---------------------------------------------------------------------------

def test_valid_citation_verified():
    """A citation that exists in the evidence text should be verified."""
    claim = {
        "text": "42.3% Zone A from FEMA flood zones",
        "field": "evidence_used",
        "claim_type": "citation",
    }
    result = verify_claim_programmatic(claim, SOURCE_ROW, EVIDENCE_TEXT)
    assert result["tier1_label"] == "verified", \
        f"expected verified, got {result['tier1_label']}: {result['tier1_reason']}"
    print(f"  PASS: valid citation -> verified")


# ---------------------------------------------------------------------------
# Test 5: Citation referencing non-existent evidence -> N_claim
# ---------------------------------------------------------------------------

def test_fabricated_citation_n_claim():
    """A citation referencing data not in the input should be N_claim."""
    claim = {
        "text": "Elevation data shows average of 12 feet above sea level",
        "field": "evidence_used",
        "claim_type": "citation",
    }
    result = verify_claim_programmatic(claim, SOURCE_ROW, EVIDENCE_TEXT)
    assert result["tier1_label"] == "N_claim", \
        f"expected N_claim, got {result['tier1_label']}: {result['tier1_reason']}"
    print(f"  PASS: fabricated citation (elevation not in evidence) -> N_claim")


# ---------------------------------------------------------------------------
# Test 6: Spatial claim -> unverifiable (requires semantic grading)
# ---------------------------------------------------------------------------

def test_spatial_claim_unverifiable():
    """Spatial claims about neighbors can't be verified programmatically."""
    claim = {
        "text": "Neighboring ZCTAs show significant Zone A coverage",
        "field": "spatial_reasoning",
        "claim_type": "spatial",
    }
    result = verify_claim_programmatic(claim, SOURCE_ROW, EVIDENCE_TEXT)
    assert result["tier1_label"] == "unverifiable", \
        f"expected unverifiable, got {result['tier1_label']}: {result['tier1_reason']}"
    print(f"  PASS: spatial claim -> unverifiable")


# ---------------------------------------------------------------------------
# Test 7: SVI contradiction -> N_claim
# ---------------------------------------------------------------------------

def test_svi_contradiction_n_claim():
    """Claiming low vulnerability when SVI is 0.72 should be N_claim."""
    claim = {
        "text": "This area has low social vulnerability with minimal risk factors",
        "field": "vulnerability_factors",
        "claim_type": "categorical",
    }
    result = verify_claim_programmatic(claim, SOURCE_ROW, EVIDENCE_TEXT)
    assert result["tier1_label"] == "N_claim", \
        f"expected N_claim, got {result['tier1_label']}: {result['tier1_reason']}"
    print(f"  PASS: claims low vulnerability but SVI=0.72 -> N_claim")


# ---------------------------------------------------------------------------
# Test 8: Tier 1 metrics compute to hand-checkable values
# ---------------------------------------------------------------------------

def test_tier1_metrics_hand_check():
    """Verify metric formulas against known claim distributions."""
    claims = [
        {"tier1_label": "verified", "claim_type": "numeric", "text": "zone a 42.3%"},
        {"tier1_label": "verified", "claim_type": "citation", "text": "pop 45k"},
        {"tier1_label": "N_claim", "claim_type": "numeric", "text": "zone a 78%"},
        {"tier1_label": "N_claim", "claim_type": "citation", "text": "elevation 12ft"},
        {"tier1_label": "unverifiable", "claim_type": "spatial", "text": "neighbors flood"},
        {"tier1_label": "unverifiable", "claim_type": "generic", "text": "overall risk"},
    ]

    metrics = compute_tier1_metrics(claims)

    # Hand-computed:
    # n=6, verified=2, fabricated=2, unverifiable=2
    assert metrics["n_claims"] == 6
    assert metrics["n_verified"] == 2
    assert metrics["n_fabricated"] == 2
    assert metrics["n_unverifiable"] == 2

    # hallucination_rate = 2/6 = 0.3333
    assert abs(metrics["hallucination_rate"] - 0.3333) < 0.001, \
        f"hallucination_rate: expected ~0.3333, got {metrics['hallucination_rate']}"

    # verification_rate = 2/(2+2) = 0.5
    assert metrics["source_verification_rate"] == 0.5, \
        f"source_verification_rate: expected 0.5, got {metrics['source_verification_rate']}"
    # verifiable_rate = (2 verified + 2 fabricated) / 6 = 0.6667
    assert abs(metrics["verifiable_rate"] - 0.6667) < 0.001, \
        f"verifiable_rate: expected ~0.6667, got {metrics['verifiable_rate']}"

    # unsupported_evidence_rate = 1 N_claim citation / 2 total citations = 0.5
    assert metrics["unsupported_evidence_rate"] == 0.5, \
        f"unsupported_evidence_rate: expected 0.5, got {metrics['unsupported_evidence_rate']}"

    print(f"  PASS: Tier 1 metrics match hand computation")
    print(f"    hallucination_rate = {metrics['hallucination_rate']}")
    print(f"    verification_rate = {metrics['source_verification_rate']}")
    print(f"    unsupported_evidence_rate = {metrics['unsupported_evidence_rate']}")
    print(f"    evidence_coverage = {metrics['evidence_coverage']}")


# ---------------------------------------------------------------------------
# Test 9: Claim type classification
# ---------------------------------------------------------------------------

def test_claim_classification():
    """Verify claim type classifier on known inputs."""
    cases = [
        ("42.3% in Zone A", "numeric"),
        ("Population of 45,000", "numeric"),
        ("$52,000 median income", "numeric"),
        ("Zone A floodplain coverage is significant", "categorical"),
        ("SFHA designation applies", "categorical"),
        ("Neighboring areas show similar patterns", "spatial"),
        ("Adjacent ZCTAs have flood risk", "spatial"),
        ("Overall risk assessment suggests moderate concern", "generic"),
    ]
    for text, expected in cases:
        got = _classify_claim(text)
        assert got == expected, f"classify({text!r}): expected {expected}, got {got}"

    print(f"  PASS: all {len(cases)} claim classifications correct")


# ---------------------------------------------------------------------------
# Test 10: Number within tolerance -> verified
# ---------------------------------------------------------------------------

def test_numeric_within_tolerance():
    """A number within 15% tolerance should verify."""
    # 42.3 * 1.14 = 48.2 (within 15%)
    claim = {
        "text": "roughly 48% in Zone A, the 1% annual chance floodplain",
        "field": "zone_interpretation",
        "claim_type": "numeric",
    }
    result = verify_claim_programmatic(claim, SOURCE_ROW, EVIDENCE_TEXT)
    assert result["tier1_label"] == "verified", \
        f"expected verified (48 within 15% of 42.3), got {result['tier1_label']}"
    print(f"  PASS: 48% within 15% of 42.3% -> verified")


def test_numeric_outside_tolerance():
    """A number outside 15% tolerance should be N_claim."""
    # 42.3 * 1.5 = 63.45 (outside 15%)
    claim = {
        "text": "63% in Zone A, the 1% annual chance floodplain",
        "field": "zone_interpretation",
        "claim_type": "numeric",
    }
    result = verify_claim_programmatic(claim, SOURCE_ROW, EVIDENCE_TEXT)
    assert result["tier1_label"] == "N_claim", \
        f"expected N_claim (63 outside 15% of 42.3), got {result['tier1_label']}"
    print(f"  PASS: 63% outside 15% of 42.3% -> N_claim")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def test_low_svi_vulnerability_not_overclaim():
    """Regression: vulnerability mention at low SVI must not be flagged as 'very high' overclaim."""
    src = pd.Series({"zcta_id": "Z", "flood_pct_zone_a": 40.0, "svi_overall": 0.30})
    for txt in ["This area has moderate vulnerability",
                "Low vulnerability and minimal concern"]:
        c = {"text": txt, "field": "vulnerability_factors", "claim_type": "categorical"}
        r = verify_claim_programmatic(c, src, "SVI overall: 0.30")
        assert r["tier1_label"] != "N_claim", f"{txt!r} wrongly flagged: {r['tier1_reason']}"
    print("  PASS: low-SVI vulnerability mentions not mislabeled")


def main():
    print("=== score_vlm_quality.py fixture tests ===\n")

    tests = [
        ("Claim extraction", test_claim_extraction),
        ("Correct numeric -> verified", test_correct_numeric_verified),
        ("Fabricated numeric -> N_claim", test_fabricated_numeric_n_claim),
        ("Valid citation -> verified", test_valid_citation_verified),
        ("Fabricated citation -> N_claim", test_fabricated_citation_n_claim),
        ("Spatial claim -> unverifiable", test_spatial_claim_unverifiable),
        ("SVI contradiction -> N_claim", test_svi_contradiction_n_claim),
        ("Tier 1 metrics hand-check", test_tier1_metrics_hand_check),
        ("Claim type classification", test_claim_classification),
        ("Numeric within tolerance", test_numeric_within_tolerance),
        ("Numeric outside tolerance", test_numeric_outside_tolerance),
        ("Low-SVI vuln not overclaim (regression)", test_low_svi_vulnerability_not_overclaim),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"[{name}]")
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1
        print()

    print(f"=== {passed} passed, {failed} failed ===")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
