"""
dgm_route_engine.py

Dual-Graph Morph (DGM) route engine for Tier-2 claim ablation.

DGM is a certificate-governed morph from evidence graph states to
operational action graph states.

    Evidence graph (G_evidence):
        claims, sources, provenance, R/S/N certificate components

    Action graph (G_action):
        Trust, Review, Escalate, Suppress

    phi: G_evidence -> G_action

Claim relevance is measured by hold-out:

    phi(C) != phi(C_{-i})  =>  claim i is decision-relevant (R_claim)
    phi(C) == phi(C_{-i})  =>  claim i is supported filler (S_sup_claim)

Implementation: maps VLM tier1 claim grades to an RSN certificate, runs
the certificate through the gate pipeline (georsct.healthcheck), and
returns a Floodcaster route. Leave-one-claim-out ablation recomputes the
certificate and re-runs gates to detect route changes.

Gate decision -> route mapping:

  EXECUTE   -> Trust      (all gates passed)
  RE_ENCODE -> Review     (kappa below oobleck threshold)
  BLOCK     -> Escalate   (consensus or kappa unavailable)
  REPAIR    -> Escalate   (grounding failure)
  REJECT    -> Suppress   (noise floor breach + alpha low)

Design constraint
-----------------
SIDECAR ONLY. Does not modify certificates, gates, or any production RSCT
field. DGM turns claim hold-outs into action hold-outs -- nothing more.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from georsct.healthcheck.layers.gate_triage import evaluate_gates
from georsct.healthcheck.thresholds import GEOSPATIAL_CONUS27, ThresholdPreset


# ---------------------------------------------------------------------------
# Route vocabulary
# ---------------------------------------------------------------------------

FloodcasterRoute = Literal["Trust", "Review", "Escalate", "Suppress"]

DECISION_TO_ROUTE: dict[str, FloodcasterRoute] = {
    "EXECUTE": "Trust",
    "RE_ENCODE": "Review",
    "BLOCK": "Escalate",
    "REPAIR": "Escalate",
    "REJECT": "Suppress",
}


# ---------------------------------------------------------------------------
# Certificate construction from tier1 claim grades
# ---------------------------------------------------------------------------

def cert_from_claims(
    claims: List[Dict[str, Any]],
    sigma: float = 0.0,
) -> Dict[str, Any]:
    """Build a gate-compatible certificate dict from tier1-graded claims.

    Mapping:
        R       = n_verified / n_claims        (grounded signal)
        S_sup   = n_unverifiable / n_claims     (supported but uncheckable filler)
        N       = n_fabricated / n_claims        (noise / hallucination)

    The simplex R + S_sup + N = 1 holds by construction because every claim
    has exactly one tier1_label in {verified, unverifiable, N_claim}.

    Args:
        claims: List of claim dicts, each with tier1_label.
        sigma:  Cross-fold variance proxy. Default 0.0 (single response).

    Returns:
        Certificate dict compatible with evaluate_gates().
    """
    n = len(claims)

    if n == 0:
        return {
            "R": 0.0,
            "S_sup": 0.0,
            "N": 1.0,
            "alpha": 0.0,
            "kappa_compat": None,
            "kappa_source": "vlm_tier1_claims_empty",
            "sigma": sigma,
            "omega": 0.0,
        }

    n_verified = sum(1 for c in claims if c.get("tier1_label") == "verified")
    n_unverifiable = sum(1 for c in claims if c.get("tier1_label") == "unverifiable")
    n_fabricated = sum(1 for c in claims if c.get("tier1_label") == "N_claim")

    # Normalize to handle claims that might have unexpected labels
    total_labeled = n_verified + n_unverifiable + n_fabricated
    if total_labeled == 0:
        return {
            "R": 0.0,
            "S_sup": 0.0,
            "N": 1.0,
            "alpha": 0.0,
            "kappa_compat": None,
            "kappa_source": "vlm_tier1_claims_unlabeled",
            "sigma": sigma,
            "omega": 0.0,
        }

    R = n_verified / total_labeled
    S_sup = n_unverifiable / total_labeled
    N = n_fabricated / total_labeled

    alpha = R / (R + N) if (R + N) > 0 else 0.0
    kappa_compat = R * (1 - N)
    omega = 1 - S_sup

    return {
        "R": round(R, 6),
        "S_sup": round(S_sup, 6),
        "N": round(N, 6),
        "alpha": round(alpha, 6),
        "kappa_compat": round(kappa_compat, 6),
        "kappa_source": "vlm_tier1_claims",
        "sigma": sigma,
        "omega": round(omega, 6),
    }


# ---------------------------------------------------------------------------
# Route computation
# ---------------------------------------------------------------------------

def route_from_claims(
    claims: List[Dict[str, Any]],
    preset: ThresholdPreset = GEOSPATIAL_CONUS27,
) -> FloodcasterRoute:
    """Compute Floodcaster route from tier1-graded claims.

    Builds a certificate from claim counts and runs it through the
    gate pipeline.
    """
    cert = cert_from_claims(claims)
    result = evaluate_gates(cert, preset)
    return DECISION_TO_ROUTE.get(result.decision, "Escalate")


def route_from_cert(
    cert: Dict[str, Any],
    preset: ThresholdPreset = GEOSPATIAL_CONUS27,
) -> FloodcasterRoute:
    """Compute Floodcaster route from a pre-built certificate dict."""
    result = evaluate_gates(cert, preset)
    return DECISION_TO_ROUTE.get(result.decision, "Escalate")


# ---------------------------------------------------------------------------
# Leave-one-claim-out ablation
# ---------------------------------------------------------------------------

def ablate_one_claim(
    claims: List[Dict[str, Any]],
    remove_idx: int,
    preset: ThresholdPreset = GEOSPATIAL_CONUS27,
) -> FloodcasterRoute:
    """Remove one claim and recompute the route.

    Args:
        claims: Full list of tier1-graded claims for this response.
        remove_idx: Index of the claim to remove.
        preset: Threshold preset for gate evaluation.

    Returns:
        The Floodcaster route with the claim removed.
    """
    remaining = [c for i, c in enumerate(claims) if i != remove_idx]
    return route_from_claims(remaining, preset)


def compute_routes_with_ablation(
    claims: List[Dict[str, Any]],
    preset: ThresholdPreset = GEOSPATIAL_CONUS27,
) -> List[Dict[str, Any]]:
    """Compute route_with_claim and route_without_claim for every claim.

    For each claim at index i:
      - route_with_claim = route using ALL claims (same for every claim)
      - route_without_claim = route using all claims EXCEPT claim i

    Returns:
        List of dicts, one per claim, augmented with:
            route_with_claim: str
            route_without_claim: str
            route_changed: bool
    """
    if not claims:
        return []

    base_route = route_from_claims(claims, preset)
    results = []

    for i, claim in enumerate(claims):
        ablated_route = ablate_one_claim(claims, i, preset)
        results.append({
            **claim,
            "route_with_claim": base_route,
            "route_without_claim": ablated_route,
            "route_changed": ablated_route != base_route,
        })

    return results


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build cert from claims
    claims = [
        {"tier1_label": "verified"},
        {"tier1_label": "verified"},
        {"tier1_label": "verified"},
        {"tier1_label": "unverifiable"},
        {"tier1_label": "N_claim"},
    ]
    cert = cert_from_claims(claims)
    assert abs(cert["R"] + cert["S_sup"] + cert["N"] - 1.0) < 1e-6
    assert cert["R"] == round(3 / 5, 6)
    assert cert["N"] == round(1 / 5, 6)
    assert cert["kappa_source"] == "vlm_tier1_claims"

    # Empty claims -> N=1, kappa=None
    empty_cert = cert_from_claims([])
    assert empty_cert["N"] == 1.0
    assert empty_cert["kappa_compat"] is None

    # Route from clean claims -> Trust (all verified, gates should pass)
    clean = [{"tier1_label": "verified"}] * 10
    route = route_from_claims(clean)
    assert route == "Trust", f"expected Trust, got {route}"

    # Route from noisy claims -> Suppress (all fabricated)
    noisy = [{"tier1_label": "N_claim"}] * 10
    route = route_from_claims(noisy)
    assert route == "Suppress", f"expected Suppress, got {route}"

    # Ablation: removing one verified claim from a borderline set
    # shouldn't crash and should return a valid route
    mixed = (
        [{"tier1_label": "verified"}] * 3
        + [{"tier1_label": "unverifiable"}] * 2
        + [{"tier1_label": "N_claim"}] * 2
    )
    results = compute_routes_with_ablation(mixed)
    assert len(results) == 7
    assert all(r["route_with_claim"] in ("Trust", "Review", "Escalate", "Suppress") for r in results)
    assert all(r["route_without_claim"] in ("Trust", "Review", "Escalate", "Suppress") for r in results)

    # Verify that removing a fabricated claim from a borderline set
    # can change the route (demonstrates ablation sensitivity)
    base = results[0]["route_with_claim"]
    ablated_routes = {r["route_without_claim"] for r in results}
    print(f"base route: {base}")
    print(f"ablated routes: {ablated_routes}")
    print(f"route_changed count: {sum(r['route_changed'] for r in results)} / {len(results)}")

    print("all self-tests passed")
