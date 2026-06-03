#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   generator: Martin (human-authored, DOE v1.3 spec)
#   see: ../exp/s035-model-ladder/DOE_R4_vlm.md (Response-Level Evidence-
#        Grounding Audit v1.3)
# =============================================================================
"""score_vlm_quality.py -- Phase R4.4: Response-level evidence-grounding audit.

Extracts atomic claims from VLM response text fields, grades each claim
against the source feature layers (NOT the PNG), and produces per-response
quality diagnostics.

Two tiers:
  Tier 1 (programmatic, trusted): number/zone/feature matching against
      source parquet columns. Produces hallucination_rate, verifiable_rate, source_verification_rate,
      evidence_coverage, unsupported_evidence_rate.
  Tier 2 (semantic, provisional): LLM-assisted R_claim/S_sup_claim split.
      Produces grounded_signal_rate, filler_rate. Provisional until
      inter-grader reliability (Krippendorff's alpha) clears threshold
      on human calibration sample.

This is a SIDECAR DIAGNOSTIC, not the RSCT certificate.

Usage:
    python score_vlm_quality.py --scenario houston --vlm gpt4o --upload
    python score_vlm_quality.py --scenario houston --vlm gpt4o --dry-run
"""

import argparse
import io
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
# Repo-local S3 deps are only needed at runtime (score_one_zcta/main), not for
# the pure grading functions. Wrap so the module imports and the fixture tests
# run anywhere (e.g. CI) without the repo present.
try:
    from _coverage_common import BUCKET, OUTPUT_KEYS, get_s3_client
    from _s3_result import upload_json_result
    _RUNTIME_DEPS_AVAILABLE = True
except ImportError:
    BUCKET = None
    OUTPUT_KEYS = {}
    get_s3_client = None
    upload_json_result = None
    _RUNTIME_DEPS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]
VLMS = ["gpt4o", "gemini_flash", "jina", "nova", "qwen"]

# VLM tag mapping (must match compute_vlm_comparison.py)
VLM_TAGS = {
    "gemini_flash": "gemini_gemini_3_5_flash",
}

# Source feature columns available for programmatic verification.
# These are the ground-truth columns from the event_features parquets.
VERIFIABLE_FEATURES = {
    # Flood zones (percentages)
    "flood_pct_zone_a": {"type": "pct", "aliases": [
        "zone a", "zone ae", "1% annual", "100-year", "sfha",
        "special flood hazard",
    ]},
    "flood_pct_zone_x": {"type": "pct", "aliases": [
        "zone x", "minimal flood", "minimal hazard",
    ]},
    "flood_pct_zone_x500": {"type": "pct", "aliases": [
        "zone x500", "x-500", "0.2% annual", "500-year",
        "moderate flood",
    ]},
    # Demographics
    "population": {"type": "int", "aliases": [
        "population", "residents", "people",
    ]},
    "acs_median_hh_income": {"type": "currency", "aliases": [
        "median income", "household income", "median household",
    ]},
    # SVI
    "svi_overall": {"type": "float", "aliases": [
        "svi", "social vulnerability", "vulnerability index",
    ]},
    # Historical events
    "obs_nfip_event_claims": {"type": "int", "aliases": [
        "nfip claims", "nfip", "insurance claims", "flood claims",
    ]},
    "obs_has_311": {"type": "binary", "aliases": [
        "311", "flood reports", "service request",
    ]},
}

# Tolerance for numeric matching (relative)
NUMBER_TOLERANCE = 0.15  # 15% relative tolerance for number verification


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def extract_claims(parsed_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract atomic claims from VLM response text fields.

    Returns a list of claim dicts:
        {"text": str, "field": str, "claim_type": str}

    claim_type: numeric, categorical, spatial, citation, generic
    """
    claims = []

    # zone_interpretation: usually 1-3 sentences about flood zones
    zone_text = parsed_response.get("zone_interpretation", "")
    if zone_text:
        for sent in _split_sentences(zone_text):
            claims.append({
                "text": sent.strip(),
                "field": "zone_interpretation",
                "claim_type": _classify_claim(sent),
            })

    # vulnerability_factors: list of strings
    factors = parsed_response.get("vulnerability_factors", [])
    if isinstance(factors, list):
        for f in factors:
            if isinstance(f, str) and f.strip():
                claims.append({
                    "text": f.strip(),
                    "field": "vulnerability_factors",
                    "claim_type": _classify_claim(f),
                })

    # spatial_reasoning: text about neighbors
    spatial_text = parsed_response.get("spatial_reasoning", "")
    if spatial_text:
        for sent in _split_sentences(spatial_text):
            claims.append({
                "text": sent.strip(),
                "field": "spatial_reasoning",
                "claim_type": _classify_claim(sent),
            })

    # evidence_used: list of citations
    evidence = parsed_response.get("evidence_used", [])
    if isinstance(evidence, list):
        for e in evidence:
            if isinstance(e, str) and e.strip():
                claims.append({
                    "text": e.strip(),
                    "field": "evidence_used",
                    "claim_type": "citation",
                })

    return claims


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. Simple rule-based splitter."""
    # Split on period/semicolon followed by space or end
    parts = re.split(r'(?<=[.;])\s+', text)
    # Also split on explicit conjunctions that introduce new claims
    result = []
    for p in parts:
        if len(p) > 150:
            # Long segment -- try splitting on commas that introduce clauses
            sub = re.split(r',\s+(?:which|where|while|and|suggesting|indicating)', p)
            result.extend(s.strip() for s in sub if s.strip())
        elif p.strip():
            result.append(p.strip())
    return result


def _classify_claim(text: str) -> str:
    """Classify a claim as numeric, categorical, spatial, or generic."""
    t = text.lower()

    # Numeric: contains a number with context
    if re.search(r'\d+\.?\d*\s*%', t):
        return "numeric"
    if re.search(r'\$[\d,]+', t):
        return "numeric"
    if re.search(r'(?:population|income|claims|score)\s*(?:of|is|:)?\s*[\d,]+', t):
        return "numeric"

    # Categorical: mentions flood zones, SVI levels, risk categories
    zone_terms = ["zone a", "zone x", "zone ae", "zone ve", "sfha",
                  "floodplain", "flood zone", "minimal hazard",
                  "high hazard", "coastal"]
    if any(term in t for term in zone_terms):
        return "categorical"

    # Spatial: mentions neighbors, surrounding, adjacent, proximity
    spatial_terms = ["neighbor", "adjacent", "surrounding", "nearby",
                     "proximity", "border", "contiguous", "next to"]
    if any(term in t for term in spatial_terms):
        return "spatial"

    return "generic"


# ---------------------------------------------------------------------------
# Tier 1: Programmatic verification (trusted)
# ---------------------------------------------------------------------------

def verify_claim_programmatic(
    claim: Dict[str, Any],
    source_row: pd.Series,
    evidence_text: str,
) -> Dict[str, Any]:
    """Grade a claim programmatically against source features.

    Returns the claim dict augmented with:
        tier1_label: "N_claim" | "verified" | "unverifiable"
        tier1_reason: str
    """
    text = claim["text"].lower()
    ctype = claim["claim_type"]

    # -- Numeric claims: extract numbers and match against source --
    if ctype == "numeric":
        result = _verify_numeric(text, source_row)
        if result is not None:
            claim["tier1_label"] = result[0]
            claim["tier1_reason"] = result[1]
            return claim

    # -- Categorical claims: verify zone names exist in source --
    if ctype == "categorical":
        result = _verify_categorical(text, source_row)
        if result is not None:
            claim["tier1_label"] = result[0]
            claim["tier1_reason"] = result[1]
            return claim

    # -- Citation claims: verify cited evidence exists in input --
    # Tier 1 judges EVIDENCE PRESENCE only. Whether the inference drawn from
    # that evidence is supported (e.g. "population 45,000 proves high risk")
    # is a relevance judgement reserved for Tier 2. We record both so the
    # distinction is never silently collapsed.
    if ctype == "citation":
        result = _verify_citation(text, evidence_text, source_row)
        claim["tier1_label"] = result[0]
        claim["tier1_reason"] = result[1]
        claim["evidence_present"] = (result[0] == "verified")
        claim["inference_status"] = "not_evaluated"  # Tier 2
        return claim

    # -- Spatial and generic: cannot verify programmatically --
    claim["tier1_label"] = "unverifiable"
    claim["tier1_reason"] = "requires semantic grading"
    return claim


def _verify_numeric(text: str, source_row: pd.Series) -> Optional[Tuple[str, str]]:
    """Check if numbers in a claim match source feature values."""
    # Extract all numbers from the claim
    numbers = []
    # Percentages -- skip boilerplate zone-definition percentages (1%, 0.2%)
    # that describe what the zone IS, not a data value about this ZCTA
    _ZONE_BOILERPLATE_PCT = {1.0, 0.2}
    for m in re.finditer(r'(\d+\.?\d*)\s*%', text):
        val = float(m.group(1))
        if val in _ZONE_BOILERPLATE_PCT:
            # Check if this is zone definition context ("1% annual", "0.2% annual")
            context = text[max(0, m.start() - 5):m.end() + 15]
            if "annual" in context or "chance" in context:
                continue
        numbers.append(("pct", val))
    # Dollar amounts
    for m in re.finditer(r'\$\s*([\d,]+\.?\d*)', text):
        val = float(m.group(1).replace(",", ""))
        numbers.append(("currency", val))
    # Plain integers in context
    for m in re.finditer(r'(?:population|claims|score)[:\s]+(?:of\s+)?([\d,]+)', text):
        val = float(m.group(1).replace(",", ""))
        numbers.append(("int", val))

    if not numbers:
        return None  # No extractable numbers

    # Try to match each number against source features
    matched = 0
    fabricated = 0
    for num_type, num_val in numbers:
        found_match = False
        for col, meta in VERIFIABLE_FEATURES.items():
            # Check if the claim text mentions this feature
            if not any(alias in text for alias in meta["aliases"]):
                continue
            source_val = source_row.get(col)
            if pd.isna(source_val):
                continue
            source_val = float(source_val)
            # Compare with tolerance
            if source_val == 0:
                if num_val == 0:
                    found_match = True
                    break
            elif abs(num_val - source_val) / max(abs(source_val), 1e-6) <= NUMBER_TOLERANCE:
                found_match = True
                break
        if found_match:
            matched += 1
        else:
            fabricated += 1

    if fabricated > 0 and matched == 0:
        return ("N_claim", "numeric value not found in source features")
    if fabricated > 0:
        return ("N_claim", "some numeric values fabricated (mixed)")
    if matched > 0:
        return ("verified", "numeric values match source features")
    return None


def _verify_categorical(text: str, source_row: pd.Series) -> Optional[Tuple[str, str]]:
    """Check if categorical claims (zone names, SVI levels) are consistent."""
    # Check flood zone claims against actual zone percentages
    zone_claims = []
    if "zone a" in text or "zone ae" in text or "1%" in text:
        val = source_row.get("flood_pct_zone_a", 0)
        if not pd.isna(val):
            zone_claims.append(("zone_a", float(val)))
    if "zone x" in text and "x500" not in text and "x-500" not in text:
        val = source_row.get("flood_pct_zone_x", 0)
        if not pd.isna(val):
            zone_claims.append(("zone_x", float(val)))

    # Check for fabricated zones (VE, D, etc. that may not exist)
    fabricated_zones = []
    if "zone ve" in text or "coastal high hazard" in text:
        # VE only exists in coastal scenarios -- check if source has it
        if source_row.get("flood_pct_zone_a", 0) == 0 and source_row.get("flood_pct_zone_x500", 0) == 0:
            fabricated_zones.append("zone_ve")

    if fabricated_zones:
        return ("N_claim", f"references non-existent zone: {fabricated_zones}")

    # SVI interpretation check
    svi_val = source_row.get("svi_overall")
    if not pd.isna(svi_val) and svi_val is not None:
        svi_val = float(svi_val)
        if "low" in text and "vulnerability" in text and svi_val >= 0.5:
            return ("N_claim", f"claims low vulnerability but SVI={svi_val:.2f}")
        if ("very high" in text or "extremely" in text) and "vulnerability" in text and svi_val < 0.5:
            return ("N_claim", f"claims very high vulnerability but SVI={svi_val:.2f}")

    if zone_claims:
        return ("verified", "zone references consistent with source")
    return None


def _verify_citation(
    text: str, evidence_text: str, source_row: pd.Series,
) -> Tuple[str, str]:
    """Check if a cited evidence element actually exists in the input."""
    t = text.lower().strip()

    # Check if the citation references something from the evidence text
    evidence_lower = evidence_text.lower() if evidence_text else ""

    # Extract key phrases from the citation
    # Common patterns: "X% in Zone A", "population of N", "SVI score of X"
    numbers_in_citation = re.findall(r'[\d,]+\.?\d*', t)

    # Check if any numbers from citation appear in evidence text
    if numbers_in_citation:
        for num_str in numbers_in_citation:
            if num_str in evidence_lower:
                return ("verified", "cited evidence present in input (inference not evaluated -- Tier 2)")

    # Check if key terms from citation appear in evidence
    key_terms = ["zone a", "zone x", "zone x500", "population", "income",
                 "svi", "nfip", "311", "flood", "claims"]
    term_match = any(term in t and term in evidence_lower for term in key_terms)
    if term_match:
        return ("verified", "cited term present in input (inference not evaluated -- Tier 2)")

    # Check for map-only references (visual claims)
    visual_terms = ["map", "image", "color", "blue", "yellow", "red",
                    "gray", "legend", "choropleth", "shading"]
    if any(vt in t for vt in visual_terms):
        return ("unverifiable", "visual claim -- requires map adjudication")

    return ("N_claim", "cited evidence not found in input text or source")


# ---------------------------------------------------------------------------
# Tier 1 aggregation metrics
# ---------------------------------------------------------------------------

def compute_tier1_metrics(graded_claims: List[Dict]) -> Dict[str, Any]:
    """Compute Tier 1 (programmatic, trusted) metrics from graded claims.

    Returns:
        n_claims: total atomic claims extracted
        n_verified: claims with matching source data
        n_fabricated: claims flagged N_claim by programmatic check
        n_unverifiable: claims requiring semantic grading
        hallucination_rate: n_fabricated / n_claims
        source_verification_rate: n_verified / (n_verified + n_fabricated)
            NOTE: This is NOT claim_purity (DOE's R/(R+N)). "verified"
            means "the number/zone checked out against source," not
            "grounded AND decision-relevant." A verified claim can still
            be filler (S_sup_claim). Real claim_purity requires the
            R_claim/S_sup_claim split, which lives in Tier 2.
        evidence_coverage: fraction of important features referenced
        unsupported_evidence_rate: fraction of citations not in input
    """
    if not graded_claims:
        return {
            "n_claims": 0,
            "n_verified": 0,
            "n_fabricated": 0,
            "n_unverifiable": 0,
            "hallucination_rate": None,
            "verifiable_rate": None,
            "source_verification_rate": None,
            "evidence_coverage": None,
            "unsupported_evidence_rate": None,
        }

    n = len(graded_claims)
    n_verified = sum(1 for c in graded_claims if c.get("tier1_label") == "verified")
    n_fabricated = sum(1 for c in graded_claims if c.get("tier1_label") == "N_claim")
    n_unverifiable = sum(1 for c in graded_claims if c.get("tier1_label") == "unverifiable")

    hallucination_rate = round(n_fabricated / n, 4) if n > 0 else None
    # verifiable_rate distinguishes a grounded explanation from a vague one:
    # a low hallucination_rate is only meaningful if much of the text was
    # actually checkable. low hallucination + low verifiable = vague filler.
    verifiable_rate = round((n_verified + n_fabricated) / n, 4) if n > 0 else None
    denom = n_verified + n_fabricated
    # NOT claim_purity (DOE's R/(R+N)): "verified" means source-supported, which
    # is necessary but not sufficient for decision-relevant (a verified claim can
    # still be filler, S_sup). Hence the explicit name.
    source_verification_rate = round(n_verified / denom, 4) if denom > 0 else None

    # Evidence coverage: which important features did the response mention?
    all_text = " ".join(c["text"].lower() for c in graded_claims)
    important_features = [
        "flood_pct_zone_a", "flood_pct_zone_x", "flood_pct_zone_x500",
        "population", "acs_median_hh_income", "svi_overall",
        "obs_nfip_event_claims",
    ]
    features_referenced = 0
    for feat in important_features:
        meta = VERIFIABLE_FEATURES.get(feat, {})
        aliases = meta.get("aliases", [])
        if any(alias in all_text for alias in aliases):
            features_referenced += 1
    evidence_coverage = round(features_referenced / len(important_features), 4)

    # Unsupported evidence rate: citations flagged as N_claim
    citations = [c for c in graded_claims if c.get("claim_type") == "citation"]
    n_citations = len(citations)
    n_unsupported = sum(1 for c in citations if c.get("tier1_label") == "N_claim")
    unsupported_evidence_rate = round(n_unsupported / n_citations, 4) if n_citations > 0 else None

    return {
        "n_claims": n,
        "n_verified": n_verified,
        "n_fabricated": n_fabricated,
        "n_unverifiable": n_unverifiable,
        "hallucination_rate": hallucination_rate,
        "verifiable_rate": verifiable_rate,
        "source_verification_rate": source_verification_rate,
        "evidence_coverage": evidence_coverage,
        "unsupported_evidence_rate": unsupported_evidence_rate,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str) -> Optional[pd.DataFrame]:
    """Load parquet from S3, return None on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def _load_text_evidence(s3, scenario: str, zcta_id: str) -> str:
    """Load text evidence from S3. Returns empty string on failure."""
    key = f"{RESULTS_PREFIX}/evidence/{scenario}/{zcta_id}.txt"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return resp["Body"].read().decode()
    except Exception:
        return ""


def _parse_raw_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the raw VLM response JSON. Returns None on failure."""
    if not raw or not isinstance(raw, str):
        return None
    # Direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    # Markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Per-ZCTA scoring
# ---------------------------------------------------------------------------

def score_one_zcta(
    s3,
    zcta_id: str,
    vlm_row: pd.Series,
    source_row: pd.Series,
    scenario: str,
) -> Dict[str, Any]:
    """Score one VLM response against source features.

    Returns a flat dict of quality metrics for this ZCTA.
    """
    result = {
        "zcta_id": zcta_id,
        "vlm": vlm_row.get("vlm", "unknown"),
        "parse_success": bool(vlm_row.get("parse_success", False)),
    }

    # If parse failed, no claims to grade
    if not result["parse_success"]:
        result.update({
            "n_claims": 0,
            "n_verified": 0,
            "n_fabricated": 0,
            "n_unverifiable": 0,
            "hallucination_rate": None,
            "verifiable_rate": None,
            "source_verification_rate": None,
            "evidence_coverage": None,
            "unsupported_evidence_rate": None,
            "claim_types": {},
        })
        return result

    # Parse the raw response
    parsed = _parse_raw_response(vlm_row.get("raw_response", ""))
    if parsed is None:
        result["parse_success"] = False
        result.update({
            "n_claims": 0,
            "n_verified": 0,
            "n_fabricated": 0,
            "n_unverifiable": 0,
            "hallucination_rate": None,
            "verifiable_rate": None,
            "source_verification_rate": None,
            "evidence_coverage": None,
            "unsupported_evidence_rate": None,
            "claim_types": {},
        })
        return result

    # Load evidence text (what the VLM was shown)
    evidence_text = _load_text_evidence(s3, scenario, zcta_id)

    # Step 1: Extract atomic claims
    claims = extract_claims(parsed)

    # Step 2: Tier 1 programmatic grading
    for claim in claims:
        verify_claim_programmatic(claim, source_row, evidence_text)

    # Step 3: Aggregate Tier 1 metrics
    metrics = compute_tier1_metrics(claims)
    result.update(metrics)

    # Claim type distribution
    type_counts = {}
    for c in claims:
        ct = c.get("claim_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    result["claim_types"] = type_counts

    # Store individual claim grades for calibration sample
    result["_claims_detail"] = claims

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase R4.4: Response-level evidence-grounding audit"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--vlm", required=True, choices=VLMS)
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument(
        "--calibration-sample", type=int, default=20,
        help="Number of ZCTAs to flag for human calibration (default: 20)"
    )
    args = parser.parse_args()

    scenario = args.scenario
    vlm_id = args.vlm
    vlm_tag = VLM_TAGS.get(vlm_id, vlm_id)

    if args.dry_run:
        log.info("DRY RUN: would score %s/%s", vlm_id, scenario)
        log.info("Reads: r4_%s_%s.parquet + %s + evidence texts",
                 vlm_tag, scenario, OUTPUT_KEYS[scenario])
        log.info("Writes: %s/r4_quality_%s_%s.parquet", RESULTS_PREFIX, vlm_tag, scenario)
        log.info("Tier 1 (programmatic): hallucination_rate, verifiable_rate, source_verification_rate, evidence_coverage")
        log.info("Tier 2 (semantic): NOT YET IMPLEMENTED -- requires LLM grader + calibration")
        log.info("Calibration sample: %d ZCTAs flagged for human review", args.calibration_sample)
        return 0

    s3 = get_s3_client() if _RUNTIME_DEPS_AVAILABLE else None
    if s3 is None:
        log.error("Runtime S3 deps (_coverage_common/_s3_result) unavailable; "
                  "run inside the repo. Pure grading functions are importable "
                  "and testable without them.")
        return 1

    # Load VLM results
    vlm_key = f"{RESULTS_PREFIX}/r4_{vlm_tag}_{scenario}.parquet"
    vlm_df = _load_parquet(s3, vlm_key)
    if vlm_df is None:
        log.error("VLM results not found: %s", vlm_key)
        return 1
    vlm_df["zcta_id"] = vlm_df["zcta_id"].astype(str)
    log.info("Loaded %d VLM responses from %s", len(vlm_df), vlm_key)

    # Load source features (ground truth)
    source_df = _load_parquet(s3, OUTPUT_KEYS[scenario])
    if source_df is None:
        log.error("Source features not found: %s", OUTPUT_KEYS[scenario])
        return 1
    source_df["zcta_id"] = source_df["zcta_id"].astype(str)
    source_df = source_df.drop_duplicates("zcta_id").set_index("zcta_id")
    log.info("Loaded %d source feature rows", len(source_df))

    # Score each ZCTA
    records = []
    claims_for_calibration = []
    completed = 0
    total = len(vlm_df)

    for _, vlm_row in vlm_df.iterrows():
        zcta_id = str(vlm_row["zcta_id"])

        # Get source row
        if zcta_id not in source_df.index:
            log.warning("ZCTA %s not in source features, skipping", zcta_id)
            continue

        source_row = source_df.loc[zcta_id]
        result = score_one_zcta(s3, zcta_id, vlm_row, source_row, scenario)

        # Extract claims detail for calibration, then remove from record
        claims_detail = result.pop("_claims_detail", [])
        if claims_detail:
            claims_for_calibration.append({
                "zcta_id": zcta_id,
                "claims": claims_detail,
            })

        records.append(result)
        completed += 1
        if completed % 50 == 0 or completed == total:
            log.info("  scored %d / %d", completed, total)

    df = pd.DataFrame(records)

    # Flatten claim_types dict into columns
    if "claim_types" in df.columns:
        type_df = pd.json_normalize(df["claim_types"].fillna({}))
        type_df.columns = [f"n_claims_{c}" for c in type_df.columns]
        df = pd.concat([df.drop(columns=["claim_types"]), type_df], axis=1)

    # Save parquet
    out_dir = (
        Path(__file__).parent.parent
        / "exp" / "s035-model-ladder" / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    local_parquet = out_dir / f"r4_quality_{vlm_tag}_{scenario}.parquet"
    df.to_parquet(local_parquet, index=False)
    log.info("Written to %s", local_parquet)

    if args.upload:
        parquet_key = f"{RESULTS_PREFIX}/r4_quality_{vlm_tag}_{scenario}.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=parquet_key, Body=buf.getvalue())
        log.info("Uploaded parquet to s3://%s/%s", BUCKET, parquet_key)

    # Select calibration sample (stratified: mix of high/low hallucination)
    calibration_ids = _select_calibration_sample(
        df, claims_for_calibration, args.calibration_sample,
    )

    # Build calibration sample JSON (for human review)
    calibration_data = []
    for entry in claims_for_calibration:
        if entry["zcta_id"] in calibration_ids:
            calibration_data.append(entry)

    calibration_file = out_dir / f"r4_calibration_{vlm_tag}_{scenario}.json"
    with open(calibration_file, "w") as f:
        json.dump({
            "vlm": vlm_id,
            "scenario": scenario,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(calibration_data),
            "pre_registered_rubric": {
                "R_claim": "Supported by source features AND relevant to flood risk decision",
                "S_sup_claim": "True/supported but not specific or actionable for this ZCTA",
                "N_claim": "Unsupported, contradicted, fabricated, or citing non-existent evidence",
            },
            "calibration_scope": {
                "labels": "Grade each claim R_claim / S_sup_claim / N_claim",
                "extraction": (
                    "Also validate claim extraction: did the extractor "
                    "pull the right atomic claims from the response? "
                    "Mark any missed claims or spurious splits. Every "
                    "downstream rate inherits extraction error."
                ),
            },
            "pre_registered_parameters": {
                "numeric_tolerance_pct": 15,
                "numeric_tolerance_note": (
                    "Numbers within 15% relative difference of source "
                    "value count as matching. Chosen before seeing results."
                ),
                "important_features_for_coverage": [
                    "flood_pct_zone_a", "flood_pct_zone_x",
                    "flood_pct_zone_x500", "population",
                    "acs_median_hh_income", "svi_overall",
                    "obs_nfip_event_claims",
                ],
            },
            "reliability_threshold": {
                "metric": "krippendorff_alpha",
                "minimum": 0.67,
                "target": 0.80,
                "note": "Below 0.67 = Tier 2 metrics are unreliable, do not report",
            },
            "samples": calibration_data,
        }, f, indent=2)
    log.info("Calibration sample (%d ZCTAs) written to %s",
             len(calibration_data), calibration_file)

    if args.upload:
        cal_key = f"{RESULTS_PREFIX}/r4_calibration_{vlm_tag}_{scenario}.json"
        with open(calibration_file, "rb") as f:
            s3.put_object(Bucket=BUCKET, Key=cal_key, Body=f.read(),
                          ContentType="application/json")
        log.info("Uploaded calibration to s3://%s/%s", BUCKET, cal_key)

    # Summary
    n_scored = len(df[df["n_claims"] > 0])
    mean_claims = df.loc[df["n_claims"] > 0, "n_claims"].mean() if n_scored > 0 else 0
    mean_halluc = df.loc[df["hallucination_rate"].notna(), "hallucination_rate"].mean()
    mean_verif = df.loc[df["source_verification_rate"].notna(), "source_verification_rate"].mean()
    mean_coverage = df.loc[df["evidence_coverage"].notna(), "evidence_coverage"].mean()

    summary = {
        "phase": "R4.4_evidence_grounding_audit",
        "scenario": scenario,
        "vlm": vlm_id,
        "vlm_tag": vlm_tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": "tier1_programmatic",
        "tier2_status": "NOT_IMPLEMENTED",
        "n_responses": len(df),
        "n_scored": n_scored,
        "n_parse_failures": len(df) - n_scored,
        "mean_claims_per_response": round(float(mean_claims), 1) if n_scored > 0 else 0,
        "tier1_metrics": {
            "mean_hallucination_rate": round(float(mean_halluc), 4) if pd.notna(mean_halluc) else None,
            "mean_source_verification_rate": round(float(mean_verif), 4) if pd.notna(mean_verif) else None,
            "mean_evidence_coverage": round(float(mean_coverage), 4) if pd.notna(mean_coverage) else None,
        },
        "calibration": {
            "n_samples": len(calibration_data),
            "zcta_ids": calibration_ids,
            "rubric_pre_registered": True,
            "reliability_metric": "krippendorff_alpha",
            "reliability_threshold": 0.67,
        },
    }

    summary_file = out_dir / f"r4_quality_{vlm_tag}_{scenario}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary written to %s", summary_file)

    if args.upload:
        summary_key = f"{RESULTS_PREFIX}/r4_quality_{vlm_tag}_{scenario}_summary.json"
        upload_json_result(s3, BUCKET, summary_key, summary)
        log.info("Uploaded summary to s3://%s/%s", BUCKET, summary_key)

    log.info("\n=== R4.4 Evidence-Grounding Audit: %s / %s ===", vlm_id, scenario)
    log.info("  Tier 1 (programmatic, TRUSTED):")
    log.info("    Responses scored: %d / %d", n_scored, len(df))
    log.info("    Mean claims/response: %.1f", mean_claims if n_scored else 0)
    log.info("    Mean hallucination_rate: %s",
             "%.4f" % mean_halluc if pd.notna(mean_halluc) else "N/A")
    log.info("    Mean source_verification_rate: %s",
             "%.4f" % mean_verif if pd.notna(mean_verif) else "N/A")
    log.info("    Mean evidence_coverage: %s",
             "%.4f" % mean_coverage if pd.notna(mean_coverage) else "N/A")
    log.info("  Tier 2 (semantic): NOT YET IMPLEMENTED")
    log.info("    Calibration sample: %d ZCTAs flagged for human review", len(calibration_data))
    log.info("    Krippendorff alpha threshold: >= 0.67 (target 0.80)")

    return 0


def _select_calibration_sample(
    df: pd.DataFrame,
    claims_data: List[Dict],
    n_samples: int,
) -> List[str]:
    """Select a stratified calibration sample.

    Strategy: pick from low/medium/high hallucination strata to ensure
    the calibration sample covers the full range of response quality.
    """
    if len(df) == 0 or n_samples == 0:
        return []

    scored = df[df["n_claims"] > 0].copy()
    if len(scored) == 0:
        return []

    n_samples = min(n_samples, len(scored))
    zcta_ids_with_claims = {e["zcta_id"] for e in claims_data if e.get("claims")}
    scored = scored[scored["zcta_id"].isin(zcta_ids_with_claims)]

    if len(scored) == 0:
        return []

    n_samples = min(n_samples, len(scored))

    # Stratify by hallucination_rate terciles
    scored = scored.sort_values("hallucination_rate", na_position="last")
    n_per_stratum = max(1, n_samples // 3)

    low = scored.head(len(scored) // 3)
    mid = scored.iloc[len(scored) // 3: 2 * len(scored) // 3]
    high = scored.tail(len(scored) // 3)

    sample_ids = []
    for stratum in [low, mid, high]:
        if len(stratum) > 0:
            sampled = stratum.sample(n=min(n_per_stratum, len(stratum)), random_state=42)
            sample_ids.extend(sampled["zcta_id"].tolist())

    # Fill remaining slots from any stratum
    remaining = n_samples - len(sample_ids)
    if remaining > 0:
        unsampled = scored[~scored["zcta_id"].isin(sample_ids)]
        if len(unsampled) > 0:
            extra = unsampled.sample(n=min(remaining, len(unsampled)), random_state=42)
            sample_ids.extend(extra["zcta_id"].tolist())

    return sample_ids[:n_samples]


if __name__ == "__main__":
    sys.exit(main())
