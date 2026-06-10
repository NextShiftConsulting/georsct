#!/usr/bin/env python3
"""
run_tier2_quality.py -- Phase R4.4b: Tier-2 semantic relevance quorum.

Loads tier1 calibration samples, computes leave-one-claim-out route ablation
via the RSCT gate pipeline, runs independent Judge A + Judge B through the
quorum protocol, computes Krippendorff alpha, and emits sidecar-only
EXPLORATORY metrics.

Architecture:
  - Route engine: r4_route_engine.py (gate pipeline as Floodcaster route)
  - Quorum:       r4_tier2_relevance_quorum.py (INV-1 through INV-7)
  - Judges:       gemini-2.5-flash (A), gpt-4o (B), claude-sonnet (C)

Usage:
    python run_tier2_quality.py --scenario houston --vlm gpt4o --upload
    python run_tier2_quality.py --scenario houston --vlm gpt4o --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    from _coverage_common import BUCKET, get_s3_client
    from _s3_result import upload_json_result
    _RUNTIME_DEPS_AVAILABLE = True
except ImportError:
    BUCKET = None
    get_s3_client = None
    upload_json_result = None
    _RUNTIME_DEPS_AVAILABLE = False

from dgm_route_engine import compute_routes_with_ablation
from r4_tier2_relevance_quorum import (
    RUBRIC_VERSION,
    ClaimContext,
    Judge,
    JudgeOutput,
    ProductionRecord,
    ReliabilityRecord,
    Tier2Result,
    adjudicate_claim,
    apply_gate_to_production,
    calibration_gate,
    make_anthropic_judge,
    make_openai_like_judge,
    parse_judge_json,
    rubric_hash,
    summarize_tier2,
    tier1_eligible,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
SCENARIOS = ["houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"]
VLMS = ["gpt4o", "gemini_flash", "jina", "nova", "qwen"]
VLM_TAGS = {"gemini_flash": "gemini_gemini_3_5_flash"}

# Judge configuration
JUDGE_A_MODEL = "google/gemini-2.5-flash-preview"
JUDGE_A_FAMILY = "google"
JUDGE_B_MODEL = "openai/gpt-4o"
JUDGE_B_FAMILY = "openai"
JUDGE_C_MODEL = "anthropic/claude-sonnet-4-20250514"
JUDGE_C_FAMILY = "anthropic"

# Rate limiting between judge calls (seconds)
JUDGE_CALL_INTERVAL = 0.3


# ---------------------------------------------------------------------------
# Judge factory
# ---------------------------------------------------------------------------

def _make_openrouter_judge(model_id: str, family: str, api_key: str) -> Judge:
    """Build a judge using OpenRouter as the backend."""
    import openai

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    def complete_fn(prompt: str) -> JudgeOutput:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        text = response.choices[0].message.content
        return parse_judge_json(text)

    return Judge(model_id=model_id, family=family, complete_fn=complete_fn)


def build_judges(api_key: str) -> tuple[Judge, Judge, Judge]:
    """Build the three judges via OpenRouter."""
    judge_a = _make_openrouter_judge(JUDGE_A_MODEL, JUDGE_A_FAMILY, api_key)
    judge_b = _make_openrouter_judge(JUDGE_B_MODEL, JUDGE_B_FAMILY, api_key)
    judge_c = _make_openrouter_judge(JUDGE_C_MODEL, JUDGE_C_FAMILY, api_key)
    return judge_a, judge_b, judge_c


# ---------------------------------------------------------------------------
# Claim context builder
# ---------------------------------------------------------------------------

def build_claim_contexts(
    calibration_data: List[Dict],
    scenario: str,
    vlm_id: str,
) -> List[ClaimContext]:
    """Build ClaimContext objects from calibration sample + route ablation.

    For each ZCTA in the calibration sample:
      1. Filter to tier1-eligible claims (verified + unverifiable)
      2. Run leave-one-claim-out route ablation
      3. Build ClaimContext with route_with and route_without
    """
    contexts: List[ClaimContext] = []
    claim_counter = 0

    for entry in calibration_data:
        zcta_id = entry["zcta_id"]
        claims = entry.get("claims", [])

        if not claims:
            continue

        # Route ablation on ALL claims (not just eligible ones)
        # because removing a fabricated claim can also change the route
        ablated = compute_routes_with_ablation(claims)

        for i, claim_with_route in enumerate(ablated):
            tier1_label = claim_with_route.get("tier1_label", "")

            if not tier1_eligible(tier1_label):
                continue

            # Build source evidence string from claim context
            source_evidence = claim_with_route.get("tier1_reason", "")

            ctx = ClaimContext(
                claim_id=f"{zcta_id}_{claim_counter}",
                claim_text=claim_with_route.get("text", ""),
                tier1_label=tier1_label,
                zcta=zcta_id,
                scenario=scenario,
                vlm_model_id=vlm_id,
                source_evidence=source_evidence,
                route_with_claim=claim_with_route["route_with_claim"],
                route_without_claim=claim_with_route.get("route_without_claim"),
            )
            contexts.append(ctx)
            claim_counter += 1

    return contexts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tier2(
    s3,
    scenario: str,
    vlm_id: str,
    api_key: str,
    upload: bool = False,
) -> Dict[str, Any]:
    """Run Tier-2 semantic relevance quorum on calibration sample."""

    vlm_tag = VLM_TAGS.get(vlm_id, vlm_id)

    # Load calibration sample (produced by tier1 scoring)
    cal_key = f"{RESULTS_PREFIX}/r4_calibration_{vlm_tag}_{scenario}.json"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=cal_key)
        calibration = json.loads(resp["Body"].read().decode())
    except Exception as exc:
        log.error("Could not load calibration sample %s: %s", cal_key, exc)
        return {}

    calibration_data = calibration.get("samples", [])
    log.info(
        "Loaded calibration sample: %d ZCTAs from %s",
        len(calibration_data), cal_key,
    )

    # Build claim contexts with route ablation
    contexts = build_claim_contexts(calibration_data, scenario, vlm_id)
    log.info("Built %d tier2-eligible claim contexts", len(contexts))

    if not contexts:
        log.warning("No eligible claims for tier2 scoring")
        return {"n_claims": 0, "tier2_status": "EXPLORATORY"}

    # Build judges
    judge_a, judge_b, judge_c = build_judges(api_key)

    # Run quorum on each claim
    reliability_records: List[ReliabilityRecord] = []
    production_records: List[ProductionRecord] = []
    completed = 0

    for ctx in contexts:
        result = adjudicate_claim(ctx, judge_a, judge_b, judge_c)
        reliability_records.append(result.reliability)
        production_records.append(result.production)

        completed += 1
        if completed % 10 == 0 or completed == len(contexts):
            log.info("  judged %d / %d claims", completed, len(contexts))

        time.sleep(JUDGE_CALL_INTERVAL)

    # Calibration gate
    try:
        gate = calibration_gate(
            reliability_records,
            require_ablation=False,  # not all claims have ablation yet
            min_pairable=min(50, len(reliability_records)),
            bootstrap_iterations=500,
        )
    except ValueError as exc:
        log.error("Calibration gate error: %s", exc)
        gate = None

    if gate is not None:
        log.info(
            "Calibration gate: alpha=%.4f, bootstrap_lower=%.4f, "
            "n_pairable=%d, passed=%s",
            gate.alpha, gate.alpha_bootstrap_lower,
            gate.n_pairable, gate.passed,
        )
        production_records = apply_gate_to_production(production_records, gate)
        summary = summarize_tier2(production_records, gate)
    else:
        summary = None

    # Build output payload
    payload = {
        "experiment": "s035-model-ladder",
        "phase": f"tier2_quality_{vlm_tag}_{scenario}",
        "tier2_status": "EXPLORATORY",
        "rubric_version": RUBRIC_VERSION,
        "rubric_hash": rubric_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "vlm": vlm_id,
        "vlm_tag": vlm_tag,
        "n_claims_eligible": len(contexts),
        "n_claims_judged": completed,
        "judges": {
            "A": {"model": JUDGE_A_MODEL, "family": JUDGE_A_FAMILY},
            "B": {"model": JUDGE_B_MODEL, "family": JUDGE_B_FAMILY},
            "C": {"model": JUDGE_C_MODEL, "family": JUDGE_C_FAMILY},
        },
        "gate": {
            "alpha": gate.alpha if gate else None,
            "alpha_bootstrap_lower": gate.alpha_bootstrap_lower if gate else None,
            "n_pairable": gate.n_pairable if gate else 0,
            "passed": gate.passed if gate else False,
            "threshold": gate.threshold if gate else 0.67,
        },
        "summary": json.loads(summary.to_json()) if summary else None,
        "production_records": [
            json.loads(r.to_json()) for r in production_records
        ],
    }

    # Save locally
    out_dir = (
        Path(__file__).parent.parent
        / "exp" / "s035-model-ladder" / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    local_file = out_dir / f"r4_tier2_{vlm_tag}_{scenario}.json"
    with open(local_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("Written to %s", local_file)

    if upload and s3 is not None:
        s3_key = f"{RESULTS_PREFIX}/r4_tier2_{vlm_tag}_{scenario}.json"
        upload_json_result(s3, BUCKET, s3_key, payload)
        log.info("Uploaded to s3://%s/%s", BUCKET, s3_key)

    # Print summary
    log.info("\n=== R4.4b Tier-2 Relevance Quorum: %s / %s ===", vlm_id, scenario)
    log.info("  Status: EXPLORATORY")
    log.info("  Claims judged: %d", completed)
    if gate:
        log.info("  Alpha: %.4f (threshold: %.2f, passed: %s)",
                 gate.alpha, gate.threshold, gate.passed)
    if summary:
        log.info("  Grounded signal rate: %.4f", summary.grounded_signal_rate)
        log.info("  Filler rate: %.4f", summary.filler_rate)
        log.info("  Unresolved rate: %.4f", summary.unresolved_rate)
        log.info("  Label distribution: %s", summary.label_distribution)

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase R4.4b: Tier-2 semantic relevance quorum"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--vlm", required=True, choices=VLMS)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        vlm_tag = VLM_TAGS.get(args.vlm, args.vlm)
        log.info("DRY RUN: tier2 scoring %s/%s", args.vlm, args.scenario)
        log.info("  Reads: r4_calibration_%s_%s.json", vlm_tag, args.scenario)
        log.info("  Writes: r4_tier2_%s_%s.json", vlm_tag, args.scenario)
        log.info("  Judges: %s (A), %s (B), %s (C)",
                 JUDGE_A_MODEL, JUDGE_B_MODEL, JUDGE_C_MODEL)
        log.info("  Status: EXPLORATORY (INV-5)")
        log.info("  Rubric: %s (%s)", RUBRIC_VERSION, rubric_hash())
        return 0

    if not _RUNTIME_DEPS_AVAILABLE:
        log.error("Runtime deps unavailable; run inside the repo")
        return 1

    # Credentials via swarm_auth
    from swarm_auth import get_credential
    api_key = get_credential("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not available via swarm_auth")
        return 1

    s3 = get_s3_client()
    run_tier2(s3, args.scenario, args.vlm, api_key, args.upload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
