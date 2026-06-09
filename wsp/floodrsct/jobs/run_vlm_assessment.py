#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   generator: kimi (Moonshot AI, via OpenRouter)
#   cleanup_by: Martin
#   cleanup_summary: Remove unused imports (boto3, duplicate time), add
#       shebang/docstring/logging template, fix upload_json_result signature
#       (s3, BUCKET, key, payload), fix df.to_parquet (needs BytesIO not
#       bare call), share S3 client, add dry-run early exit, add new_orleans
#       to scenarios, add temperature=0.0 per DOE deterministic inference
#   see: ../exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml
# =============================================================================
"""run_vlm_assessment.py -- Phase R4.3: VLM flood risk assessment.

Sends map image + text evidence to a VLM per ZCTA and collects
structured JSON risk assessments. One run per (scenario, vlm).

Usage:
    python run_vlm_assessment.py --scenario houston --vlm gemini --upload
    python run_vlm_assessment.py --scenario houston --vlm gemini --dry-run
"""

import argparse
import io
import json
import logging
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client
from _s3_result import upload_json_result

from yrsn.adapters.outbound.vlm import (
    GPT4oVisionAdapter,
    GeminiVisionAdapter,
    JinaVLMAdapter,
    BedrockNovaVisionAdapter,
    Qwen2VLAdapter,
)

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
VLMS = ["gpt4o", "gemini", "jina", "nova", "qwen"]

VLM_ADAPTERS = {
    "gpt4o": GPT4oVisionAdapter,
    "gemini": GeminiVisionAdapter,
    "jina": JinaVLMAdapter,
    "nova": BedrockNovaVisionAdapter,
    "qwen": Qwen2VLAdapter,
}

# Concurrency config per VLM.
# workers: max parallel API calls (I/O-bound, threading is fine)
# min_interval: minimum seconds between dispatching calls (0 = no throttle)
#
# Jina paid tier supports 50 concurrent. Their VLM is slow (~15-50s/call)
# so concurrency is the only way to get throughput.
# Gemini free tier is 15 RPM = 4s interval, but with paid key we can push.
# Nova/GPT-4o are fast (<5s) so moderate concurrency suffices.
VLM_CONCURRENCY = {
    "gpt4o":  {"workers": 4, "min_interval": 1.5},  # reduced: 10/0.5 hit TPM ceiling
    "gemini": {"workers":  8, "min_interval": 2.0},
    "jina":   {"workers": 15, "min_interval": 0.0},  # slow API, pure concurrency
    "nova":   {"workers": 10, "min_interval": 0.0},   # Bedrock, fast
    "qwen":   {"workers":  8, "min_interval": 1.0},   # OpenRouter
}

# Fixed prompt across all VLMs (DOE R4 spec)
PROMPT = """\
You are assessing flood risk for a US Census ZCTA (ZIP Code Tabulation Area).

Given:
1. A map showing the ZCTA and surrounding area with FEMA flood zones
2. Text evidence about the ZCTA's demographics, infrastructure, and history

Produce a structured flood risk assessment as JSON:

{
  "risk_score": <float 0-1, overall flood risk>,
  "confidence": <float 0-1, your confidence in this assessment>,
  "zone_interpretation": "<what the flood zone map tells you>",
  "vulnerability_factors": ["<factor 1>", "<factor 2>"],
  "spatial_reasoning": "<how neighboring areas affect this ZCTA's risk>",
  "evidence_used": ["<specific visual/text elements referenced>"]
}

Be precise. Reference specific visual elements from the map and specific
numbers from the text. If you cannot determine something, say so."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_text_evidence(s3, scenario: str, zcta_id: str) -> str:
    """Load text evidence file from S3."""
    key = f"{RESULTS_PREFIX}/evidence/{scenario}/{zcta_id}.txt"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return resp["Body"].read().decode()


def _download_map_image(s3, scenario: str, zcta_id: str, tmp_dir: Path) -> Path:
    """Download map PNG from S3 to temp directory."""
    key = f"{RESULTS_PREFIX}/maps/{scenario}/{zcta_id}.png"
    local_path = tmp_dir / f"{zcta_id}.png"
    s3.download_file(BUCKET, key, str(local_path))
    return local_path


def _parse_vlm_response(raw: str) -> Dict[str, Any] | None:
    """Extract structured JSON from VLM response. Returns None on failure."""
    # Try direct JSON parse
    try:
        data = json.loads(raw)
        return {
            "risk_score": float(data.get("risk_score", 0.0)),
            "confidence": float(data.get("confidence", 0.0)),
            "zone_interpretation": str(data.get("zone_interpretation", "")),
            "vulnerability_factors": list(data.get("vulnerability_factors", [])),
            "spatial_reasoning": str(data.get("spatial_reasoning", "")),
            "evidence_used": list(data.get("evidence_used", [])),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try extracting JSON from markdown fences
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return {
                "risk_score": float(data.get("risk_score", 0.0)),
                "confidence": float(data.get("confidence", 0.0)),
                "zone_interpretation": str(data.get("zone_interpretation", "")),
                "vulnerability_factors": list(data.get("vulnerability_factors", [])),
                "spatial_reasoning": str(data.get("spatial_reasoning", "")),
                "evidence_used": list(data.get("evidence_used", [])),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return None


def _assess_one_zcta(
    s3, adapter, scenario: str, vlm_id: str, zcta_id: str,
    fold: int, tmp_dir: Path,
) -> Dict[str, Any]:
    """Run VLM assessment for one ZCTA. Returns a result record."""
    image_path = _download_map_image(s3, scenario, zcta_id, tmp_dir)
    evidence = _load_text_evidence(s3, scenario, zcta_id)
    full_prompt = PROMPT + "\n\nText Evidence:\n" + evidence

    max_retries = 5
    base_delay = 2.0  # seconds

    start = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            # DOE spec: temperature=0.0, greedy decoding
            resp = adapter.complete_with_reasoning(
                full_prompt,
                image_path=str(image_path),
                temperature=0.0,
                max_tokens=2048,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            parsed = _parse_vlm_response(resp["content"])

            return {
                "zcta_id": zcta_id,
                "vlm": vlm_id,
                "fold": fold,
                "risk_score": parsed["risk_score"] if parsed else None,
                "confidence": parsed["confidence"] if parsed else None,
                "parse_success": parsed is not None,
                "fixup_needed": False,
                "raw_response": resp["content"],
                "latency_ms": latency_ms,
                "prompt_tokens": resp["usage"].get("prompt_tokens", 0),
                "completion_tokens": resp["usage"].get("completion_tokens", 0),
            }
        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = "429" in exc_str or "rate_limit" in exc_str.lower()
            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                log.warning(
                    "Rate limit for %s/%s (attempt %d/%d), retrying in %.1fs",
                    vlm_id, zcta_id, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue

            latency_ms = int((time.monotonic() - start) * 1000)
            log.error("VLM error for %s/%s: %s", vlm_id, zcta_id, exc)
            return {
                "zcta_id": zcta_id,
                "vlm": vlm_id,
                "fold": fold,
                "risk_score": None,
                "confidence": None,
                "parse_success": False,
                "fixup_needed": False,
                "raw_response": exc_str,
                "latency_ms": latency_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase R4.3: VLM assessment")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--vlm", required=True, choices=VLMS)
    parser.add_argument("--model", default=None, help="Override VLM model name")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    scenario = args.scenario
    vlm_id = args.vlm
    # Tag for output filenames: vlm_id or vlm_id_modelslug when overridden
    if args.model:
        model_slug = args.model.replace("/", "_").replace(".", "_").replace("-", "_")
        vlm_tag = f"{vlm_id}_{model_slug}"
    else:
        vlm_tag = vlm_id

    if args.dry_run:
        log.info("DRY RUN: would assess ZCTAs for %s with %s", scenario, vlm_id)
        log.info("Reads: maps/{scenario}/*.png + evidence/{scenario}/*.txt")
        log.info("Writes: %s/r4_%s_%s.parquet", RESULTS_PREFIX, vlm_id, scenario)
        return 0

    s3 = get_s3_client()

    # Load fold assignments
    folds_key = f"folds/{scenario}_folds.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=folds_key)
        folds_df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
        # Folds assigned at ZCTA level -- deduplicate to one row per ZCTA
        # and use spatial_blocked folds to match R0-R2 DOE
        folds_dedup = folds_df.drop_duplicates("zcta_id")
        zcta_folds = dict(zip(folds_dedup["zcta_id"], folds_dedup["fold_spatial_blocked"]))
    except Exception as exc:
        log.warning("Could not load folds from %s: %s -- using fold=0", folds_key, exc)
        zcta_folds = {}

    # Discover available map PNGs for this scenario
    prefix = f"{RESULTS_PREFIX}/maps/{scenario}/"
    zcta_ids = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".png"):
                zcta_ids.append(Path(key).stem)

    log.info("Found %d map PNGs for %s", len(zcta_ids), scenario)
    if not zcta_ids:
        log.error("No maps found -- run render_zcta_maps.py first")
        return 1

    # Create adapter once
    adapter_kwargs = {"use_rsct_prompt": False}
    if args.model:
        adapter_kwargs["model"] = args.model
        log.info("Model override: %s", args.model)
    adapter = VLM_ADAPTERS[vlm_id](**adapter_kwargs)

    concurrency = VLM_CONCURRENCY.get(vlm_id, {"workers": 4, "min_interval": 1.0})
    workers = concurrency["workers"]
    min_interval = concurrency["min_interval"]
    log.info(
        "Concurrency: %d workers, %.1fs min_interval for %s",
        workers, min_interval, vlm_id,
    )

    # Token-bucket rate limiter: acquire a slot, sleep only the remaining
    # interval, then release. Does NOT hold the lock during the API call.
    _dispatch_lock = threading.Lock()
    _last_dispatch = [0.0]

    def _throttled_assess(zcta_id: str, fold: int, tmp_path: Path) -> Dict[str, Any]:
        if min_interval > 0:
            # Acquire slot: compute when we can dispatch, then release lock
            # and sleep OUTSIDE the lock so other threads aren't blocked.
            with _dispatch_lock:
                now = time.monotonic()
                earliest = _last_dispatch[0] + min_interval
                _last_dispatch[0] = max(now, earliest)
                my_dispatch = _last_dispatch[0]
            wait = my_dispatch - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        return _assess_one_zcta(s3, adapter, scenario, vlm_id, zcta_id, fold, tmp_path)

    records = []
    completed = 0
    total = len(zcta_ids)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _throttled_assess, zcta_id, zcta_folds.get(zcta_id, 0), tmp_path
                ): zcta_id
                for zcta_id in zcta_ids
            }
            for future in as_completed(futures):
                zcta_id = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    log.error("Worker exception for %s: %s", zcta_id, exc)
                    record = {
                        "zcta_id": zcta_id, "vlm": vlm_id,
                        "fold": zcta_folds.get(zcta_id, 0),
                        "risk_score": None, "confidence": None,
                        "parse_success": False, "fixup_needed": False,
                        "raw_response": str(exc), "latency_ms": 0,
                        "prompt_tokens": 0, "completion_tokens": 0,
                    }
                records.append(record)
                completed += 1

                parsed = "OK" if record["parse_success"] else "FAIL"
                score = record["risk_score"]
                if completed % 10 == 0 or completed == total:
                    log.info(
                        "  [%d/%d] %s parse=%s score=%s latency=%dms",
                        completed, total, zcta_id, parsed,
                        "%.2f" % score if score is not None else "N/A",
                        record["latency_ms"],
                    )

    df = pd.DataFrame(records)

    # Save parquet to S3
    parquet_key = f"{RESULTS_PREFIX}/r4_{vlm_tag}_{scenario}.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    if args.upload:
        s3.put_object(Bucket=BUCKET, Key=parquet_key, Body=buf.getvalue())
        log.info("Uploaded parquet to s3://%s/%s", BUCKET, parquet_key)

    # Save local copy
    out_dir = (
        Path(__file__).parent.parent
        / "exp" / "s035-model-ladder" / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    local_parquet = out_dir / f"r4_{vlm_tag}_{scenario}.parquet"
    df.to_parquet(local_parquet, index=False)
    log.info("Written to %s", local_parquet)

    # Summary
    n_total = len(df)
    n_parsed = int(df["parse_success"].sum())
    n_null = int(df["risk_score"].isna().sum())

    summary = {
        "phase": "R4.3_vlm_assessment",
        "scenario": scenario,
        "vlm": vlm_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_parse_failures": n_total - n_parsed,
        "n_null_scores": n_null,
        "parse_rate": round(n_parsed / n_total, 3) if n_total > 0 else 0.0,
    }

    if args.upload:
        summary_key = f"{RESULTS_PREFIX}/r4_{vlm_tag}_{scenario}_summary.json"
        upload_json_result(s3, BUCKET, summary_key, summary)
        log.info("Uploaded summary to s3://%s/%s", BUCKET, summary_key)

    log.info(
        "Done: %d ZCTAs, %d parsed (%.0f%%), %d null scores",
        n_total, n_parsed,
        100 * n_parsed / n_total if n_total else 0, n_null,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
