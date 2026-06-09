"""Evolver module for R5 harness evolution.

Builds prompts for the evolver model and parses patch responses.
The evolver receives a failure report and the current harness, then
proposes an allowlisted patch.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .harness_schema import HarnessPatch, HarnessVersion, PatchOperation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evolver prompt template
# ---------------------------------------------------------------------------

EVOLVER_SYSTEM_PROMPT = """\
You are a harness evolution specialist for geospatial flood risk assessment.

Your role: analyze failure patterns from a frozen VLM's flood risk assessments
and propose targeted edits to the evidence template, feature policy, rubric,
or scenario memory. The VLM itself cannot change -- only the harness around it.

RULES:
1. You may ONLY modify: evidence_template, feature_policy, rubric, scenario_memory
2. You may NOT modify: map renderer, legend, color scale, image size, scoring code
3. You may NOT reference specific held-out ZCTAs or their labels
4. You may NOT memorize specific ZCTA -> risk mappings
5. You may NOT weaken schema requirements (uncertainty, evidence citation)
6. Spatial coherence is diagnostic-only -- do NOT optimize for it
7. Propose general principles, not ZCTA-specific rules

OUTPUT FORMAT:
Return a JSON object with:
{
  "rationale": "why this edit should help",
  "predicted_effect": {"metric": "zone_macro_f1", "direction": "increase"},
  "operations": [
    {"op": "replace", "path": "<component>/<field>", "value": <new_value>}
  ]
}

Only use ops: add, replace, remove.
Paths must start with one of: evidence_template, feature_policy, rubric, scenario_memory.
"""


def build_evolver_prompt(
    harness: HarnessVersion,
    failure_report: dict,
    step: int,
) -> str:
    """Build the user prompt for the evolver model."""
    harness_summary = {
        "evidence_template": {
            "preamble": harness.editable.evidence_template.preamble[:500],
            "feature_order": harness.editable.evidence_template.feature_order,
            "max_features": harness.editable.evidence_template.max_features,
        },
        "feature_policy": {
            "include": harness.editable.feature_policy.include,
            "exclude": harness.editable.feature_policy.exclude,
        },
        "rubric": {
            "instructions": harness.editable.rubric.instructions[:500],
            "risk_levels": harness.editable.rubric.risk_levels,
        },
        "scenario_memory_count": len(
            harness.editable.scenario_memory.entries
        ),
    }

    return f"""\
## Evolution Step {step}

### Current Harness (summary)
```json
{json.dumps(harness_summary, indent=2)}
```

### Failure Report
```json
{json.dumps(failure_report, indent=2)}
```

### Task
Analyze the failure patterns and propose a harness patch that addresses
the most impactful failure group. Focus on general principles that will
transfer to unseen ZCTAs, not memorized fixes.

Return your response as a single JSON object with rationale, predicted_effect,
and operations.
"""


# ---------------------------------------------------------------------------
# Parse evolver response
# ---------------------------------------------------------------------------

def parse_evolver_response(
    response_text: str,
    step: int,
    from_harness_id: str,
    evolver_model: str,
) -> HarnessPatch:
    """Parse the evolver's JSON response into a HarnessPatch."""
    # Extract JSON from response (may be wrapped in markdown code block)
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    parsed = json.loads(text)

    operations = []
    for op_dict in parsed.get("operations", []):
        operations.append(PatchOperation(
            op=op_dict["op"],
            path=op_dict["path"],
            value=op_dict.get("value"),
        ))

    to_harness_id = f"harness_v{step:03d}"

    return HarnessPatch(
        patch_id=f"patch_t{step:03d}",
        from_harness=from_harness_id,
        to_harness=to_harness_id,
        evolver_model=evolver_model,
        operations=operations,
        evolver_rationale=parsed.get("rationale", ""),
        predicted_effect=parsed.get("predicted_effect", {}),
    )
