# DOE: R4 — Vision-Language Model Representation Arm

**Experiment ID:** s035-model-ladder / R4
**Domain:** GeoRSCT / Flood Risk — VLM Representation
**Status:** DESIGN (not locked)
**Date:** 2026-06-01
**Depends on:** R0-R2 results (money table exists), VLM adapters in yrsn

---

## Abstract

Test whether VLMs can extract flood risk signal directly from map images +
text evidence, bypassing engineered tabular features entirely. Four VLMs
(Gemini 2.0 Flash, Jina VLM, Amazon Nova Lite, Qwen2.5-VL-72B) receive the
same (map image, FEMA text, prompt) and produce structured flood risk
assessments. The DOE question: is VLM choice a significant factor, or is R4
solver-robust?

---

## Motivation

R0-R2 build representations by engineering features from raw data. R4 asks:
can a VLM derive the representation from the raw evidence directly?

rsct-vision proves this works for embedded hardware (camera → VLM → structured
signals → action loop). The R4 arm tests whether the same pattern transfers
to geospatial: flood map image + FEMA text → VLM → structured risk assessment.

If R4 works, R3 (trained CNN on raster patches) becomes optional — the VLM
is both feature extractor and classifier.

---

## Hypotheses

### H7: VLMs Extract Measurable Flood Risk Signal from Map + Text

**Statement:** At least one VLM produces structured flood risk assessments
that correlate with observed NFIP claims (Spearman rho > 0.3) across ZCTAs,
using only a map image + FEMA text as input.

| Variable Type | Description |
|---------------|-------------|
| Independent | VLM provider (Gemini Flash, Jina VLM, Nova Lite, Qwen2.5-VL) |
| Dependent | Spearman rho(VLM_risk_score, obs_nfip_event_claims) |
| Control | Prompt (fixed), map rendering (fixed), text source (fixed) |

### H8: VLM Choice Is Not a Significant Factor

**Statement:** The three VLMs produce risk scores with pairwise Spearman
rho > 0.7 (high inter-rater agreement), suggesting R4 is solver-robust.

| Variable Type | Description |
|---------------|-------------|
| Independent | VLM provider pair |
| Dependent | Pairwise Spearman rho between VLM risk scores |
| Control | Same input (map + text + prompt) |

### H9: R4 VLM Scores Correlate with R0-R2 Kappa Diagnostics

**Statement (exploratory):** ZCTAs where R0 kappa diagnostics flag failures
(low diag_leakage, low diag_residual_spatial) receive different VLM risk
scores than ZCTAs where R0 passes — suggesting VLMs detect the same spatial
structure the diagnostics measure.

---

## VLM Adapters

All three implemented in `yrsn/adapters/outbound/vlm.py`, sharing `ILLMClient`
port and `complete_with_reasoning()` interface.

| VLM | Adapter Class | Model ID | Gateway | Credential | Cost/1K images |
|-----|---------------|----------|---------|------------|----------------|
| Gemini 2.0 Flash | `GeminiVisionAdapter` | `gemini-2.0-flash` | Google AI | `GOOGLE_API_KEY` | FREE (15 RPM) |
| Jina VLM | `JinaVLMAdapter` | `jina-vlm` | Jina AI | `JINA_API_KEY` | ~$1 |
| Amazon Nova Lite | `BedrockNovaVisionAdapter` | `us.amazon.nova-lite-v1:0` | Bedrock | IAM (no key) | ~$0.10 |
| Qwen2.5-VL-72B | `Qwen2VLAdapter` | `qwen/qwen2.5-vl-72b-instruct` | OpenRouter | `OPENROUTER_API_KEY` | ~$3 |

All return:
```python
{
    "content": str,              # Final assessment
    "reasoning_content": str,    # Step-by-step reasoning (separate trace for Claude)
    "reasoning_tokens": int,
    "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}
}
```

---

## Input Representation

Each ZCTA receives a standardized input package:

### Map Image (rendered per ZCTA)

Static PNG rendered from GeoParquet + FEMA NFHL layers:

| Layer | Source | Visual Encoding |
|-------|--------|-----------------|
| ZCTA boundary | TIGER/Line | Black outline |
| Flood zones | FEMA NFHL | Color fill: AE=blue, VE=red, X=gray, A=light blue |
| Target ZCTA | Highlighted | Yellow fill + label |
| Neighbor ZCTAs | Queen contiguity | Thin outlines, flood zone coloring |
| Legend | Generated | Zone codes + colors |
| Scale bar | Computed | Kilometers |

Rendering: `geopandas` + `matplotlib`, fixed DPI (300), fixed extent
(ZCTA bbox + 20% buffer). One PNG per ZCTA.

### Text Evidence (structured per ZCTA)

```
ZCTA {id} in {county}, {state}.

FEMA Flood Zones:
- {pct}% in Zone AE (1% annual chance floodplain)
- {pct}% in Zone VE (coastal high hazard)
- {pct}% in Zone X (minimal flood hazard)

Demographics (ACS):
- Population: {pop}, Median income: ${income}
- SVI overall: {svi} ({interpretation})

Infrastructure:
- Nearest hospital: {km} km, {beds} beds
- Nearest pharmacy: {km} km

Historical Events:
- NFIP claims: {n} events, ${total} total losses
- 311 flood reports: {n} (if available)
```

### Prompt (fixed across all VLMs)

```
You are assessing flood risk for a US Census ZCTA (ZIP Code Tabulation Area).

Given:
1. A map showing the ZCTA and surrounding area with FEMA flood zones
2. Text evidence about the ZCTA's demographics, infrastructure, and history

Produce a structured flood risk assessment:

{
  "risk_score": <float 0-1, overall flood risk>,
  "confidence": <float 0-1, your confidence in this assessment>,
  "zone_interpretation": "<what the flood zone map tells you>",
  "vulnerability_factors": ["<factor 1>", "<factor 2>", ...],
  "spatial_reasoning": "<how neighboring areas affect this ZCTA's risk>",
  "evidence_used": ["<specific visual/text elements referenced>"]
}

Be precise. Reference specific visual elements from the map and specific
numbers from the text. If you cannot determine something, say so.
```

---

## Response Schema & Datapoints

### Primary Outcome (per ZCTA x VLM)

| Column | Type | Description |
|--------|------|-------------|
| `zcta_id` | str | ZCTA identifier |
| `vlm` | str | Provider name (claude/gemini/qwen) |
| `risk_score` | float | VLM-produced risk score [0,1] |
| `confidence` | float | VLM self-reported confidence [0,1] |

### Spatial Reasoning Quality (scored by human or second VLM)

| Column | Type | Description |
|--------|------|-------------|
| `layout_accuracy` | float [0,1] | Did VLM correctly identify spatial relationships? |
| `legend_accuracy` | float [0,1] | Did VLM correctly read choropleth legend + map colors? |
| `localization_accuracy` | float [0,1] | Did VLM identify the correct geographic area? |

### Cross-Modal Grounding

| Column | Type | Description |
|--------|------|-------------|
| `text_image_consistency` | float [0,1] | Answer consistent with both text and image? |
| `evidence_citation_count` | int | Number of specific visual/text elements referenced |
| `evidence_citation_quality` | float [0,1] | References specific elements vs generic prose |

### Structured Output Discipline

| Column | Type | Description |
|--------|------|-------------|
| `parse_success` | bool | Response parsed into expected JSON schema? |
| `fixup_needed` | bool | Required regex/manual fixup to parse? |
| `hallucination_count` | int | Zones/values invented (not in image or text) |
| `refusal` | bool | VLM declined to assess (legitimate uncertainty) |

### Calibration

| Column | Type | Description |
|--------|------|-------------|
| `confidence_calibrated` | float | |confidence - accuracy| (lower = better) |
| `risk_score_vs_nfip` | float | Spearman rho with observed NFIP claims |

### Cost & Latency

| Column | Type | Description |
|--------|------|-------------|
| `prompt_tokens` | int | Input tokens (from adapter) |
| `completion_tokens` | int | Output tokens (from adapter) |
| `reasoning_tokens` | int | Thinking tokens (Claude only) |
| `latency_ms` | int | Wall-clock time for complete_with_reasoning() |
| `cost_usd` | float | Derived: tokens x provider rate |

---

## Experiment Matrix

| Phase | Script | Input | Output | Instance |
|-------|--------|-------|--------|----------|
| R4.1 | `render_zcta_maps.py` | GeoParquet + NFHL | `maps/{scenario}/{zcta_id}.png` | local |
| R4.2 | `build_zcta_evidence.py` | Assembled parquet | `evidence/{scenario}/{zcta_id}.txt` | local |
| R4.3 | `run_vlm_assessment.py` | Maps + evidence + prompt | `results/s035/r4_{vlm}_{scenario}.parquet` | local (API calls) |
| R4.4 | `score_vlm_quality.py` | R4.3 outputs + ground truth | `results/s035/r4_quality_scores.parquet` | local |
| R4.5 | `compute_vlm_comparison.py` | All R4 outputs + R0-R2 results | `results/s035/r4_money_table.json` | local |

### Cost Estimate

| VLM | ZCTAs | Cost/ZCTA | Total | Time |
|-----|-------|-----------|-------|------|
| Gemini Flash | 1,596 | FREE | $0 | ~2 hr (15 RPM) |
| Jina VLM | 1,596 | ~$0.001 | ~$1.60 | ~1 hr |
| Nova Lite | 1,596 | ~$0.0001 | ~$0.16 | ~30 min |
| Qwen2.5-VL | 1,596 | ~$0.003 | ~$5 | ~1 hr |
| **Total** | | | **~$7** | **~5 hr** |

No SageMaker needed — API calls from local or lightweight instance.

---

## Money Table Extension

R4 adds columns to the existing s035 money table:

```
scenario | target | ... existing R0-R2 columns ... | rho_claude | rho_gemini | rho_qwen | vlm_agreement | vlm_vs_r2_delta
```

| Column | Description |
|--------|-------------|
| `rho_gemini` | Spearman(gemini_risk_score, obs_nfip_claims) |
| `rho_jina` | Spearman(jina_risk_score, obs_nfip_claims) |
| `rho_nova` | Spearman(nova_risk_score, obs_nfip_claims) |
| `rho_qwen` | Spearman(qwen_risk_score, obs_nfip_claims) |
| `vlm_agreement` | Mean pairwise Spearman across 4 VLMs |
| `vlm_vs_r2_delta` | max(rho_vlm) - rho(pred_R2, obs) |

---

## Null-Input Controls (v1.1)

Every evaluation batch includes control inputs interspersed with real ZCTAs.
If a VLM scores controls the same as real inputs, it is returning priors
not image understanding.

| Control | Image | Text | Expected Response |
|---------|-------|------|-------------------|
| `null_blank` | Solid white PNG (same dimensions) | None | risk_score near 0.5 (maximum uncertainty) or refusal |
| `null_noise` | Uniform random noise PNG | None | risk_score near 0.5 or refusal |
| `null_inverted` | Real ZCTA map with inverted colormap | Same text as real ZCTA | risk_score should DIFFER from non-inverted |
| `null_mismatch` | Real ZCTA map from scenario A | Text evidence from unrelated ZCTA in scenario B | Inconsistency detection or different score |

**Gate:** If `mean(|score_real - score_null_blank|) < 0.1` for any VLM,
that VLM fails the discrimination gate and is reported as "no signal."

### Control injection protocol

- 10% of inputs per batch are controls (random positions)
- Controls use the same prompt as real ZCTAs
- Control results stripped before computing primary metrics
- Reported separately in `results/s035/r4_null_controls.json`

---

## Prompt Ablation (v1.1)

Three prompt variants to quantify what the VLM actually reads.

| Variant | Image | Text | Prompt Change |
|---------|-------|------|---------------|
| `P0_image_only` | Map PNG | None | "Assess flood risk from this map image only." |
| `P1_image_legend` | Map PNG + legend | None | Standard prompt minus text evidence section |
| `P2_full` | Map PNG + legend | Full text evidence | Standard prompt (baseline) |

**Analysis:** If `rho(P0) ~ rho(P2)`, text evidence adds nothing -- VLM
reads the map. If `rho(P0) << rho(P2)` and `rho(P2) ~ rho(null)`, VLM
reads the text numbers and ignores the map entirely.

---

## Deterministic Inference (v1.1)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.0 (greedy) | Eliminate stochastic variation |
| Top-p | 1.0 | No nucleus sampling |
| Model version | Pinned ARN/version per VLM | Bedrock model versions can change silently |
| Repetitions | k=3 per ZCTA | Agreement check: if 3 runs disagree, signal is noise |
| Seed | Provider-specific where supported | Gemini supports seed; others use temp=0 |

Log per-call: model version, request ID, latency_ms, token counts.

---

## Fold-Structured Evaluation (v1.1)

R4 uses the **same spatial-blocked folds** as R0-R2. This makes error bars
comparable even though R4 is zero-shot (no training).

| Step | Description |
|------|-------------|
| 1 | Run VLM on ALL ZCTAs (no train/test distinction for inference) |
| 2 | Compute Spearman rho(risk_score, obs_nfip_event_claims) **per fold** |
| 3 | Report mean rho +/- std across 5 folds |
| 4 | Wilcoxon signed-rank on per-fold rho(R4) vs per-fold R2(R0-R2) for direct comparison |

This gives R4 the same evaluation structure as R0-R2: 5 paired observations
per (scenario, target) cell, same Wilcoxon machinery.

---

## Evaluation Separation (v1.1)

R4 results are reported in their own section of the paper, NEVER mixed
into the R0-R2 money table.

| What R4 gets | What R4 does NOT get |
|--------------|---------------------|
| Own table: rho per (VLM, scenario, prompt_variant) | Row in the R0-R2 money table |
| Own figure: null controls vs real inputs | Kappa cascade diagnostics |
| Fold-structured error bars (same fold IDs) | H2/H3 uplift comparisons |
| Direct rho comparison with R2 headline | Holm-Bonferroni family membership |

Narrative framing: "We also asked whether off-the-shelf VLMs extract flood
signal from map images. Here is the evidence."

---

## Success Criteria

| Hypothesis | Criterion | Status |
|------------|-----------|--------|
| H7 (gate) | Any VLM rho > 0.3 with NFIP claims AND null controls discriminated | PENDING |
| H8 | Pairwise VLM rho > 0.7 | PENDING |
| H9 | Flagged vs unflagged VLMs differ (exploratory) | PENDING |

---

## Kill Rules

- H7 FAIL on all 4 VLMs → VLMs cannot extract flood risk from maps; report as negative result
- Null controls not discriminated (gate above) → VLM returns priors, not signal; report as negative result
- Parse success < 50% on any VLM → adapter needs prompt engineering before rerun
- All 4 VLMs produce constant risk_score (zero variance) → prompt is broken
- P0 rho ~ P2 rho AND both ~ null → VLM is noise regardless of input modality

---

## DO NOT Constraints

- Do NOT fine-tune any VLM (frozen inference only -- controlled experiment)
- Do NOT vary the prompt across VLMs (same prompt variant, different solver)
- Do NOT use VLM risk_score as a feature in R0-R2 solvers (R4 is independent)
- Do NOT cherry-pick ZCTAs -- run all modelable ZCTAs per scenario
- Do NOT use GPU instances (API calls only)
- Do NOT place R4 results in the R0-R2 money table or kappa cascade
- Do NOT use temperature > 0 (greedy decoding only)

---

## Relationship to R0-R2

R4 is a **parallel exploratory arm**, not a sequential extension:

```
R0 -> R1 -> R2  (engineered features, tabular solvers, primary ladder)
R4              (raw evidence, VLM solver, exploratory)
```

R4 is evaluated with the same fold structure but reported in its own section.
The comparison is descriptive: per-fold rho(R4) placed alongside per-fold
metric(R0/R1/R2) for the same (scenario, target) cell. No formal hypothesis
test compares R4 to R0-R2 -- the evaluation protocols are too different
(zero-shot rho vs supervised CV metric).

Possible outcomes:
- R4 rho competitive with R2 → interesting positive result, suggests VLMs extract geospatial signal
- R4 rho near zero or null-indistinguishable → honest negative result, saves other researchers time
- R4 rho driven by text, not image (P0 << P2) → VLM is reading numbers, not maps

All three outcomes are publishable.

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-06-01 | Initial R4 DOE: four VLMs, datapoint schema, cost estimate |
| v1.1 | 2026-06-02 | Null-input controls, prompt ablation (P0/P1/P2), deterministic inference (temp=0, pinned versions, k=3), fold-structured evaluation, evaluation separation from R0-R2 money table |
