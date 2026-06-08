# Future Work: LLM as Representation Encoder in Model-Ladder

**Status:** Deferred (not in s035 v1.3 DOE scope)
**Prerequisite:** s035 R0/R1/R2 results with HistGBDT + Ridge + GNN complete
**Target venue:** SIGSPATIAL 2026 full paper Section 9 (Future Work) or follow-up submission

---

## Motivation

The s035 model-ladder tests whether GeoRSCT audit flags predict representation
uplift across three levels (R0 static, R1 hydrology, R2 temporal) and three
solver families (linear, tree, graph). A natural fourth representation source
is **LLM text embeddings** — semantic context that no engineered feature or
graph topology can capture.

This is NOT RAG. RAG conflates retrieval quality, prompt sensitivity, and
generation variance with representation quality. A controlled ablation requires
isolating the representation source. LLM embeddings do that.

---

## Design

### LLM as Column, Not Row

The model-ladder has two axes:

```
                    Tabular only    + Graph topology    + LLM semantic
                    (Ridge/HGBDT)   (GNN latents)       (text embeddings)
R0 (static)            x                x                    x
R1 (+ hydrology)       x                x                    x
R2 (+ temporal)        x                x                    x
```

LLM embeddings are a **representation source** (column), not a feature level
(row). They sit alongside GNN latents as an alternative way to encode ZCTA
context. The same HistGBDT trains on LLM embeddings + tabular features, using
the same folds, targets, and splits.

### Text Sources Per ZCTA

| Source | Coverage | Content | Expected Signal |
|--------|----------|---------|-----------------|
| FEMA NFHL zone narratives | All scenarios | "Zone AE: areas subject to inundation by the 1-percent-annual-chance flood..." | Regulatory flood risk semantics |
| 311 complaint text | Houston, NYC | Free-text flooding complaints with location | Ground-truth damage language |
| HWM field descriptions | Houston | "High water mark observed at 3.2m on residential structure..." | Physical flood evidence |
| NFIP claim descriptions | All scenarios | Policy type, building type, flood cause | Insurance domain knowledge |
| Historical event narratives | All scenarios | NOAA storm event database text entries | Event context and severity |

### Embedding Pipeline

1. For each ZCTA, concatenate all available text sources into a structured prompt:
   ```
   ZCTA {id} in {county}, {state}.
   Flood zone: {FEMA zone description}
   Recent events: {event narratives}
   311 reports: {complaint excerpts}
   Infrastructure: {NFIP policy summary}
   ```

2. Embed with a frozen encoder (no fine-tuning — controlled experiment):
   - **Candidate encoders:** Titan (1024D, already in stack via Bedrock),
     all-MiniLM-L6-v2 (384D, local/free), Jina v3 (1024D)
   - Project to 32D via PCA to match GNN latent dimension
   - One embedding per ZCTA (static), broadcast across events

3. Concatenate projected embeddings with tabular features:
   ```
   X = [R0_features | R1_features | R2_features | llm_32d]
   ```

4. Train same HistGBDT + Ridge on expanded feature set, same folds.

### What This Tests

**Primary question:** Does semantic context from text add predictive signal
beyond engineered features + graph topology?

**Kappa diagnostic predictions:**
- High `diag_solver` (HistGBDT agrees with Ridge) + high `diag_residual_spatial`
  (no error clustering) → LLM embeddings unlikely to help (existing features
  already capture the signal)
- Low `diag_solver` + low `diag_residual_spatial` → complex spatially-structured
  signal that neither linear nor tree models capture → LLM semantics may help
  if the missing signal is in text

**Control comparison:**
- LLM embeddings vs GNN latents: both are 32D dense representations of ZCTA
  context, but from different modalities (text vs graph)
- If GNN >> LLM: topology matters more than semantics for flood prediction
- If LLM >> GNN: domain knowledge in text matters more than spatial structure
- If both help additively: different failure modes, combine them

---

## Why Not RAG

| Approach | What It Tests | Confounds | Controlled? |
|----------|--------------|-----------|-------------|
| **LLM embedding** | Does text-derived representation add signal? | Encoder choice, projection dimension | Yes — same folds, same solver |
| **RAG** | Does retrieval + reasoning improve prediction? | Retrieval quality, chunk size, prompt, generation variance, LLM choice | No — too many moving parts |

RAG is the right architecture for a production system. LLM embedding is the
right experimental design for an ablation study. The paper needs the latter.

RAG could be tested as a follow-up once the representation value is established:
if LLM embeddings help, then RAG (which dynamically selects which text to embed)
should help more. But that's a separate hypothesis.

---

## VLA / VLM / JEPA Status

| Model Family | Status | Details |
|-------------|--------|---------|
| **VLM** (Vision-Language Model) | **PROMOTED to R4 arm** | Three adapters implemented in `yrsn/adapters/outbound/vlm.py`: Claude Opus (`ClaudeVisionAdapter`), Gemini Flash (`GeminiVisionAdapter`), Qwen2.5-VL-72B (`Qwen2VLAdapter`). Does NOT need satellite imagery — uses rendered FEMA flood zone maps + structured text evidence. Full DOE in `DOE_R4_vlm.md`. rsct-vision proves the pattern works for embedded hardware. |
| **VLA** (Vision-Language-Action) | Deferred (R5) | No action space in ZCTA flood prediction. Demo/explanation layer only. |
| **JEPA** (Joint Embedding Predictive Architecture) | Deferred | Self-supervised spatial representation learning. Interesting but needs training from scratch on ZCTA spatial context. No existing code or pretrained model for this domain. |

VLM was promoted because: (1) rsct-vision demonstrated the camera→VLM→structured-output
pattern works without training a domain-specific CNN, (2) adapters already exist in
the yrsn hex arch, (3) rendered flood maps + FEMA text is a cheaper input pipeline
than satellite imagery, (4) the three-way comparison tests solver robustness at ~$29 total.

---

## Prerequisites

1. **s035 v1.3 complete** — R0/R1/R2 results with HistGBDT + Ridge + GNN across
   all 4 scenarios. The kappa diagnostics and money table must exist before
   adding another representation column.

2. **Text data on S3** — FEMA zone narratives and NFIP descriptions are already
   in assembled parquets (as metadata columns, not features). 311 text is in
   raw CSVs. Need extraction + structuring job.

3. **Embedding compute** — Titan via Bedrock or local SBERT. Cost: ~$2 for
   1,600 ZCTAs x 500 tokens (Titan). Trivial.

4. **Projection** — PCA to 32D. Fit on train folds only (no data leakage).
   Save projection matrix for reproducibility.

---

## Estimated Effort

| Task | Effort | Dependency |
|------|--------|------------|
| Text extraction + structuring per ZCTA | 1 day | Raw data on S3 |
| Embedding pipeline (Titan or SBERT) | 0.5 day | swarm_auth for Bedrock |
| PCA projection + feature join | 0.5 day | Embeddings computed |
| Training runs (reuse existing scripts) | 1 hour (SageMaker) | Features joined |
| Kappa recomputation + money table update | 0.5 day | Training complete |
| Paper section writing | 1 day | Results available |

**Total: ~3.5 days after s035 v1.3 completes.**

---

## Paper Framing

If included in SIGSPATIAL 2026:

> "We further test whether LLM-derived semantic representations of ZCTA flood
> context add predictive signal beyond engineered features (R0-R2) and learned
> graph topology (GNN). Using frozen Titan embeddings of structured ZCTA
> descriptions projected to 32 dimensions, we find [result]. The GeoRSCT
> diagnostic correctly predicted that scenarios with [kappa pattern] would
> benefit from semantic features, while scenarios with [other pattern] would
> not."

If deferred to follow-up:

> "Future work will extend the representation ladder to include LLM-derived
> semantic features from FEMA narratives and 311 complaint text, testing
> whether domain knowledge encoded in text complements the spatial and
> temporal features evaluated here."
