# RSCT Context

Beyond Binary Retrieval Scores: R/S/N Decomposition as a Geometric Evaluation Framework for Agentic Context Quality

## Overview

Binary relevance labels conflate two failure modes:
- **Noise (N)**: Random, off-topic content (easy to detect)
- **Superfluous (S)**: On-topic but non-entailing content (causes hallucination)

R/S/N decomposition separates these via geometric features, enabling fine-grained retrieval evaluation without retraining.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from rsct import GeometryExtractor, KappaGate

# Extract 263-dim geometry features
extractor = GeometryExtractor()
features = extractor.extract(query_emb, passage_emb)

# Predict S_sup probability
gate = KappaGate.load("models/kappa_tree.pkl")
p_s_sup = gate.predict_proba(features)
```

## Experiments

| DOE | Description |
|-----|-------------|
| DOE-1 | Substrate independence (E5, MiniLM, Nemotron) |
| DOE-2 | Geometry sufficiency (263-dim vs interaction head) |
| DOE-3 | Confusion zone (R/S_sup discrimination) |
| DOE-4 | OOD detection via geometry |
| DOE-5 | Agent benchmark (quality-gated RAG) |
| DOE-6 | Geospatial modality (PDFM embeddings) |

```bash
python -m experiments.doe1_substrate.run
```

## Structure

```
rsct-context/
├── rsct/                      # Core library
├── experiments/               # DOE implementations
├── models/                    # Pretrained checkpoints
├── notebooks/                 # Figure reproduction
└── data/                      # Dataset instructions
```

## Datasets

- MIRACL (training)
- FEVER, HotpotQA, SciFact (cross-distribution)
- PDFM CONUS-27 (geospatial)

## License

Apache 2.0
