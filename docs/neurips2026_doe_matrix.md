# NeurIPS 2026 — RSCT Geometric Evaluation Framework
## DOE Experiment Matrix & Work Process

**Target:** NeurIPS 2026 Evaluations & Datasets Track
**Abstract deadline:** May 4, 2026 AOE
**Full paper deadline:** May 6, 2026 AOE
**Working title:** *Beyond Binary Retrieval Scores: R/S/N Decomposition as a Geometric Evaluation Framework for Agentic Context Quality*
**Author:** Rudy Martin (Next Shift Consulting LLC)

---

## 1. Experiment Matrix (One-Page View)

| DOE | Question | Factors | Primary Metrics | Datasets | Paper Section | Run Order | Status |
|-----|----------|---------|-----------------|----------|---------------|-----------|--------|
| **DOE-1** | Does the geometry tree hold across embedding substrates? | Embedding model × Scoring method | Balanced acc, Macro F1, per-class AUC, S_sup recall | MIRACL, FEVER, HotpotQA, SciFact + Nemotron embed | §4.3, §4.5 | **1st** | NOT STARTED |
| **DOE-2** | Can τ_edge be derived without grid search? | Threshold method × Dataset | Balanced acc at frozen threshold, transfer gap | MIRACL, FEVER, HotpotQA, SciFact | §4.4, §5 | **3rd** | NOT STARTED |
| **DOE-3** | Is the R/S_sup confusion zone real and consistent? | Dataset × Score family × Region definition | Cohen's d, overlap coefficient, error concentration | MIRACL, FEVER, HotpotQA, SciFact | §5.1, §5.2 | **2nd** | PARTIAL (S016C in-domain done) |
| **DOE-4** | Can the confusion zone become a governance workflow? | Escalation policy × Reporting granularity | % auto-resolved, % escalated, error capture rate | Same model outputs from DOE-1/3 | §5.3, §5.4 | **4th** | DESIGN ONLY |
| **DOE-5** | Which geometric features matter and why? | Importance method × Dataset | Feature ranking, top-k stability, SHAP | In-domain + cross-domain | §3.3, §3.4 | **5th** | PARTIAL (S016C feature importance done) |
| **DOE-6** | Do geometric features transfer to geospatial embeddings? | Embedding substrate (text vs geo) × Task domain | Balanced acc, S_sup recall, feature overlap | NQ-Geo filter, PDFM pairs | §4.6 (optional) | **6th** | NOT STARTED |

**Kill rules:**
- DOE-1: If cosine beats tree by >3pp across all substrates → narrow claims to "characterization, not method"
- DOE-2: If threshold is unstable (>5pp variance across bootstraps) → present as local calibration
- DOE-3: If confusion zone disappears cross-dataset → it's corpus-specific, not a general finding
- DOE-6: If AUC < 0.52 on geospatial → drop from paper, save for Viasat-specific follow-up

---

## 2. Required Datasets & Links

### 2.1 Core Text Retrieval (Already In Hand from S016C)

| Dataset | Source | Format | License | Status |
|---------|--------|--------|---------|--------|
| **MIRACL** (en) | https://huggingface.co/datasets/miracl/miracl | Query→passage with relevance labels | Apache 2.0 | ✅ Embedded, tree trained |
| **FEVER** | https://huggingface.co/datasets/fever/fever | Claim verification as retrieval | CC BY-SA 3.0 | ✅ Cross-dist done |
| **HotpotQA** | https://huggingface.co/datasets/hotpot_qa | Multi-hop QA retrieval | CC BY-SA 4.0 | ✅ Cross-dist done |
| **SciFact** | https://huggingface.co/datasets/allenai/scifact | Scientific claim verification | CC BY-NC 2.0 | ✅ Cross-dist done |

### 2.2 Embedding Models to Compare (DOE-1)

| Model | Source | Dim | Inference | License | Action Needed |
|-------|--------|-----|-----------|---------|---------------|
| **S016C baseline** (Titan/Jina/ModernBERT) | Already in NPZ files on S3 | 1024→64 projected | Cheapest (tree only) | N/A | ✅ Done |
| **MiniLM-L6-v2** | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | 384 | Fast | Apache 2.0 | ⬜ Generate embeddings |
| **Nemotron embed 1B** | https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2 | 2048 | Medium | NVIDIA Open Model License | ⬜ Generate embeddings |
| **Nemotron embed 8B** (optional) | https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1 (dataset) / model via NIM | 4096 | Heavier | NVIDIA Open Model License | ⬜ Optional |

### 2.3 Safety/Content Baseline (DOE-1 comparator)

| Model | Source | Type | License | Action Needed |
|-------|--------|------|---------|---------------|
| **Nemotron 3 Content Safety** | https://huggingface.co/nvidia/Nemotron-3-Content-Safety | 4B classifier (safe/unsafe) | NVIDIA Open + Gemma ToU | ⬜ Run on S016C test set for latency/accuracy comparison |
| **Nemotron Content Safety Reasoning** | https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B | 4B with reasoning traces | NVIDIA Open + Gemma ToU | ⬜ Optional deeper comparison |

### 2.4 Geospatial Extension (DOE-6)

| Dataset | Source | Format | License | Action Needed |
|---------|--------|--------|---------|---------------|
| **PDFM Embeddings** (Google) | https://github.com/google-research/population-dynamics | GNN embeddings for US postal codes/counties + 27 benchmark tasks | Apache 2.0 (code), research access (embeddings) | ⬜ Request access, download conus27 benchmark |
| **PDFM conus27 benchmark** | Same repo, `benchmarks/` directory | CSV with health/socioeconomic/environmental ground truth | Apache 2.0 | ⬜ Download |
| **NQ geographic filter** | https://ai.google.com/research/NaturalQuestions (or BigQuery: `bigquery-public-data.natural_questions`) | Filter existing NQ for geographic queries | CC BY-SA 3.0 | ⬜ Run NER filter script |
| **GeoGLUE** | https://arxiv.org/abs/2305.06545 / check HuggingFace | Geographic language understanding | Research use | ⬜ Evaluate availability |

### 2.5 Nemotron Agent Safety Data (supplementary)

| Dataset | Source | Format | License | Action Needed |
|---------|--------|--------|---------|---------------|
| **Nemotron Agentic Safety** | https://huggingface.co/nvidia (search "agentic safety") | ~11K agent workflow traces | NVIDIA Open | ⬜ Locate exact HF path, evaluate R/S/N labeling feasibility |
| **Nemotron Content Safety Dataset v2** | Referenced in Content Safety model card | Safety taxonomy + annotations | NVIDIA | ⬜ Check public availability |

---

## 3. Work Process — Week-by-Week

### Week 1: April 4–10 (Substrate Swap & Confusion Zone)

**Goal:** Complete DOE-1 core comparisons and DOE-3 cross-dataset characterization.

**Day 1–2 (Fri–Sat):**
```
□ Download MiniLM-L6-v2 from HuggingFace
□ Generate MiniLM embeddings for MIRACL test set
□ Compute cosine similarity AUC on MIRACL (the baseline everyone will ask about)
□ Pull cross-distribution P(S_sup) histograms from existing S016C results
  → Extract per-class conditional distributions from the 330-line JSON
  → This answers: is confusion zone tighter on FEVER/HotpotQA? (data ceiling diagnostic)
```

**Day 3–4 (Sun–Mon):**
```
□ Download Nemotron embed 1B (llama-nemotron-embed-vl-1b-v2)
□ Generate Nemotron embeddings for MIRACL test set
  → Note: 2048-dim, will need projection or direct geometric feature computation
□ Run S016C tree pipeline on Nemotron embeddings
□ Compute cosine AUC on Nemotron embeddings
□ Build DOE-1 comparison table: S016C tree vs MiniLM cosine vs Nemotron cosine
```

**Day 5–7 (Tue–Thu):**
```
□ DOE-3: Compute Cohen's d between R and S_sup for each cross-dist dataset
□ DOE-3: Compute overlap coefficient / Bhattacharyya distance
□ DOE-3: Measure error concentration inside vs outside confusion zone
□ Generate main figure: score density overlap plots (R vs S_sup) per dataset
□ NQ geographic filter: Run spaCy NER on NQ queries, extract geo-tagged subset
□ Start LaTeX skeleton — title, abstract placeholder, section headings, figure slots
```

**Week 1 deliverables:**
- [ ] DOE-1 comparison table (3+ embedding substrates)
- [ ] DOE-3 cross-dataset confusion zone table
- [ ] DOE-3 main figure (density overlaps)
- [ ] NQ-Geo subset ready for DOE-6
- [ ] LaTeX skeleton committed

---

### Week 2: April 11–17 (Threshold Derivation & Feature Analysis)

**Goal:** Complete DOE-2, DOE-5, and start DOE-6.

**Day 1–2:**
```
□ DOE-2: Extract P(S_sup) distribution from S016C dev set
□ DOE-2: Find breakpoint (elbow/crossover) in distribution
□ DOE-2: Map breakpoint to cosine threshold space
□ DOE-2: Freeze threshold, evaluate on held-out + cross-domain
□ DOE-2: Bootstrap stability analysis (100 resamples)
```

**Day 3–4:**
```
□ DOE-5: Run SHAP on frozen S016C tree
□ DOE-5: Compute permutation importance on cross-domain sets
□ DOE-5: Verify directional_asymmetry stability across datasets
□ DOE-5: Generate feature importance plot + partial dependence for top features
□ Write §3.3 (feature analysis) and §3.4 (theoretical connection to BCA)
```

**Day 5–7:**
```
□ DOE-6 fast path: Run tree on NQ-Geo subset → report AUC
□ DOE-6 PDFM: Request embedding access if not yet approved
□ DOE-6 PDFM: If access granted, download conus27 + embeddings
□ DOE-6 PDFM: Construct (u, v, R/S/N) triples using quintile labeling:
    - Same quintile = R
    - Adjacent quintile = S_sup  
    - Distant quintile = N
□ DOE-6 PDFM: Compute geometric features in native PDFM space
□ DOE-6 PDFM: Run tree → report AUC, feature rankings
□ Nemotron Content Safety: Run on S016C test samples, measure latency + accuracy
```

**Week 2 deliverables:**
- [ ] DOE-2 breakpoint plot + threshold transfer table
- [ ] DOE-2 pseudo-algorithm box for paper
- [ ] DOE-5 feature importance figure + SHAP
- [ ] DOE-6 NQ-Geo AUC (one row in transfer table)
- [ ] DOE-6 PDFM AUC (if access granted)
- [ ] §3 methods section drafted
- [ ] §4.1–4.4 results sections drafted

---

### Week 3: April 18–24 (Writing & Integration)

**Goal:** Complete all experimental sections, draft full paper.

**Day 1–3:**
```
□ DOE-4: Design certificate schema (not an experiment — a specification)
□ DOE-4: Compute % auto-resolved / % escalated / error capture from DOE-3 outputs
□ DOE-4: Build confusion zone workflow figure
□ DOE-4: Write SR 11-7 mapping table
□ Write §5.1–5.4 (confusion zone + implications)
□ Write §4.5 (Nemotron Content Safety comparison)
□ Write §4.6 (geospatial transfer, if results available)
```

**Day 4–5:**
```
□ Write §1 (Introduction — Nemotron 3 motivation, binary evaluation insufficient)
□ Write §2 (Background — RSCT framework, Nemotron embed stack)
□ Write §6 (Discussion — future work: hardening, error propagation, manager augmentation)
□ Draft abstract
□ All figures finalized
□ All tables finalized
```

**Day 6–7:**
```
□ Related work section
□ Appendix: full experimental details, hyperparameters, compute budget
□ Cross-reference all figures/tables
□ Internal review pass
```

**Week 3 deliverables:**
- [ ] Complete draft (all sections)
- [ ] All figures and tables
- [ ] Appendix with full DOE details

---

### Week 4: April 25–May 6 (Polish & Submit)

**April 25–30:**
```
□ Style pass — ensure E&D track requirements are met
□ Verify all dataset hosting / reproducibility requirements
□ Croissant metadata file (required for E&D track)
□ Code repository preparation (GitHub, anonymized for review)
□ Supplementary materials
□ Co-author review
```

**May 1–3:**
```
□ Final editing pass
□ Check page limits
□ Verify OpenReview formatting
□ Both authors create/verify OpenReview profiles
```

**May 4 (Abstract deadline):**
```
□ Submit abstract on OpenReview
```

**May 5–6 (Paper deadline):**
```
□ Final supplementary materials upload
□ Submit full paper + supplementary by May 6 AOE
```

---

## 4. Infrastructure & Compute

### Existing Assets (from S016C)
- AWS SageMaker with GPU quota (ml.g4dn or ml.p3 instances)
- S3 bucket: `yrsn-datasets/s016c/` with pre-embedded NPZ files
- `sagemaker_s016c_tree.py` — tree training/evaluation pipeline
- `sm_inspect.py` — environment introspection
- Cross-distribution results JSON already downloaded

### New Requirements
| Resource | Purpose | Estimated Cost | Timeline |
|----------|---------|---------------|----------|
| HuggingFace model downloads | MiniLM, Nemotron embed | Free | Week 1 |
| PDFM embedding access | Geospatial extension | Free (research) | Request immediately |
| Nemotron Content Safety inference | Baseline comparison | Free (HF) or NIM credits | Week 2 |
| LaTeX / Overleaf | Paper writing | Free tier sufficient | Week 1 onward |
| GitHub repo (anonymized) | Code release for E&D track | Free | Week 4 |

### Compute Budget Estimate
| Task | Instance | Hours | Cost |
|------|----------|-------|------|
| Embedding generation (MiniLM + Nemotron) | ml.g4dn.xlarge | ~4h | ~$3 |
| Tree training/eval across substrates | ml.m5.xlarge | ~2h | ~$1 |
| SHAP computation | ml.m5.2xlarge | ~3h | ~$3 |
| PDFM feature computation | local or ml.m5.xlarge | ~2h | ~$1 |
| **Total** | | **~11h** | **~$8** |

---

## 5. Paper Section ↔ DOE ↔ Figure Mapping

| Section | Content | DOE Source | Key Figure/Table |
|---------|---------|-----------|-----------------|
| §1 Intro | Nemotron 3 context explosion, binary eval insufficient | Motivation | — |
| §2 Background | RSCT framework, Y=R+S+N, geometric certification | Theory | Fig 1: R/S/N simplex |
| §3.1 Projection architecture | 1024→256→64, rotor model | S016C setup | Fig 2: Architecture |
| §3.2 Tree-based classifier | Geometry features, training | S016C | — |
| §3.3 Feature analysis | Why directional_asymmetry matters | DOE-5 | Fig 3: Feature importance |
| §3.4 Three-region derivation | Confusion zone discovery | DOE-3 | Fig 4: Score density overlap |
| §4.1 Within-distribution | MIRACL results | S016C (done) | Table 1: Multi-seed AUC |
| §4.2 Cross-distribution | FEVER/HotpotQA/SciFact | S016C (done) | Table 2: Transfer results |
| §4.3 Substrate comparison | MiniLM vs Nemotron vs tree | DOE-1 | Table 3: Substrate comparison |
| §4.4 Threshold derivation | Breakpoint → τ_edge | DOE-2 | Fig 5: Breakpoint plot, Alg 1 |
| §4.5 External comparator | Nemotron Content Safety | DOE-1 | Table 4: Cost vs accuracy |
| §4.6 Geospatial transfer | NQ-Geo / PDFM results | DOE-6 | Table 5: Cross-domain (optional) |
| §5.1 Confusion zone | R/S_sup overlap characterization | DOE-3 | Fig 4 (same as §3.4) |
| §5.2 Cross-dataset consistency | Confusion zone across corpora | DOE-3 | Table 6: Per-dataset overlap |
| §5.3 Escalation framework | Three-region certificate chain | DOE-4 | Fig 6: Workflow diagram |
| §5.4 Regulatory mapping | SR 11-7, MRM implications | DOE-4 | Table 7: SR 11-7 mapping |
| §6 Discussion | Future: hardening, error prop, manager augmentation | — | — |

---

## 6. Tracker Template

Copy this for each DOE and update as you go:

```
DOE-ID:        
Question:      
Status:        [NOT STARTED | IN PROGRESS | BLOCKED | COMPLETE | KILLED]
Fixed assets:  
Factor grid:   
Datasets:      
Metrics:       
Success rule:  
Kill rule:     
Figure output: 
Paper section: 
Blockers:      
Notes:         
```

---

## 7. Critical Path Dependencies

```
S016C results (DONE)
    │
    ├─→ DOE-1: Substrate swap (Week 1)
    │       │
    │       └─→ DOE-3: Confusion zone (Week 1, uses DOE-1 outputs)
    │               │
    │               ├─→ DOE-2: Threshold derivation (Week 2, uses DOE-3 zone bounds)
    │               │
    │               └─→ DOE-4: Certificate design (Week 3, uses DOE-3 statistics)
    │
    ├─→ DOE-5: Feature analysis (Week 2, uses frozen S016C model)
    │
    └─→ DOE-6: Geospatial extension (Week 2, independent path)
            │
            └─→ PDFM access approval (GATE — request immediately)
```

**Single point of failure:** PDFM embedding access approval. Request today. If denied, DOE-6 falls back to NQ-Geo filter only (no PDFM), which is still a valid geospatial transfer result.

---

## 8. Immediate Actions (Today, April 3)

```
1. □ Request PDFM embedding access: https://github.com/google-research/population-dynamics
2. □ Clone PDFM repo: git clone https://github.com/google-research/population-dynamics.git
3. □ Download MiniLM: huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
4. □ Download Nemotron embed 1B: huggingface-cli download nvidia/llama-nemotron-embed-vl-1b-v2
5. □ Extract cross-dist P(S_sup) histograms from S016C JSON (the labeling audit diagnostic)
6. □ Create Overleaf project with NeurIPS 2026 template
7. □ Create OpenReview profile (if not existing) — required for submission
8. □ Set up anonymized GitHub repo for code release
```
