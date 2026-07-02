# Academic State of the Art: Flood Damage Estimation, Prediction, and Ground Truth Construction

**Date:** 2026-07-02
**Scope:** Peer-reviewed literature, government reports, and research institution outputs relevant to FloodRSCT's certificate-based approach to flood damage data governance.

---

## 1. Taxonomy of Flood Damage Models (Merz et al. Lineage)

### 1.1 The Merz et al. Framework

The foundational taxonomy comes from **Merz, Kreibich, Schwarze & Thieken (2010)**, "Assessment of economic flood damage," *Natural Hazards and Earth System Sciences*, 10, 1697-1724. [ABSTRACT-ONLY -- PDF fetched but binary-encoded, could not extract full text; claims below are corroborated by multiple citing papers.]

Key categories in the Merz taxonomy:

| Category | Description | Dominant? |
|----------|-------------|-----------|
| **Empirical** | Derived from observed flood event data (field surveys, insurance claims). Data-hungry but context-specific. | Yes, historically |
| **Synthetic/Expert-based** | "What-if" analysis by domain experts estimating damage at various flood depths. Used when empirical data unavailable. | Yes, in data-scarce regions |
| **Multi-variable** | Extends beyond depth-only to include velocity, duration, return period, building characteristics. | Growing |
| **Absolute vs. Relative** | Absolute models estimate monetary loss; relative models estimate damage ratio (% of replacement value). | Both common |

The dominant paradigm remains **depth-damage functions (DDFs)** -- monotonic curves relating flood water depth to damage ratio. These originate from FIA/USACE work in the 1970s and remain embedded in Hazus, JBA, and most commercial models.

**Key Merz et al. conclusion (2010):** Depth alone explains a small fraction of damage variance. Building characteristics, contamination, flood duration, velocity, and precautionary measures all matter. Despite this, DDFs remain the standard because alternatives require data that is rarely available at scale.

### 1.2 Gerl et al. (2016) -- The Benchmarking Problem

**Gerl, Kreibich, Franco, Marechal & Schroter (2016)**, "A Review of Flood Loss Models as Basis for Harmonization and Benchmarking," *PLOS ONE*, 11(7): e0159791. [ABSTRACT-ONLY]

- Compiled "nearly a thousand vulnerability functions" across models globally.
- Found only ~50% of loss models were accompanied by explicit validation at the time of their proposal.
- Proposed harmonization via reduction to joint input variables, but acknowledged this is fundamentally difficult due to model heterogeneity.

**Relevance to FloodRSCT:** The fact that half of published loss models have *no validation* is a striking gap. RSCT's certificate framework would flag these as having undefined Relevance.

---

## 2. Physics-Based, Empirical, and Commercial Approaches

### 2.1 Wing et al. (2017, 2020) -- Large-Scale Flood Modeling and the DDF Problem

**Wing, Bates, Sampson, Smith, Johnson & Erickson (2017)**, "Validation of a 30 m resolution flood hazard model of the conterminous United States," *Water Resources Research*, 53(9): 7968-7986. [TRAINING-RECALL]

- Used LISFLOOD-FP (local inertial shallow water equations) at 30m resolution across CONUS.
- Validated against FEMA flood maps: 78% spatial agreement (82% in coastal areas).
- Trade-off: national consistency vs. local detail (rectangular channels, simplified defenses).

**Wing, Pinter, Bates & Kousky (2020)**, "New insights into US flood vulnerability revealed from flood insurance big data," *Nature Communications*, 11, 1444. [VERIFIED -- full text read via PMC]

This is the most important paper for FloodRSCT's positioning. Key findings:

1. **DDFs match NFIP observations poorly.** The FIA curve produces a *negative coefficient of determination* against actual claims -- meaning the mean value is a better predictor than the standard curve.
2. **Observed damage follows a beta distribution**, not a monotonic function. Damage concentrates bimodally at 0-10% and 90-100% loss ratios, shifting toward total loss with increased depth.
3. **Mean absolute errors of 84-105%** when applying standard DDFs to NFIP claims data.
4. **Overestimation at shallow depth (~25% too high at 1 ft), underestimation at deep depth (~25% too low).**
5. Analyzed 2,085,015 NFIP claims.

**Critical implication for FloodRSCT:** Wing et al. (2020) demonstrate that the *instrument* (DDFs) is broken, not just noisy. Our R=0.000 finding for New Orleans is consistent with their finding that DDFs have negative R-squared. The difference is that we treat this as a *data validity signal* (the certificate says "this data source has zero relevance to the prediction target"), while Wing et al. treat it as a *model improvement opportunity*.

### 2.2 Hazus -- FEMA's Standard Tool

**Scawthorn, Blais, Seligson, Tate, Chang, Mifflin, Thomas, Murphy & Jones (2006)**, "HAZUS-MH flood loss estimation methodology. II. Damage and loss assessment," *Natural Hazards Review*, 7(2): 72-81. [ABSTRACT-ONLY]

**Known failures and limitations** (synthesized from multiple sources including the Hazus 5.1 Technical Manual):

| Limitation | Detail |
|-----------|--------|
| Outdated DDFs | Based on FIA data through 2001; curves developed 1970-1997 |
| NFIP coverage caps distort curves | "Substantial damage" threshold at 50% of structure value reflects insurance caps, not physics |
| No uncertainty quantification | Hazus provides point estimates only; upper-bound losses can be 3x lower-bound |
| Building type oversimplification | Unlike earthquake model, flood model ignores construction type, design quality |
| DEM sensitivity | Digital elevation data choice is the *most influential* modeling parameter |
| No structural failure modeling | Assumes major structural components survive; only estimates finish/content damage |
| Individual building accuracy poor | Aggregate-level reasonable; individual-building estimates diverge significantly |

### 2.3 JBA Risk Management -- Commercial Global Model

**JBA Risk Management** (commercial; model details from press releases and product pages, not peer-reviewed). [ABSTRACT-ONLY]

- First probabilistic catastrophe model for flood at global scale.
- 15M+ simulated flood events, 30m resolution maps.
- Updated March 2026 with CMIP6 climate event sets (2080 timeline).
- Built on Oasis Loss Modelling Framework.
- Validation primarily through (re)insurance industry comparison, not academic peer review.
- Uses building footprint disaggregation for exposure resolution.

**Limitation for our purposes:** JBA models are proprietary and validation details are not publicly available. The model is designed for portfolio-level (re)insurance pricing, not for diagnosing data quality at the claim level.

---

## 3. ML/AI Flood Damage Prediction

### 3.1 FloodGenome (Liu & Mostafavi, 2025)

**Liu, C. and Mostafavi, A. (2025)**, "FloodGenome: interpretable machine learning for decoding features shaping property flood risk predisposition in cities," *Environmental Research: Infrastructure and Sustainability*. [VERIFIED -- full text read via IOP Science]

| Attribute | Detail |
|-----------|--------|
| **Model** | Random forest + k-means clustering + SHAP |
| **Target** | Property flood risk predisposition (4 levels: low/medium/high/extreme) |
| **Resolution** | Census block groups (training); 500m grid (application) |
| **Data** | NFIP claims 2003-2023, NLCD, NHD, NSI, NASA elevation |
| **MSAs studied** | Houston, Miami, New Orleans, New York |
| **ROC-AUC** | Low: 0.89, Medium: 0.83, High: 0.88, Extreme: 0.96 |
| **Key concept** | "Flood risk predisposition" = event-independent, inherent susceptibility |

**Critical observation for FloodRSCT:** FloodGenome studies the *same four MSAs* we study. Their approach treats NFIP claims as reliable ground truth without questioning data quality per-MSA. Our finding that New Orleans NFIP data has R=0.000 directly challenges the validity of their New Orleans results. If the ground truth is corrupt, high ROC-AUC is meaningless.

### 3.2 FloodDamageCast (Liu, Huang, Yin, Brody & Mostafavi, 2024)

**Liu, C.-F., Huang, L., Yin, K., Brody, S. & Mostafavi, A. (2024)**, "FloodDamageCast: Building flood damage nowcasting with machine-learning and data augmentation," *International Journal of Disaster Risk Reduction*. [ABSTRACT-ONLY -- arXiv abstract verified]

| Attribute | Detail |
|-----------|--------|
| **Model** | GAN-based data augmentation + LightGBM |
| **Target** | Residential flood damage nowcasting |
| **Resolution** | 500m x 500m grid cells |
| **Geography** | Harris County, TX (Hurricane Harvey) |
| **Key problem** | Class imbalance (96.4% majority class) |
| **Key result** | Identifies high-damage spatial areas missed by baselines |

**Note:** Same research group as FloodGenome (Texas A&M, Mostafavi lab). The "nowcasting" framing is novel -- they aim for near-real-time damage estimation during events rather than post-hoc analysis.

### 3.3 iClaim (Yang, Shen et al., 2022)

**Yang, Q., Shen, X., Yang, F., Anagnostou, E.N. et al. (2022)**, "Predicting Flood Property Insurance Claims over CONUS, Fusing Big Earth Observation Data," *Bulletin of the American Meteorological Society*, 103(3). [ABSTRACT-ONLY]

| Attribute | Detail |
|-----------|--------|
| **Model** | Random forest (two-step: damage classification + claim count regression) |
| **Target** | Number of NFIP claims per grid cell per event |
| **Data** | NFIP + flood extent + precipitation + river stage + topography |
| **Evaluation** | 446,446 grid samples, 589 flood events, 2016-2019 |
| **R-squared** | Grid/event: >0.5; County/event: >0.9; Event cumulative: >0.95 |
| **Innovation** | First to use NRT remotely-sensed flood extent as predictor |

**Key insight:** iClaim is the closest existing work to multi-source fusion for flood damage prediction. However, it predicts *claim counts*, not damage amounts, and does not assess data quality -- it assumes NFIP claims are the authoritative target.

### 3.4 Wagenaar et al. (2017) -- Multi-Variable Models

**Wagenaar, de Jong & Bouwer (2017)**, "Multi-variable flood damage modelling with limited data using supervised learning approaches," *NHESS*, 17, 1683-1696. [ABSTRACT-ONLY]

- Netherlands Meuse flood 1993 data enriched with 2D simulation outputs (velocity, duration, return period) and cadastre data.
- Tree-based methods (regression trees, bagging, random forest) outperformed Bayesian networks.
- 20% reduction in MAE compared to depth-only models.

### 3.5 Schroter et al. (2018) -- 3D City Models

**Schroter, Ludtke, Redweik, Meier, Bochow, Ross, Nagel & Kreibich (2018)**, "Flood loss estimation using 3D city models and remote sensing data," *Environmental Modelling & Software*, 105, 118-131. [ABSTRACT-ONLY]

- Used random forests on 3D city model features (building area, height) + remote sensing urban structure types.
- Building geometric properties explained flood vulnerability.
- Accuracy comparable to models using detailed empirical data.

### 3.6 Wagenaar et al. (2020) -- ML Future Vision

**Wagenaar, Curran, Balbi, Bhardwaj, Soden, Hartato, Sarica, Ruangpan, Molinario & Lallemant (2020)**, "Invited perspectives: How machine learning will change flood risk and impact assessment," *NHESS*, 20, 1149-1161. [VERIFIED -- full text read]

Key projections:
- ML will improve exposure mapping (building detection from satellite), hazard modeling (SAR flood detection), and impact prediction (multi-variable damage models).
- **Critical bottleneck:** "Many ML applications in flood risk and impact modelling appear to be limited by a lack of data, especially training data." Flood events are rare; data collection during emergencies is difficult.
- Computer vision could extract missing building attributes from street-level imagery.
- **Data standardization protocols** and collaborative stakeholder agreements needed.

**Relevance to FloodRSCT:** This paper identifies data scarcity as the key barrier but does not propose a framework for *evaluating* data quality. RSCT certificates fill exactly this gap.

---

## 4. The Ground Truth Problem

### 4.1 Insurance Penetration: The 70% Gap

Multiple sources converge on the same finding: **NFIP captures at most 30% of actual flood losses in the US.**

| Source | Finding |
|--------|---------|
| Neptune Flood (2025 report) | 70% of annual flood losses ($17.1B) are uninsured |
| Wing et al. (2020) | 2M+ claims analyzed, but these represent only insured properties |
| FEMA penetration data | 3.9% of all US housing units had active NFIP policy in 2019 |
| Harris County (Harvey) | ~15% NFIP penetration |
| Kousky & Shabman (2014, RFF) | NFIP was never intended to be actuarially sound |
| Nayak et al. (2025) | 1% of policies (severe repetitive loss) account for 30% of all payouts |

### 4.2 NFIP Claims as Ground Truth -- Known Problems

Beyond low penetration, NFIP claims data has structural problems:

1. **Coverage caps distort loss ratios.** The 50% "substantial damage" threshold in NFIP creates an artificial ceiling on recorded loss ratios.
2. **Sandy claims fraud.** Congressional hearings (CHRG-114) documented that engineering reports submitted for Sandy claims were fraudulently altered. FEMA settled 1,631 of 1,633 court cases for ~$164M.
3. **Post-disaster policy churn.** Kousky (2017) showed insurance uptake spikes post-disaster then drops to baseline within 3 years. This creates *temporal heterogeneity* in what NFIP claims measure.
4. **Risk Rating 2.0 structural break.** Since October 2021, NFIP pricing methodology changed fundamentally, causing 6% policy decline in <1 year. This is a structural break in the claims-generating process.
5. **Community Rating System (CRS) distortion.** CRS provides premium discounts for community mitigation activities. Participation varies by community, creating spatial heterogeneity in insurance uptake incentives.

### 4.3 FEMA Individual Assistance as Complement

Research combining NFIP + FEMA IA data (primarily in Harris County, TX context):

- IA captures uninsured/underinsured losses but does not aim for full compensation.
- In Harris County: 41,606 NFIP records vs. 66,366 IA records -- IA captures ~60% more structures.
- NFIP and IA can be spatially joined to building polygons for cross-validation.
- Records with zero damage values must be excluded (zero = "not related to flooding" rather than "no damage").

### 4.4 What This Means for FloodRSCT

Our finding that New Orleans NFIP has R=0.000 is *consistent with the literature* but *no one else frames it this way*. The literature says:
- DDFs have negative R-squared (Wing et al. 2020)
- NFIP data are "messy" (Wing et al. 2020)
- Insurance penetration is catastrophically low in many areas
- Post-Katrina claims processing in New Orleans was uniquely compromised

**No one in the literature treats R=0.000 as a diagnostic signal.** Instead, they treat it as noise to be worked around. This is the core novelty of FloodRSCT.

---

## 5. Multi-Source Data Fusion

### 5.1 Current State

The literature on multi-source flood data fusion is active but fragmented:

| Approach | Sources Fused | Resolution | Focus |
|----------|--------------|------------|-------|
| **iClaim** (Yang et al. 2022) | NFIP + satellite flood extent + precip + river stage | Grid/county | Claim count prediction |
| **ALTIS** (Vinaykumar & Kamasani, 2026) | Sentinel-1 SAR + DEM + NFIP DDFs + parcel footprints | Property-level | Insurance triage |
| **Multi3Net** (Rudner et al. 2018) | Multi-resolution, multi-sensor satellite imagery | Building-level | Flood damage segmentation |
| **SAR + Social Media** (various, 2025) | Sentinel SAR + geotagged tweets | Pixel/block | Flood extent |
| **FloodGenome** (Liu & Mostafavi, 2025) | NFIP + topography + hydrology + built environment | Census block group | Risk classification |

### 5.2 What Nobody Has Done

**No published work combines NFIP claims + FEMA IA + Hazus estimates + remote sensing + spatial features in a single framework with explicit data quality evaluation per source.**

The closest is ALTIS (2026), which fuses SAR with NFIP-calibrated DDFs, but it does not assess data quality -- it assumes the calibration is valid.

### 5.3 ALTIS (2026) -- Insurance-Grade Flood Triage

**Vinaykumar, A. & Kamasani, P. (2026)**, "ALTIS: Automated Loss Triage and Impact Scoring from Sentinel-1 SAR for Property-Level Flood Damage Assessment," arXiv:2603.13803. [ABSTRACT-ONLY]

Novel contributions:
- Defines "Insurance-Grade Flood Triage (IGFT)" as a new task.
- Introduces two insurance-aligned metrics: Inspection Reduction Rate (IRR) and Triage Efficiency Score (TES).
- Key insight: "A system with superior map accuracy (high IoU) can produce inferior triage performance if its false positives are concentrated on high-value properties."
- Pipeline calibrates DDFs against NFIP claims -- *but does not question whether NFIP claims are valid ground truth*.

---

## 6. Does Anything Like RSCT's Certificate Framework Exist?

### 6.1 Short Answer: No.

No published work in the flood damage literature uses a certificate-based approach to evaluate data source quality with explicit Relevance-Stability-Novelty decomposition. The closest analogues exist in adjacent fields:

### 6.2 Adjacent Frameworks

| Framework | Domain | What It Does | How It Differs from RSCT |
|-----------|--------|-------------|-------------------------|
| **Model Cards** (Mitchell et al. 2019) | ML governance | Documents model performance, limitations, intended use | Static documentation, not a computed signal. No per-source quality decomposition. |
| **Datasheets for Datasets** (Gebru et al. 2021) | ML governance | Documents dataset provenance, composition, biases | Human-authored narrative, not machine-computed. No concept of R/S/N decomposition. |
| **Data Nutrition Labels** (Holland et al.) | Data governance | Standardized dataset summary statistics | Descriptive statistics, not predictive validity assessment. |
| **DQMAF** (2025, MDPI) | Data quality | ML-driven quality profiling (completeness, consistency, duplication) | Evaluates data attributes, not predictive relevance to a task. |
| **US Patent 11,321,286** | Data quality | "Systems and methods for data quality certification" -- generates origin reliability + subjective evaluation + variance ratings | Closest conceptually! But general-purpose, not flood/domain-specific, and does not decompose into R/S/N. |
| **FactSheets** (IBM, Arnold et al. 2019) | AI governance | Supplier declarations of conformity for AI services | Process documentation, not signal quality assessment. |

### 6.3 The Structural Break Detection Literature

**LSTM-Based Detection of Structural Breaks in Property Insurance Loss Reserving** (arXiv, 2025) is relevant:
- Identifies four structural break drivers: severity shocks, frequency changes, development pattern shifts, claims handling disruptions.
- Chain Ladder requires 3-5 development periods to detect regime shifts.
- Reserve errors exceeding 30% documented following major catastrophes.

This paper treats structural breaks as an actuarial problem to be detected and corrected. FloodRSCT's Stability (S) signal does something analogous but treats temporal instability as a *certificate property* of the data source itself.

### 6.4 The Nayak et al. (2025) Hyperclustering Framework

**Nayak, Gentine & Lall (2025)**, "Catastrophic 'hyperclustering' and recurrent losses: diagnosing U.S. flood insurance insolvency triggers," *npj Natural Hazards*, 2(83). [ABSTRACT-ONLY]

- Uses unsupervised ML + game theory to diagnose NFIP insolvency triggers.
- Identifies "hyperclustering" (large-scale correlated flood events) and recurrent loss patterns.
- All eight 99.9th percentile events occurred in the 21st century.
- Analyzes 2M+ claims and 80M policy records.
- Code available: github.com/adamnayak/Hyperclusters

**Relevance:** Nayak et al. diagnose *insurance portfolio-level* patterns, not *data quality per source per geography*. Their "hyperclustering" concept parallels our finding that certain events (Katrina, Sandy, Harvey) create data regime changes, but they don't formalize this as a signal about data validity.

---

## 7. Key Gaps the Literature Leaves Open -- And Where FloodRSCT Sits

### Gap 1: No Data Quality Governance for Flood Damage Sources

Every ML paper in this space treats NFIP claims as ground truth. Wing et al. (2020) showed this is problematic (negative R-squared for DDFs), but the response has been to *build better models* rather than to *certify data source quality*. Nobody asks: "Is NFIP data even *relevant* to flood damage prediction in this specific geography?"

**FloodRSCT's contribution:** R=0.000 for New Orleans NFIP is not a model failure -- it is a *data validity signal*. The certificate framework makes this distinction explicit.

### Gap 2: No Temporal Stability Assessment

Sandy's SCRP distorted NYC's NFIP claims data through structural breaks in the claims-generating process. Risk Rating 2.0 (2021) created another structural break in all NFIP data. No published framework formally assesses temporal stability of flood damage data sources.

**FloodRSCT's contribution:** The Stability (S) component of the certificate explicitly measures temporal consistency. NYC's S-degradation at the temporal representation level directly captures what the actuarial literature treats as a "reserve error."

### Gap 3: No Cross-Source Consistency Check

Houston shows anti-correlation between NFIP claims and Hazus damage estimates. The literature documents that Hazus overestimates at shallow depths and underestimates at deep depths (Wing et al. 2020), and that NFIP penetration in Harris County is ~15%. But nobody asks: "What does it mean when two authoritative sources *disagree* at the sign level?"

**FloodRSCT's contribution:** Multi-source certificate comparison. When two sources yield opposite signals, at least one has R approaching zero for the target. The certificate framework makes this diagnosable.

### Gap 4: Validation Gap in Published Models

Gerl et al. (2016) found ~50% of flood loss models lack validation. FloodGenome achieves high ROC-AUC on the same MSAs where we find data corruption. The literature does not have a mechanism to distinguish "high accuracy on valid data" from "high accuracy on corrupted data."

**FloodRSCT's contribution:** Certificates are computed per-source, per-geography, per-time-period. A model can have high accuracy on Houston NFIP but the certificate reveals whether that accuracy is meaningful or an artifact of data corruption in the target.

### Gap 5: The Uninsured Loss Problem Is Unsolved

70% of flood losses are uninsured. ML approaches (e.g., random forest estimators) attempt to fill this gap but have no mechanism to evaluate whether their estimates are valid. The "ground truth" for uninsured damage simply does not exist.

**FloodRSCT's contribution:** The Novelty (N) component captures how much of the prediction comes from sources that cannot be validated against historical data. High N with low R flags predictions that extend beyond the evidence base.

---

## Summary Table: Key Papers and Their Relationship to FloodRSCT

| Paper | Year | Venue | Key Finding | Evidence Level | RSCT Relevance |
|-------|------|-------|-------------|----------------|----------------|
| Merz et al. | 2010 | NHESS | Depth alone is insufficient for damage estimation | ABSTRACT-ONLY | Defines the problem FloodRSCT addresses |
| Gerl et al. | 2016 | PLOS ONE | 50% of loss models lack validation | ABSTRACT-ONLY | Motivates certificate-based validation |
| Wing et al. | 2017 | WRR | 30m CONUS flood model, 78% agreement with FEMA | TRAINING-RECALL | Physics baseline we compare against |
| Wagenaar et al. | 2017 | NHESS | Multi-variable models reduce MAE by 20% | ABSTRACT-ONLY | ML improvement over DDFs |
| Schroter et al. | 2018 | Env. Mod. Soft. | 3D city models + RF for flood loss | ABSTRACT-ONLY | Alternative feature engineering |
| Mosavi et al. | 2018 | Water | ML flood prediction survey | ABSTRACT-ONLY | Literature context |
| Wing et al. | 2020 | Nature Comms | DDFs have negative R-squared against NFIP claims | VERIFIED | **Core evidence for R=0.000 validity** |
| Wagenaar et al. | 2020 | NHESS | ML will transform flood risk assessment; data scarcity is bottleneck | VERIFIED | Identifies gap FloodRSCT fills |
| Mobley et al. | 2021 | NHESS | RF flood hazard mapping, 3x more structures at risk than FEMA identifies | ABSTRACT-ONLY | FEMA map inadequacy |
| Yang/Shen et al. | 2022 | BAMS | iClaim: fused NFIP+satellite, R-sq>0.95 at event level | ABSTRACT-ONLY | Multi-source fusion, but no quality check |
| Liu et al. | 2024 | IJDRR | FloodDamageCast: GAN+LightGBM for damage nowcasting | ABSTRACT-ONLY | Competitor approach |
| Liu & Mostafavi | 2025 | ERI&S | FloodGenome: same 4 MSAs, assumes NFIP = ground truth | VERIFIED | **Direct comparison target** |
| Nayak et al. | 2025 | npj Nat. Haz. | NFIP hyperclustering drives insolvency; 1% policies = 30% losses | ABSTRACT-ONLY | NFIP structural problems |
| Vinaykumar & Kamasani | 2026 | arXiv | ALTIS: SAR-to-insurance triage pipeline | ABSTRACT-ONLY | Insurance-grade metrics, no data quality |
| LSTM structural breaks | 2025 | arXiv | Structural break detection in insurance reserves | ABSTRACT-ONLY | Parallels Stability signal |

---

## Confidence Assessment

| Sub-question | Confidence | Notes |
|-------------|------------|-------|
| 1. Merz taxonomy | HIGH | Well-established, multiply confirmed |
| 2. Physics/empirical/ML comparison | HIGH | Wing et al. 2020 is definitive on DDF failures |
| 3. ML approaches | HIGH | FloodGenome, FloodDamageCast, iClaim well-documented |
| 4. Ground truth problem | HIGH | 70% uninsured gap is multiply confirmed |
| 5. Multi-source fusion | MEDIUM | Active but fragmented; no unified NFIP+IA+RS framework found |
| 6. Certificate-based approach | HIGH (negative) | Nothing like RSCT certificates exists in flood literature |
| 7. Literature gaps | HIGH | Clear gaps identified, directly addressable by FloodRSCT |

---

## Methodological Note

This report was produced by systematic web search of academic databases, followed by full-text reading where accessible. Evidence levels are marked as:
- **VERIFIED**: Full text (or substantial sections) read during this session
- **ABSTRACT-ONLY**: Only abstract, search snippets, or metadata reviewed
- **TRAINING-RECALL**: Known from pre-training knowledge, not re-verified in this session

Sources that could not be fully read (paywalled, PDF parsing failures) are marked accordingly. Claims from ABSTRACT-ONLY sources should be treated as provisional until full-text verification.
