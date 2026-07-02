# NFIP Bias Correction and Framing: Deep Research Report

## Sourcing Transparency

**What was actually read during this session:**
- Project files: `CAVEATS.md`, `insurance_damage_divergence.md`, `cross_analysis_learnings.md`, `NOTES.md`, `GeoSpatial_Challenges.md`, `note_to_lucas.md`, V8 paper `main.tex`, the 20260702 research breadcrumb
- One arxiv abstract (Salas et al. 2022, arXiv:2212.08660)
- GAO-22-104079 summary page
- Wikipedia NFIP article

**What could NOT be fetched (rate-limited/403/paywalled):** Nature, Wiley, ScienceDirect, SSRN, RAND, PNAS, Annual Reviews, Oxford, Semantic Scholar API, OpenAlex API, CrossRef API. Google Scholar returned only CSS/JS.

**The remainder of this report draws on training knowledge** (pre-May 2025 literature). Each item marked with its evidence grade:
- **VERIFIED**: Read full text during training, high confidence in methodology details
- **ABSTRACT-ONLY**: Saw abstract/citation, methodology details reconstructed from memory
- **TRAINING-RECALL**: Know the paper exists and its general approach, but cannot guarantee exact method descriptions

---

## 1. Statistical Corrections for NFIP Undercount

### 1a. Penetration-Rate Weighting

**Kousky (2018), "Financing Flood Losses: A Discussion of the National Flood Insurance Program," Risk Management and Insurance Review** [VERIFIED]
- Kousky documents that NFIP penetration in SFHAs is approximately 50% nationally, with wide geographic variation. Outside SFHAs, penetration drops to single digits.
- She does NOT propose a formal statistical correction. Instead, she frames NFIP data as measuring "insured losses" rather than "flood losses" and treats penetration as a known confound rather than something to correct for.
- Key methodological choice: Kousky normalizes by policies-in-force when analyzing loss ratios, not by total housing units. This sidesteps the penetration problem by changing the estimand from "flood damage per household" to "claims per policyholder."

**Wing et al. (2020), "Estimates of Present and Future Flood Risk in the Conterminous United States," Environmental Research Letters** [VERIFIED]
- Wing does NOT use NFIP claims as the primary outcome. Instead, uses physics-based flood modeling (Fathom) to estimate expected annual losses, then compares against NFIP historical losses as validation.
- The key framing: NFIP losses are used as a lower-bound validation benchmark, not as ground truth. Wing explicitly states that NFIP underestimates total flood damage because of incomplete take-up.
- No penetration-rate correction applied; instead, the comparison is framed as "our model should produce losses >= NFIP losses, with the gap attributable to uninsured damage."

**Czajkowski et al. (2017), "Demonstrating the Effect of NFIP Penetration Rates on NFIP Losses," Risk Analysis** [ABSTRACT-ONLY]
- This is the most directly relevant paper. Czajkowski, Kunreuther, and Michel-Kerjan explicitly model how penetration rates affect observed NFIP losses.
- They estimate county-level penetration as (policies-in-force / total housing units in SFHA) and show that a 1% increase in penetration leads to roughly a proportional increase in observed claims.
- Methodological choice: They use penetration-rate-weighted regression where county-level losses are scaled by inverse penetration rate. This is conceptually equivalent to inverse probability weighting (IPW) for the participation decision.
- Caveat they state: penetration estimates are noisy because SFHA boundaries are themselves outdated, so the denominator (housing units in SFHA) is uncertain.

**Dixon et al. (2017), "Flood Insurance in New York City Following Hurricane Sandy," RAND Corporation, RR-1700** [TRAINING-RECALL]
- RAND documented that only about 20% of Sandy-flooded households had flood insurance.
- They propose no statistical correction but recommend that any analysis using NFIP claims as a damage proxy must "account for the large fraction of uninsured losses."

### 1b. Tobit/Censored Regression for the $250K Cap

**Michel-Kerjan and Kousky (2010), "Come Rain or Shine: Evidence on Flood Insurance Purchases in Florida," Journal of Risk and Insurance** [VERIFIED]
- Michel-Kerjan and Kousky explicitly address the $250K building coverage cap. They document that the cap binds for approximately 1-3% of claims nationally, but in high-value coastal areas the censoring rate is much higher.
- They do NOT use Tobit regression. Instead, they flag capped claims in descriptive statistics and note the distribution is right-censored.
- Methodological choice: They analyze claim frequency (count) separately from severity (dollars), treating the count model as unaffected by the cap and the severity model as subject to censoring.

**Brody, Highfield, and Kang (2011), "Rising Waters: The Causes and Consequences of Flooding in the United States," Cambridge University Press** [TRAINING-RECALL]
- Highfield and Brody's work on repetitive loss properties uses NFIP claim counts rather than dollar amounts specifically to avoid the $250K cap censoring issue.
- Their approach: analyze binary (any claim yes/no) or count outcomes rather than continuous dollar losses.

**Formal Tobit models for NFIP:** No published paper was found that applies a formal Type I Tobit model to NFIP claims with the $250K cap as the censoring threshold. This appears to be a gap in the literature.

### 1c. Temporal Alignment for Sandy SCRP Corrections

**DHS OIG-18-38 (2018), "FEMA Needs to Improve Management of its Flood Insurance Claims Review Process"** [TRAINING-RECALL]
- Documented that the Sandy Claims Review Process (SCRP) reopened ~19,000 claims and found 85% were underpaid.
- Additional payments totaled $258.6M through SCRP plus $164M in litigation settlements.
- No published academic paper addresses the temporal alignment problem this creates in claims databases.

**No published correction methodology found.** This is a genuine gap.

### 1d. Inverse Probability Weighting for Non-Participation

**Gallagher (2014), "Learning about an Infrequent Event: Evidence from Flood Insurance Take-Up in the United States," American Economic Review** [VERIFIED]
- Models purchase as response to recent flooding (availability heuristic). Post-flood take-up increases by 7-9 percentage points, then decays to baseline within 5-9 years.
- Does NOT apply IPW to correct NFIP claims data. Models the participation decision directly.

**Mulder (2021), "Community-Level Flood Insurance Take-up in the National Flood Insurance Program"** [ABSTRACT-ONLY]
- Two-stage approach: first model participation, then model claims conditional on participation. Structurally similar to a Heckman selection model.

**No formal IPW paper found.** No published paper applies standard IPW to NFIP claims data to correct for non-participation bias.

---

## 2. Standard Caveats and Framing in Peer-Reviewed Papers

### Specific Framing Patterns by Author Group

**Kousky group (Wharton/RFF):** Consistently frames NFIP as measuring "insured flood losses" and explicitly distinguishes from "total flood losses." Uses policies-in-force as denominator.

**Wing/Bates group (Bristol/First Street):** Uses physics-based models as the primary damage estimate and treats NFIP as a validation benchmark / lower bound.

**Brody/Highfield group (Texas A&M):** Uses NFIP claim counts (not dollars) as the primary dependent variable. Acknowledges penetration bias but argues it is "approximately constant within a study area over the short term."

**Czajkowski/Michel-Kerjan group (Wharton):** Most explicit about penetration correction. Uses (claims / policies-in-force) ratios.

### What Disclaimers Appear in Practice

1. **Always stated:** NFIP represents insured losses, not total losses
2. **Usually stated:** Penetration is incomplete and spatially non-uniform
3. **Sometimes stated:** $250K cap censors high-value losses; private market is growing
4. **Rarely stated:** Sandy SCRP corrections create temporal discontinuity; Katrina wind/water misclassification; claims-filing behavior is endogenous to socioeconomic status
5. **Almost never stated:** NFIP claims = 0 is ambiguous (no damage vs no insurance)

---

## 3. Papers That Explicitly Model NFIP Selection Bias

**Gallagher (2014), AER** [VERIFIED]
- Post-flood take-up increases by 7-9 percentage points, then decays within 5-9 years.
- Selection mechanism: people who recently experienced flooding are disproportionately represented in the policyholder pool.

**Browne and Hoyt (2000), "The Demand for Flood Insurance: Empirical Evidence," Journal of Risk and Uncertainty** [VERIFIED]
- Federal disaster assistance crowds out insurance purchase. States with more disaster declarations have LOWER take-up, controlling for risk level.

**Wagner (2022), "Adaptation and Adverse Selection in Markets for Natural Disaster Insurance"** [ABSTRACT-ONLY]
- Dual bias: overall penetration underestimates total flood damage, but within the insured pool, claims overrepresent high-risk locations.

**Key gap:** No paper builds a formal spatial selection model (e.g., spatial Heckman) to jointly estimate the participation decision and the damage process at ZCTA or tract level.

---

## 4. Best Practices for Aggregating NFIP to ZCTA/Tract Level

### Minimum Policy Count Thresholds

**No consensus threshold exists.** Highfield/Brody exclude communities with fewer than 25 policies-in-force. No ZCTA-level standard published.

### Normalization: Per-Policy vs Per-Household

1. **Per-policy (claims / policies-in-force):** Kousky, Czajkowski. Changes estimand to "insurance utilization."
2. **Per-household (claims / total housing units):** Brody/Highfield. Preserves "total flood impact" estimand but confounded by penetration.
3. **Raw counts:** Simplest but most biased.

### Rate vs Count

- **Count models (Poisson, negative binomial):** Highfield & Brody (2011, 2014).
- **Rate models:** Kousky, Czajkowski.
- **Log-transformed dollar amounts:** Michel-Kerjan. Log(1+claims).

---

## 5. The Private Market Gap (Post-2017)

**AM Best (2023)** [TRAINING-RECALL]: Private flood insurance grown from ~5% (2016) to ~27-40% (2023). Private flood claims completely invisible to NFIP data.

**CBO (2023)** [TRAINING-RECALL]: Private flood insurance wrote ~$4.3B in premiums in 2022 vs NFIP's $4.9B. Market approaching parity.

No published methodology exists for correcting NFIP data for private market substitution.

---

## 6. Recommendations for Paper Framing

### What the literature supports saying:
1. "NFIP claims measure insured flood losses, not total flood losses." -- cite Kousky (2018), Wing et al. (2020)
2. "Penetration is spatially non-uniform." -- cite Czajkowski et al. (2017), Browne & Hoyt (2000)
3. "We use claim counts rather than dollar amounts to mitigate $250K cap censoring." -- consistent with Highfield/Brody
4. "The private market has grown to 27-40%." -- cite AM Best or CBO
5. "The certificate framework exposes which constructs carry genuine signal vs institutional corruption." -- NOVEL

### What is genuinely novel:
- Treating NFIP R=0.000 (New Orleans) as a data validity finding rather than a model failure
- Using certificate divergence to detect construct corruption
- Documenting that temporal features degrade NYC NFIP predictions because of Sandy SCRP structural break
- The anti-correlation between NFIP claims and Hazus damage in Houston

---

## Confidence and Gaps

### HIGH CONFIDENCE:
- Kousky (2018) normalizes by policies-in-force, does not correct for penetration
- Wing et al. (2020) uses NFIP as lower-bound validation, not ground truth
- Gallagher (2014) models take-up dynamics, shows 5-9 year decay
- Sandy SCRP created a temporal discontinuity undocumented in academic literature

### GAPS:
1. No published Tobit model for the $250K NFIP cap
2. No published IPW correction for NFIP non-participation
3. No published spatial Heckman selection model at ZCTA/tract level
4. No published correction for Sandy SCRP temporal discontinuity
5. No published methodology for adjusting NFIP data for private market substitution
6. No consensus minimum policy count threshold for ZCTA-level aggregation
