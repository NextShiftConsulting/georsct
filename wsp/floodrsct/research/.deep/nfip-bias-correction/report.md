# Statistical Correction of NFIP Claims Data Biases for Flood Damage Estimation

**Research Date:** 2026-07-02
**Scope:** Methods to correct known biases in NFIP claims when used as a proxy for actual flood damage at ZCTA level

---

## Executive Summary

NFIP claims data is the most widely used proxy for flood damage in the United States, but it is systematically biased through at least seven documented mechanisms. This report surveys the statistical literature on correcting each bias class and synthesizes actionable methods for the FloodRSCT framework. The key finding is that **no single correction method addresses all seven biases simultaneously** -- a layered correction pipeline is required, combining econometric selection models, actuarial credibility weighting, Tobit censoring corrections, structural break detection, and multi-source composite estimation.

---

## Sub-question 1: Statistical Methods for Insurance Selection Bias Correction

**Finding:** Three families of methods are applicable: (a) Heckman-type selection models, (b) propensity score / inverse probability weighting, and (c) domain-specific covariate shift corrections. None has been applied directly to NFIP selection bias for damage estimation at ZCTA level, but all have close analogs in the flood damage literature.

### Evidence

#### 1a. Heckman Two-Step Selection Model

The Heckman correction (Heckman, 1979) is the canonical econometric approach for non-random sample selection. It models two equations jointly:

1. **Selection equation** (probit): models the probability of having NFIP insurance as a function of covariates (flood zone designation, mortgage status, income, risk perception, community participation in NFIP)
2. **Outcome equation**: models flood damage conditional on being insured, with the inverse Mills ratio from the selection equation as an additional regressor to correct for selection bias

**Applicability to NFIP:** The selection equation would model NFIP take-up as a function of:
- SFHA designation (mandatory purchase requirement for federally-backed mortgages)
- Mortgage status (mandatory vs. voluntary purchase)
- Community Rating System (CRS) class
- Income / property value
- Prior flood experience
- Risk Rating 2.0 premium level

The key requirement is an **exclusion restriction** -- a variable that affects insurance purchase but not damage magnitude. Candidates include: mortgage origination timing, lender enforcement intensity, CRS discount availability, and distance to insurance agent offices.

**Limitations:** "Common problems with its application include its use with dichotomous dependent variables, difficulties with calculating the hazard rate, mis-estimated standard error estimates, and collinearity between the correction term and other regressors" (Bushway et al., cited in CCJS review). The two-step estimator is consistent but inefficient relative to MLE; however, it is "generally more stable when the data are problematic" (Stata documentation).

#### 1b. Propensity Score Matching and Inverse Probability Weighting

Hudson et al. (2014) applied propensity score matching (PSM) to flood damage data from German households flooded in 2002, 2005, and 2006, published in *Natural Hazards and Earth System Sciences*. They found that "bias-corrected effectiveness estimates detected substantial overestimates of mitigation measures' effectiveness if bias is not controlled for, ranging from nearly EUR 1700 to 15,000 per measure" (Hudson et al., 2014).

**Inverse Probability Weighting (IPW)** is preferable to PSM for NFIP correction because:
- It retains all observations rather than discarding unmatched units
- It can weight NFIP-insured properties by 1/P(insured) to create a pseudo-population representative of all properties
- Doubly robust (AIPW) estimators combine propensity score and outcome models, remaining consistent if either model is correctly specified

**Practical considerations:**
- Weights should be stabilized and trimmed at the 1st/99th percentiles to avoid extreme values
- The propensity score model requires individual-level data on insurance purchase decisions, which is available through OpenFEMA policy data cross-referenced with parcel-level property records

#### 1c. Covariate Shift / Domain Adaptation Methods

Wagenaar et al. (2020) published "Improved Transferability of Data-Driven Damage Models Through Sample Selection Bias Correction" in *Risk Analysis*, 41(1), 37-55. They applied three methods:

1. **Kernel Mean Matching (KMM):** Assigns weights to training data so independent variable means match between training and test distributions
2. **Cluster-Based Estimation (CBE):** Weights training clusters proportionally to match test cluster frequencies using CWx = (Nx,test/Ntest) x (Ntrain/Nx,train)
3. **Single Variable Distribution Matching (SVDM):** Matches on the most critical variable (water depth for floods)

Results showed "Mean Bias Error reductions exceeded 30% in macromodels and reached 50% in micromodels" (Wagenaar et al., 2020).

**Source quality:** The Wagenaar et al. paper is a peer-reviewed primary source published in a top risk analysis journal. The Hudson et al. PSM paper is peer-reviewed in NHESS. The Heckman method is foundational econometrics (Nobel Prize, 2000). No study was found applying any of these methods specifically to NFIP selection bias at ZCTA level -- this is a genuine gap.

### Recommended Approach for FloodRSCT

Use a **two-stage IPW correction**:

1. **Stage 1 (Selection model):** Estimate P(NFIP_insured | X) at the property or ZCTA level using:
   - SFHA status from FEMA NFHL
   - Median household income from ACS
   - Mortgage prevalence from HMDA
   - CRS class from NFIP community data
   - Historical flood frequency from NOAA

2. **Stage 2 (Weighted estimation):** Weight observed NFIP claims by 1/P(insured) to estimate population-level damage. Use AIPW (doubly robust) if an outcome model is also specified.

---

## Sub-question 2: Actuarial Approaches to Undercounting and Credibility Weighting

**Finding:** Actuarial science addresses undercounting through (a) catastrophe model-based loss amplification, (b) credibility weighting that blends sparse individual experience with broader portfolio data, and (c) explicit adjustment for coverage-limit censoring via Tobit-type models. ASOP No. 39 is the governing standard.

### Evidence

#### 2a. Catastrophe Model Adjustment Factors

Post-Hurricane Andrew (1992), the insurance industry recognized that "calculations based strictly on historical losses may underestimate projected losses" (NAIC Catastrophe Modeling Primer, 2025). Modern catastrophe models address undercounting through:

- **Demand surge factors:** Post-disaster inflation in construction costs (typically 20-40% for major events)
- **Loss amplification factors:** Accounting for secondary effects (evacuation costs, business interruption, sociological factors) identified after Hurricane Katrina
- **Exposure growth trending:** "Losses for each year are converted to current values by multiplying by a trend factor that accounts for change in exposure over time and inflation by zip code" (FEMA Risk Rating 2.0 Methodology)

The Actuarial Standard of Practice No. 39 (ASOP 39) requires actuaries to "address biases by adjusting the historical data used to form future cost estimates and determining a provision for catastrophe losses" including "limiting losses in the underlying data and using increased limits factors or excess loss factors based on industry data" (Actuarial Standards Board, 2014).

#### 2b. Credibility Weighting (Buhlmann-Straub)

The Buhlmann credibility formula Z = N / (N + K) provides the optimal weight for blending individual experience with population-level data, where:
- N = exposure measure (e.g., number of claims in a ZCTA)
- K = ratio of within-group variance to between-group variance

For ZCTA-level flood damage estimation, this means:
- ZCTAs with many claims get high credibility weight on their own data
- ZCTAs with few claims are pulled toward the broader regional or national average
- The credibility parameter K can be estimated from the data using ANOVA-type decomposition

**Application to FloodRSCT:** Use Buhlmann-Straub credibility to blend ZCTA-level NFIP claims experience with:
- Cat model output (AIR, RMS, or HAZUS-based expected losses)
- Regional claims averages from similar ZCTAs
- Physical hazard model predictions

This naturally handles the sparse-data problem in low-penetration ZCTAs.

#### 2c. Tobit Correction for Coverage Cap Censoring

NFIP coverage is capped at $250,000 for residential buildings and $100,000 for contents. FEMA's own data shows that "61% of NFIP-insured single-family residences have $250,000 of building coverage" (FEMA, Coverage Limits Document), indicating the cap binds for a majority of policyholders.

This creates **right-censoring** in the claims data: observed claims are truncated at the coverage limit, understating true losses for properties with damage exceeding $250,000.

The **Tobit model** (Tobin, 1958) directly addresses this:
- Models a latent (true) damage variable Y* that is observed as Y = min(Y*, coverage_limit)
- Estimates both the probability of exceeding the limit and the expected value of the latent variable
- "OLS coefficients are attenuated when the outcome is censored. The stronger the censoring, the worse the bias. Tobit corrects it by estimating the limit-aware likelihood directly" (R-statistics.co)

For FloodRSCT, a Tobit model would:
1. Use claims data where Y = min(actual_damage, $250,000) as the dependent variable
2. Include property value, flood depth, and structure characteristics as regressors
3. Estimate the latent damage distribution, recovering the full (uncensored) loss distribution

**Advanced variant:** Use a **censored regression with truncation adjustment** (tcensReg) that simultaneously handles the coverage cap (right-censoring) and the deductible (left-truncation, since small losses below the deductible are not claimed).

**Source quality:** ASOP 39 is the authoritative actuarial standard. Credibility theory is textbook actuarial science (Buhlmann, 1967; Buhlmann & Straub, 1970). The coverage cap data is from FEMA's own publications.

---

## Sub-question 3: NFIP Penetration Rate Data at Sub-County Resolution

**Finding:** FEMA publishes NFIP penetration rates at the county level through OpenFEMA, and individual policy records are available at census tract resolution through the Redacted Policies dataset. No pre-computed ZCTA-level penetration rate dataset exists, but it can be constructed from OpenFEMA microdata.

### Evidence

#### 3a. FEMA OpenFEMA Penetration Rate Dataset

The **NFIP Residential Penetration Rates v1** dataset provides "the ratio of insured residential structures to total residential structures in that area" (FEMA, OpenFEMA). Key limitations:
- Available at **county level**, not ZIP/ZCTA/census tract
- **Excludes private flood insurance:** "Private flood insurance take-up rates are not available to FEMA and therefore, are not included in these estimates" (FEMA, OpenFEMA)
- Updated as of v1.20 (December 2025)
- Nationally, "an average of only 3.3% of households (or about 4.7 million total) across the country have NFIP coverage" (ValuePenguin, 2025)

#### 3b. Constructing Sub-County Penetration from Microdata

The **FIMA NFIP Redacted Policies v2** dataset contains 80+ million policy transactions with:
- **Census tract** (geocoded, high accuracy)
- **Reported ZIP code** (self-reported, lower accuracy)
- Truncated lat/lon (0.1 degree precision)

FEMA states: "Census tract and county fields are best used for aggregation, because they are derived from a policy or claim geocode... FEMA is confident that these values are reported with a relatively high degree of accuracy" (FEMA, OpenFEMA FAQ).

**Method to construct ZCTA-level penetration:**
1. Count active NFIP policies per census tract from Redacted Policies
2. Obtain total residential housing units per census tract from ACS (Table B25001)
3. Compute penetration = policies / housing_units per census tract
4. Crosswalk census tracts to ZCTAs using HUD USPS crosswalk files or Census relationship files
5. Aggregate to ZCTA using area-weighted or population-weighted interpolation

#### 3c. SFHA-Level Penetration Estimates

Market penetration varies dramatically by flood zone:
- "Flood insurance market penetration rates are around 50 percent for homes within SFHAs, with substantial variation across geographic regions; rates are only about 1 percent outside SFHAs" (RFF, Kousky et al.)
- Florida accounts for roughly 40% of the entire NFIP portfolio (RFF)

#### 3d. Private Market Invisibility

The private flood insurance market adds substantial uncertainty:
- Private insurers held about 27-32% of the flood insurance market by direct premiums written as of 2022-2024 (AM Best, 2024)
- "There is no nationwide database on the companies writing residential flood insurance, coverages offered, policy terms, pricing" (RFF, Emerging Private Market Report)
- Private market share varies by state: less than 25% in Florida and North Carolina, potentially higher in other states

**Implication for bias correction:** Any penetration-based correction using NFIP data alone understates true insurance coverage by 27-40%, meaning the correction factor will over-correct for uninsured losses. The private market gap must be estimated separately, possibly using state insurance regulatory filings or NAIC data.

**Source quality:** OpenFEMA data is primary government source. RFF reports are from a respected research institution with deep NFIP expertise (Carolyn Kousky is a leading NFIP researcher). AM Best market share data is industry-standard. The private market data gap is a genuine, well-documented problem.

---

## Sub-question 4: Detecting and Adjusting for Structural Breaks in NFIP Claims Time Series

**Finding:** Standard econometric structural break tests (Chow, Bai-Perron) are directly applicable to NFIP claims time series. Two major breaks are empirically documented: Risk Rating 2.0 (October 2021) and the Sandy Claims Review (2015-2017). Recent work applies LSTM-based detection for climate-driven breaks.

### Evidence

#### 4a. Known Structural Breaks in NFIP Data

**Risk Rating 2.0 (October 2021 / April 2022):**
Two peer-reviewed studies document this break using causal inference methods:

1. "To improve is to change" (ScienceDirect, 2025): "The reform pushed a minimum of 53,000 homeowners out of the insurance market in the first year... robust to alternative identification strategies, such as regression discontinuity in time and difference-in-difference estimation"

2. Gourevitch et al. (Journal of Catastrophe Risk and Resilience, 2025): "Risk Rating 2.0 has caused an 11-39% decline in new policies and a 5-13% decline in existing policies, depending on the amount premiums have increased"

These studies confirm that RR 2.0 created a **composition shift** in the insured pool -- policyholders facing the largest premium increases were most likely to cancel, meaning post-2021 claims data represents a systematically different (likely lower-risk) population.

**Sandy Claims Review (2015-2017):**
FEMA reopened 144,000 closed civil claims, resulting in "$258 million in additional payments to more than 15,000 NFIP policyholders" (PBS Frontline, 2016). The engineering report fraud involved insurance defense firms altering "factual observations of their field engineers, faking facts to reverse conclusions of causation to deny legitimate storm-damage claims" (Claims Journal, 2019). This creates both:
- An **upward adjustment** in the 2015-2017 period (additional payments)
- A **downward bias** in the original 2012-2015 Sandy claims (systematic underpayment)

**Katrina Wind-vs-Water Fraud (2005-2006):**
In *State Farm v. Rigsby* (2016, Supreme Court), adjusters were "allegedly instructed to misclassify wind damage as flood damage in order to shift State Farm's own liability to the federal government" (Dickinson Law, Penn State). USAA "admitted before the Mississippi Supreme Court that it shifted its own costs to the federal government's National Flood Insurance Program" (Vote Smart). This creates an **upward bias** in Katrina-era NFIP claims (they include wind damage that should not be there) and a corresponding gap in homeowners insurance claims.

**Post-Disaster Policy Churn:**
Research shows "the mandatory purchase requirement increased take-up rates by about 5%, with only an additional 1.5% increase not due to this requirement. Critically, these flood policies may not be maintained -- the bump in policies is gone three years after the disaster" (ScienceDirect, Voluntary Purchases). This creates a time-varying selection bias: the insured population changes composition around each major disaster.

#### 4b. Econometric Break Detection Methods

**Chow Test (Known Break Date):**
When the break date is known a priori (e.g., October 1, 2021 for RR 2.0), the Chow test (1960) splits the sample at that date and tests whether regression coefficients differ between subsamples using an F-statistic.

**Bai-Perron (Unknown Multiple Breaks):**
For detecting unknown or multiple breaks, Bai and Perron (1998, 2003) provide:
- Sequential and simultaneous testing for multiple structural changes
- Dynamic programming algorithm for efficient computation
- Confidence intervals for estimated break dates
- Applicable to NFIP claims time series at national, state, or ZCTA level

**LSTM-Based Detection (Recent):**
A 2025 arXiv paper applies LSTM networks to "15+ years of regulatory development triangle data from Florida and Louisiana, enriched with NOAA hurricane intensity indices and sea surface temperatures" for structural break detection in property insurance loss reserving. The authors "hypothesize a targeted improvement of 15-20% in reserve accuracy for catastrophe-exposed years."

#### 4c. Adjustment Methods Post-Detection

Once breaks are detected, adjustment options include:

1. **Regime-specific models:** Estimate separate damage models for pre/post-break periods, then combine using appropriate weights
2. **Dummy variable approach:** Include break indicators and interaction terms in a single model
3. **Trend adjustment:** Apply FEMA's own "loss trending" methodology, which converts historical losses to current values using zip-code-level inflation and exposure growth factors
4. **Exclusion:** For known fraud periods (Katrina wind-water, Sandy engineering), consider excluding affected claims or applying forensic correction factors

**Source quality:** The DiD studies on RR 2.0 are peer-reviewed primary research. The Sandy fraud documentation comes from congressional hearings (CHRG-114hhrg96989), Supreme Court proceedings, and investigative journalism (CBS News, PBS Frontline). The Bai-Perron methodology is foundational econometrics with extensive textbook treatment.

---

## Sub-question 5: Composite Estimation from Multiple Biased Damage Sources

**Finding:** No published framework specifically addresses combining NFIP + FEMA IA + SBA into a bias-corrected composite flood damage estimate. However, several methodological approaches from adjacent fields are directly applicable: multiple imputation with proxy variables, capture-recapture estimation, and Bayesian data fusion.

### Evidence

#### 5a. Coverage and Bias Profile of Each Source

| Source | Coverage | Trigger | Geographic Resolution | Key Bias |
|--------|----------|---------|----------------------|----------|
| NFIP Claims | Insured properties only (~4.7M policies) | Any flood | Census tract (OpenFEMA) | Low penetration, coverage caps, selection bias |
| FEMA IA (IHP) | Uninsured/underinsured individuals | Presidential Disaster Declaration only | Census tract (OpenFEMA) | Requires disaster declaration, racial/income disparities in approval |
| SBA Disaster Loans | Homeowners/renters who apply and qualify | Major disasters | ZIP code | Requires creditworthiness, application burden |

The aggregate coverage gap is enormous: "Ex ante and ex post payments amount to 12% of total flood costs -- meaning that about 88% of the costs of flood damage are uninsured" (Solomon, Optimal Flood Insurance). Specifically, "NFIP payouts average $3,440, FEMA assistance $733 per house, and SBA loans $704" per flood-weighted household (Solomon).

#### 5b. FEMA IA Bias Issues

FEMA Individual Assistance has documented racial and income-based disparities:
- "FEMA's individual assistance denial rate was four times higher for low-income applicants than it was for high-income applicants" (Texas Housers, post-Harvey)
- "Only 34 percent of all white residents who sought federal assistance said their application passed muster, that figure dropped to 28 percent for Hispanic residents and just 13 percent for black residents" (Facing South)
- FEMA IA only activates for declared disasters, missing smaller/localized flood events entirely

#### 5c. Composite Estimation Methods

**Method 1: Multiple Imputation with Linked Proxy Outcomes**

Studies show that "incorporating a linked proxy for the missing outcome as an auxiliary variable reduced bias and increased efficiency in all scenarios, even when 80% of the outcome was missing" and "high correlations (>0.5) between the outcome and its proxy substantially reduced the missing information" (BioMed Central, 2017).

For FloodRSCT, each data source serves as a proxy for the latent "true damage":
1. Treat true_damage as the partially observed outcome
2. Use NFIP claims, FEMA IA payments, and SBA loan amounts as auxiliary/proxy variables
3. Apply multiple imputation (MICE framework) to estimate the full damage distribution
4. The correlation structure between sources provides information about the missing data mechanism

**Method 2: Capture-Recapture / Multiple Systems Estimation**

Capture-recapture methods "estimate the size of closed populations based on matched incomplete samples" by "analyzing the patterns of inclusion of individuals across samples to estimate the probability of not being observed" (Annual Reviews, 2018). Applied to flood damage:
1. Match properties across NFIP claims, FEMA IA registrations, and SBA applications using address/parcel ID
2. Properties appearing in multiple sources provide overlap information
3. Properties appearing in only one source reveal source-specific capture probabilities
4. The Lincoln-Petersen estimator (for two sources) or log-linear models (for three+ sources) estimate total affected properties
5. Multiply by average damage per property to estimate total losses

**Assumptions required:** Independence of capture across sources (likely violated -- FEMA IA referrals to SBA create dependence), and a closed population (all damaged properties exist throughout the observation period).

**Method 3: Bayesian Hierarchical Data Fusion**

Combine sources in a Bayesian framework:
1. Model each source's reporting probability as a function of property/area characteristics
2. Place a prior on the latent "true damage" distribution
3. Update using likelihood contributions from each observed source
4. The posterior integrates information from all sources while accounting for source-specific biases

This approach naturally handles:
- Different geographic resolutions across sources
- Source-specific censoring (NFIP coverage caps, FEMA IA payment limits)
- Missing-not-at-random patterns (NFIP selection bias, FEMA IA declaration requirements)

**Method 4: HAZUS as a Calibration Anchor**

FEMA's HAZUS model provides physics-based damage estimates independent of claims data. Validation studies show HAZUS estimates come within "7-12% of actual NFIP claims payouts for well-studied hurricane events" (npj Natural Hazards, Collier County study for Hurricanes Irma and Ian). HAZUS can serve as:
- An independent reference for calibrating claims-based estimates
- A prior in Bayesian frameworks
- A denominator for computing "claims-to-model" ratios as geographic bias indicators

#### 5d. The Underinsurance Quantification

Amornsiripanitch et al. (2025), published in *Nature Climate Change*, provide the most rigorous recent estimate: "70% ($17.1 billion) of total flood losses would be uninsured annually" and "88% of at-risk households are underinsured with average underinsurance of $7,208 per year." Their methodology uses First Street Foundation flood scenario loss estimates at property level, matched to NFIP policy data from the Federal Reserve Bank of Philadelphia.

This paper provides a potential **grossing-up factor** for NFIP claims: if only 30% of losses are insured, then observed NFIP claims should be scaled by approximately 1/0.30 = 3.3x at the national level, with geographic variation driven by local penetration rates and property values.

**Source quality:** The Amornsiripanitch et al. paper is published in *Nature Climate Change* and authored by Federal Reserve researchers -- high credibility. The capture-recapture methodology comes from the Annual Reviews survey by Brendan Loughin -- methodologically authoritative but not yet applied to property damage. The HAZUS validation is from peer-reviewed literature. The composite estimation framework proposed here is novel synthesis -- no single published study does exactly this.

---

## Synthesis: Recommended Bias Correction Pipeline for FloodRSCT

Based on the evidence surveyed, we recommend a layered correction pipeline:

### Layer 1: Coverage Cap Correction (Tobit)
- Apply Tobit regression to NFIP claims with right-censoring at $250,000
- Recovers the latent damage distribution above the coverage cap
- Addresses: Coverage cap distortion

### Layer 2: Selection Bias Correction (IPW/AIPW)
- Estimate NFIP take-up probability at census tract / ZCTA level
- Weight claims by inverse probability of insurance
- Use AIPW for double robustness
- Addresses: Low penetration, selection bias

### Layer 3: Private Market Adjustment
- Estimate total flood insurance (NFIP + private) using state regulatory data or AM Best market share
- Apply geographic adjustment factor: total_insured = NFIP_insured / (1 - private_market_share)
- Addresses: Private market invisibility

### Layer 4: Event-Specific Fraud/Error Correction
- Katrina (2005): Flag Gulf Coast ZCTAs; consider excluding or applying wind-water apportionment factors from forensic engineering literature
- Sandy (2012): Adjust using Sandy Claims Review outcomes ($258M in additional payments across 15,000+ policyholders)
- Addresses: Katrina wind-water fraud, Sandy engineering fraud

### Layer 5: Structural Break Adjustment
- Apply Bai-Perron tests to detect breaks in ZCTA-level claims time series
- At minimum, model pre-RR2.0 (before Oct 2021) and post-RR2.0 as separate regimes
- Apply composition-shift correction using DiD estimates (11-39% policy decline by premium quartile)
- Addresses: Risk Rating 2.0 structural break, post-disaster policy churn

### Layer 6: Credibility Weighting
- Apply Buhlmann-Straub credibility to blend corrected ZCTA-level claims with HAZUS model predictions
- ZCTAs with high penetration and many claims get high credibility on observed data
- ZCTAs with low penetration are pulled toward model-based estimates
- Addresses: Small sample sizes, geographic sparsity

### Layer 7: Multi-Source Composite
- Where FEMA IA and SBA data are available (declared disasters only), use Bayesian data fusion
- HAZUS serves as the calibration anchor
- Multiple imputation fills gaps where sources do not overlap
- Addresses: Overall undercounting, source-specific biases

---

## Confidence & Gaps

| Sub-question | Confidence | Gap |
|---|---|---|
| 1. Insurance selection bias correction | Medium-High | No published application of Heckman/IPW specifically to NFIP selection for damage estimation at ZCTA level. Methods are well-established but exclusion restriction for Heckman in flood context needs empirical validation. |
| 2. Actuarial undercounting correction | High | ASOP 39 and credibility theory are well-established. Gap: cat model adjustment factors are proprietary (AIR, RMS) and not published at ZCTA resolution. |
| 3. Sub-county penetration rates | Medium | County-level rates exist from FEMA. Census-tract-level can be constructed from OpenFEMA microdata. Private market share data is sparse -- no sub-state geographic resolution available. |
| 4. Structural break detection | High | Bai-Perron methodology is mature. RR 2.0 and Sandy breaks are well-documented. Gap: no published study applies formal break detection to ZCTA-level NFIP claims time series. |
| 5. Multi-source composite | Low-Medium | No published framework combines NFIP + IA + SBA into a bias-corrected composite for flood damage. The methods exist (MI, capture-recapture, Bayesian fusion) but the application is novel. FEMA IA racial/income disparities add a bias dimension not addressed in standard correction frameworks. |

### Key Uncertainties

1. **Private market invisibility** is the hardest bias to correct. With no nationwide database on private flood insurance, any NFIP-based correction will have an irreducible blind spot of 27-40% of insured properties.

2. **NFIP data quality** is independently problematic. GAO found "almost 71 percent of WYO company claims loss files did not have the necessary documents to support the claims" (GAO-10-66, 2009), and the DHS OIG reported "transaction error rates of more than 50 percent during the first 4 months of PIVOT implementation" (OIG-21-04, 2020). A systematic revision of NFIP claims hazard data in Florida found "the provided fields in NFIP claim data are not always complete or accurate and are often missing" with "seven features having more than 55% missing values" (Applied Sciences, 2022).

3. **Temporal validity of corrections** is a concern. Penetration rates, private market shares, and policy compositions are all changing rapidly due to RR 2.0, climate change, and housing market dynamics. Any correction calibrated on historical data may not apply to future periods.

4. **The 88% uninsured finding** from Amornsiripanitch et al. (2025) is widely cited but traces to a single source (First Street Foundation flood models) for the denominator. If FSF's flood loss estimates are biased, the underinsurance estimate inherits that bias.

---

## Key Citations Table

| # | Source | Year | Type | Key Finding |
|---|--------|------|------|-------------|
| 1 | Heckman, J. "Sample selection bias as a specification error." Econometrica 47. | 1979 | Foundational paper | Two-step correction for non-random sample selection |
| 2 | Wagenaar, D. et al. "Improved Transferability of Data-Driven Damage Models Through Sample Selection Bias Correction." Risk Analysis 41(1). | 2020 | Peer-reviewed | KMM, CBE, SVDM reduce MBE by 30-50% in flood damage models |
| 3 | Hudson, P. et al. "Evaluating the effectiveness of flood damage mitigation measures by the application of propensity score matching." NHESS 14. | 2014 | Peer-reviewed | PSM reveals EUR 1,700-15,000 bias in uncorrected mitigation estimates |
| 4 | Actuarial Standards Board. ASOP No. 39: Treatment of Catastrophe Losses. | 2014 | Industry standard | Requires actuaries to adjust historical data for catastrophe biases |
| 5 | FEMA. "OpenFEMA Dataset: NFIP Residential Penetration Rates v1." | 2025 | Government data | County-level penetration rates; excludes private market |
| 6 | FEMA. "OpenFEMA Dataset: FIMA NFIP Redacted Policies v2." | 2026 | Government data | 80M+ policy transactions at census tract resolution |
| 7 | Amornsiripanitch, N. et al. "Measuring flood underinsurance in the USA." Nature Climate Change 15. | 2025 | Peer-reviewed | 70% ($17.1B) of annual flood losses uninsured; 88% of at-risk households underinsured |
| 8 | Gourevitch, J. et al. "Effects of Risk-based Pricing Reform on Flood Insurance Uptake." J. Catastrophe Risk and Resilience. | 2025 | Peer-reviewed | RR 2.0 caused 11-39% decline in new policies |
| 9 | "To improve is to change: The effects of Risk Rating 2.0 on flood insurance demand." J. Environmental Economics and Management. | 2025 | Peer-reviewed | 53,000+ policyholders exited in first year of RR 2.0 |
| 10 | GAO-10-66. "Financial Management: Improvements Needed in NFIP's Financial Controls." | 2009 | GAO report | 71% of WYO claims files lacked supporting documents |
| 11 | DHS OIG-21-04. "FIMA Made Progress Modernizing Its NFIP System, but Data Quality Needs..." | 2020 | OIG report | 50%+ transaction error rates in PIVOT system |
| 12 | PBS Frontline / Congressional Hearing CHRG-114hhrg96989. Sandy Claims Review. | 2015-16 | Gov/investigative | $258M in additional payments; 144,000 claims reopened; engineering report fraud |
| 13 | State Farm v. Rigsby, 580 U.S. ___ (2016). | 2016 | Supreme Court | Katrina wind-water misclassification to shift liability to NFIP |
| 14 | Bai, J. & Perron, P. "Estimating and testing linear models with multiple structural changes." Econometrica 66. | 1998 | Foundational paper | Multiple unknown structural break detection |
| 15 | Wing, O.E.J. et al. "New insights into US flood vulnerability revealed from flood insurance big data." Nature Communications 11. | 2020 | Peer-reviewed | First major academic use of OpenFEMA NFIP claims for vulnerability assessment |
| 16 | FEMA. "Risk Rating 2.0 Methodology and Data Sources." | 2022 | Government report | Loss trending methodology; zip-code-level exposure adjustment |
| 17 | AM Best. Special Report: U.S. Flood Insurance Market. | 2024 | Industry report | Private market = 27-32% of premiums; varies by state |
| 18 | RFF. "The Emerging Private Residential Flood Insurance Market in the United States." | ~2020 | Research institution | No nationwide database on private flood insurance; NFIP penetration ~50% in SFHA, ~1% outside |
| 19 | Applied Sciences (MDPI). "A Systematic Revision of the NFIP Claims Hazard Data in Florida." | 2022 | Peer-reviewed | Systematic errors in NFIP cause-of-loss fields; revision changed surge/rainfall attribution |
| 20 | npj Natural Hazards. Coastal flooding validation (Collier County, Irma & Ian). | 2025 | Peer-reviewed | HAZUS estimates within 7-12% of NFIP payouts |
| 21 | Solomon, A. "Optimal Flood Insurance in a Second-Best World." | 2024 | Working paper | 88% of flood costs uninsured; NFIP = $3,440, FEMA IA = $733, SBA = $704 per household |
| 22 | BioMed Central. "Multiple imputation using linked proxy outcome data." | 2017 | Peer-reviewed | Proxy-based MI reduces bias even when 80% of outcome is missing |
| 23 | GAO-23-105977. "Flood Insurance: FEMA's New Rate-Setting Methodology." | 2023 | GAO report | RR 2.0 improves actuarial soundness; $36.5B Treasury borrowing since 2005 |
| 24 | FEMA. "Increases the NFIP's maximum coverage limits." | ~2024 | Government proposal | 61% of residential policies at $250K cap; coverage limits unchanged since 1994 |

---

## Appendix A: NFIP Data Quality Issues for FloodRSCT

The following data quality problems should be addressed before any bias correction:

1. **Missing values:** 7 of 40 features have >55% missing; ICC field has 97.42% missing (arXiv:2212.08660)
2. **Geocoding accuracy:** Older claims have poor address information; coordinates truncated to 0.1 degree for privacy
3. **Cause-of-loss misclassification:** Systematic errors in Florida required full revision of catastrophe numbers and cause codes (MDPI Applied Sciences, 2022)
4. **WYO documentation gaps:** 71% of claims files lacked supporting documents (GAO-10-66)
5. **PIVOT system errors:** 50%+ error rates during migration (OIG-21-04)

## Appendix B: OpenFEMA Datasets for Constructing Corrections

| Dataset | Records | Resolution | Key Fields for Correction |
|---------|---------|------------|--------------------------|
| FIMA NFIP Redacted Claims v2 | ~2.5M+ | Census tract, ZIP | amountPaidOnBuildingClaim, amountPaidOnContentsClaim, dateOfLoss, floodZone |
| FIMA NFIP Redacted Policies v2 | 80M+ | Census tract, ZIP | policyEffectiveDate, policyTerminationDate, totalInsurancePremiumOfThePolicy, totalBuildingInsuranceCoverage |
| NFIP Residential Penetration Rates v1 | County-level | County | penetrationRate |
| NFIP Multiple Loss Properties v1 | Properties with 2+ claims | Census tract | Repetitive loss identification |
| FEMA IA Housing Registrants (v2) | Varies by disaster | Census tract, ZIP | inspectedDamage, personalPropertyDamage, haAmount |
