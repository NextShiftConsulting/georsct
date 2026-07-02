# Socioeconomic, Crowdsourced, and Non-Traditional Flood Damage Proxies

## Research Report: Alternatives and Complements to NFIP Claims at ZCTA Level

**Date:** 2026-07-02
**Context:** FloodRSCT flood damage prediction framework. Prior research covers remote sensing (SAR, NWM, First Street) and government alternatives (FEMA IA, USACE NSI, SBA loans). This report covers the socioeconomic and crowdsourced layer.

---

## Sub-question 1: 311/Non-Emergency Service Request Data as Flood Impact Proxy

**Finding:** 311 service request data is an emerging and validated proxy for flood damage at census-tract and ZIP-code resolution. Multiple peer-reviewed studies have demonstrated that specific 311 complaint categories correlate strongly with flood events, and several major cities make this data freely available with geocoordinates.

### Evidence

**Published Research:**

- **Agonafir et al. (2022)** published "Understanding New York City Street Flooding through 311 Complaints" in the *Journal of Hydrology* (605, 127300). The study analyzed NYC 311 "street flooding" complaints as crowd-sourced data to detect explanatory variables for street flooding occurrence. Key finding: drainage network characteristics were adverse contributors to street flooding, and in boroughs with low report frequency, flooding complaints were strongly influenced by small increases in predictors (precipitation, imperviousness). The work was funded by NOAA-CESSRST (Grant NA16SEC4810008) and affiliated with CUNY. (Agonafir et al., 2022)

- **Lee and Maron (2022)** published "Community-scale big data reveals disparate impacts of the Texas Winter Storm of 2021 and its managed power outage" in *Nature Humanities and Social Sciences Communications*. They used Houston's 311 Service Helpline data, filtering water-related service requests to identify burst-pipe incidents. Data was aggregated to census-tract level using lat/lon coordinates. Key findings: census tracts with more low-income and racial minority populations had higher call peaks per area per person; numbers of water-related 311 calls surged during freezing temperatures. (Lee & Maron, 2022)

- **MDPI Water (2025)** published "Uncovering the Drivers of Urban Flood Reports: An Environmental and Socioeconomic Analysis Using 311 Data" analyzing five years (Oct 2019 -- Dec 2024) of 311 flood reports in Norfolk, Virginia. Used logistic regression with environmental variables (precipitation, tide level, topographic wetness index) and socioeconomic indicators (race, income, education). Applied permutation-based feature importance rankings. Norfolk is the second most vulnerable US community to coastal flooding after New Orleans. (MDPI Water, 2025)

- **Bayesian Under-Reporting Correction (2023):** A study on arXiv proposed "A Bayesian Spatial Model to Correct Under-Reporting in Urban Crowdsourcing," directly addressing the known bias that some neighborhoods under-report despite high flood risk, while others over-report for minor events.

**Known Biases:**

- 311 data provides binary indication (flood occurred at location) but NOT depth or extent
- Reporting rates are biased by neighborhood demographics -- high-risk areas may generate few reports if residents feel nothing will be done
- Low-income and minority communities may under-report relative to actual impact
- Affluent areas may over-report for minor events

**Request Categories That Correlate with Flood Damage:**

| City | Category Name | Notes |
|------|--------------|-------|
| NYC | Street Flooding | Direct flood proxy; studied in Agonafir et al. 2022 |
| NYC | Sewer Backup/Overflow | Correlated with combined sewer overflow during storms |
| Houston | Water Leak / Burst Pipe | Used in Lee & Maron 2022 for winter storm |
| Houston | Flooding Cases | Direct category; heatmaps at houstontx.gov |
| Chicago | Flooding Complaints | Dedicated dataset on data.cityofchicago.org |
| Norfolk | Flood Reports | Used in MDPI 2025 study |

**City Data Availability:**

| City | Portal | Geocoded? | API? | Resolution |
|------|--------|-----------|------|------------|
| NYC | NYC Open Data (data.cityofnewyork.us) | Yes (lat/lon) | Yes (SODA) | Point-level |
| Houston | houstontx.gov/heatmaps | Yes (lat/lon) | Partial | Point-level |
| Chicago | data.cityofchicago.org | Yes (lat/lon) | Yes (SODA) | Point-level |
| Boston | 311.boston.gov | Yes (lat/lon) | Yes (Open311) | Point-level |
| San Francisco | data.sfgov.org | Yes (lat/lon) | Yes (SODA) | Point-level |
| Norfolk | Via 311 platform | Yes | TBD | Point-level |

The Open311 standard (open311.org) creates interoperability across cities. Boston led procurement-level adoption; Chicago, NYC, and SF have followed.

**Source quality:** Strong. Two peer-reviewed journal publications (J. Hydrology, Nature HSSC), one MDPI journal article, multiple conference papers. Primary data is freely available through open data portals.

---

## Sub-question 2: Property Value Changes as Post-Flood Damage Signals

**Finding:** Property transaction data, primarily through Zillow ZTRAX (now at ICPSR) and CoreLogic, has been extensively used in hedonic and repeat-sales models to measure flood damage capitalization. Documented price effects range from 4% to 21% depending on flood severity and insurance context. ZTRAX was discontinued in Sept 2023 but restored at ICPSR in 2026.

### Evidence

**ZTRAX Status and Access:**

- Zillow ZTRAX was the country's largest real estate database, available free to US academic/nonprofit/government researchers from 2016 to Sept 30, 2023, when it was discontinued. Raw data had to be destroyed per access agreement.
- **ICPSR Restoration (2026):** Zillow now generates ZTRAX twice yearly and provides it to ICPSR (University of Michigan). Restricted access but free to ICPSR member institutions. Covers 20+ years of deed transfers, mortgages, foreclosures, property characteristics for ~150 million parcels nationwide. (ICPSR, 2026)
- The PLACES lab (placeslab.org) linked ZTRAX to nationwide parcel maps and social-environmental variables to develop US-wide land value maps.

**Hedonic/Repeat-Sales Methods for Isolating Flood Impact:**

- **Repeat-sales specification:** Compares transactions of the same property before and after a flood event, eliminating confounding from unobserved property characteristics. One study found a **4.2% decrease** in housing prices for treated properties after NFIP insurance reforms. (ScienceDirect, Flood insurance reforms, 2023)

- **Triple-difference hedonic framework:** Distinguishes between (a) inundated properties, (b) "near misses" in floodplain but not directly flooded, and (c) controls. Found inundated properties inside floodplain underwent **21% price discount**, while near misses saw a relative price increase. (Frontiers, 2025)

- **Recovery dynamics:** Prices of properties in flood zones recover to levels ABOVE risk-adjusted price within a few years after a major event. This "flood amnesia" effect is documented by FHFA researchers. (FHFA, 2023)

- **Data preparation best practices:** Nolte et al. (2024) published "Data Practices for Studying the Impacts of Environmental Amenities and Hazards with Nationwide Property Data" in *Land Economics*, based on 14 institutions across 11 states. ZTRAX data is described as "patchy and dirty" -- significant cleaning required.

**Key Data Limitation:** ZTRAX is property-level, not ZCTA-aggregated. Aggregation to ZCTA requires geocoding + spatial join. The PLACES system standardizes coordinates and building attributes for this purpose.

**CoreLogic:**

- CoreLogic (now operating as Cotality) provides property-level flood risk scores, distance-to-coast data, First Floor Height, Lowest Adjacent Grade, and Water Surface Elevation data.
- **Not freely available** -- commercial licensing required. Used by NFIP for rate-setting.
- Partnered with ICEYE for SAR-based post-disaster property damage assessment in real time.
- Flash Flood Risk Score uses hydrology, meteorological, and environmental datasets (1-100 numeric score).
- 94% MLS coverage, 99.99% property intelligence across US.

**Alternative After ZTRAX Discontinuation:**

- **ATTOM Data** (via Dewey): nationwide property data of comparable scope
- **County assessor data:** Publicly available but fragmented; each county has different formats, update frequencies
- **ICPSR ZTRAX:** Restored in 2026, semi-annual updates

**Source quality:** Strong. Multiple peer-reviewed papers in *Journal of Housing Economics* (special issue with 7 ZTRAX papers), *Land Economics*, *Frontiers*. FHFA research papers. Primary data sources are well-documented.

---

## Sub-question 3: Social Media and Crowdsourced Flood Reports

**Finding:** Multiple crowdsourced data streams have been validated against ground truth for flood detection, but each has significant spatial biases and resolution limitations. Geotagged tweets cover less than 1% of all tweets; Waze flood reports achieve ~73% accuracy when combined with physics models; CoCoRaHS provides precipitation ground-truth but not damage; USGS STN high-water marks are the gold standard for peak water levels.

### 3A: Twitter/X Geotagged Flood Tweets

**Validation Results:**

- NLP classification accuracy of 81% for flood-related tweet classification, with location prediction accuracy of 87% using Markov models for non-geotagged tweets. (Event classification, Annals of Operations Research, 2017)
- In Jakarta, 93% of flood locations detected in regions where people tweeted about water depth. (ScienceDirect, 2018)
- National-scale UK pipeline: 420,000 geotagged tweets from 163 at-risk areas during a 10-day flood period. (ScienceDirect, Real-time Twitter data mining, 2018)
- BERT-based classifiers applied to ~40,000 articles as alternative to Twitter for flood event extraction. (arXiv, 2023)
- Mumbai: NLP-driven crowdsourcing from Twitter for urban flood monitoring validated against ground observations. (ScienceDirect, 2025)

**Critical Limitations:**

- Only **0.42--7.9%** of tweets in any data stream are geotagged
- **Population bias**: crowdsourced data overrepresents young, affluent, urban populations
- **Spatial bias**: content concentrates in populous areas during disasters (documented in multiple studies)
- Twitter/X API access became prohibitively expensive after 2023, and academic research API was discontinued
- Real-time capability is strong; historical archive accessibility is poor

**Spatial Resolution:** Point-level when geotagged; city/neighborhood-level when geoparsed from text. Aggregation to ZCTA is feasible but sample sizes are small.

### 3B: Waze Flood Reports

**Validation Results:**

- Waze Volunteered Geographic Information (VGI) validated against NOAA Local Storm Reports (LSRs) and Storm Data publications. Reports capable of detecting flash floods aligned with official issuance and provide near-real-time situational awareness. (Nature Scientific Reports, 2022)
- Hybrid ML model combining Waze data with physics-based flood models correctly predicted **73% of risk observations** during test storm events at road-segment scale. (Journal of Hydrology, 2024)
- Hampton Roads region study: flood reports verified with traffic congestion data using 50-meter buffer -- if congestion co-located with report, considered verified. (arXiv, 2024)

**FloodMapp-Waze Integration:** FloodMapp's forecasting (tidal, riverine, rainfall) projects flood inundation onto road network and sends to Waze for automated alerts. Drivers confirm flooding in-app, creating feedback loop. Launched in Norfolk, VA.

**Spatial Resolution:** Road-segment level (very high). Can be aggregated to ZCTA.
**Temporal Resolution:** Near-real-time (minutes).
**Bias:** Overrepresents areas with high driving population; underrepresents pedestrian-only and rural areas.

### 3C: CoCoRaHS

- Community Collaborative Rain, Hail and Snow Network: citizen scientists with standardized manual gauges (0.01 inch resolution).
- Founded 1997 after Fort Collins flood ($200M damage). Expanded nationally via NSF (2003) and NOAA (2006) grants.
- **Significant Weather Reports** go directly to local NWS office, enabling flash flood warnings.
- Used daily in NWS river forecast models, flood/drought forecasts, precipitation maps.
- **Limitation for FloodRSCT:** Measures precipitation, not damage. Useful as exposure variable, not outcome variable.

### 3D: USGS Short-Term Network (STN) High Water Marks

- National database of event-based HWM collections with GPS coordinates (NAD 83) and surveyed elevations (NAVD88).
- RESTful API available at stn.wim.usgs.gov; programmatic access via HyRiver Python library.
- **Key limitation:** HWMs are manually collected after events; data collection may begin weeks after flooding. Subsequent rainfall, wind, or human activity can degrade marks.
- Engineering judgment sometimes used to estimate peak elevations where reliable marks unavailable.
- HWMs are point-level; coverage depends on USGS deployment decisions (biased toward declared disasters).

### 3E: Other Crowdsourced Platforms

| Platform | Type | Coverage | Status |
|----------|------|----------|--------|
| IFSON | Smartphone photos + ML for flood stage | Research prototype | Active |
| mPING (NOAA) | Mobile weather reports including flooding | USA | Active |
| Ushahidi | Crowdsourced damage mapping | Global (event-specific) | Active |
| FloodPatrol (Philippines) | Mobile flood reports | Philippines | Active |
| PetaJakarta | Twitter-based flood mapping | Jakarta | Completed |

**Source quality:** Mixed. Waze validation study is in Nature Scientific Reports (strong). Twitter studies span Nature, J. Hydrology, ScienceDirect (strong). CoCoRaHS is an established NWS-validated program. USGS STN is a federal primary source.

---

## Sub-question 4: Infrastructure Disruption Data

**Finding:** Power outage data (EAGLE-I) is available at county level but NOT ZIP level. Road closure data (511 systems) varies by state. Cellular network disruption data is emerging. All require disaggregation to reach ZCTA resolution.

### 4A: Power Outage Data (EAGLE-I / DOE)

- **EAGLE-I** (Environment for the Analysis of Geo-Located Energy Information): DOE/ORNL platform monitoring 146+ million customers (92% US coverage).
- **Resolution: County level, 15-minute intervals.** This is the critical limitation for ZCTA-level analysis.
- Historical data: 2014--2024, publicly available via OSTI.gov (DOE Data Explorer).
- Data includes: FIPS code, county name, state, total customers without power, timestamp.
- Used operationally by CESER, FEMA, FERC, NERC.

**ZIP-Level Alternatives:**

- **GridProfile** (gridprofile.com): delivers ZIP-level outage history and territory data. Commercial service aimed at solar/generator/battery installers. Not a research dataset.
- **Utility-specific outage maps:** Many utilities publish real-time outage maps at finer resolution, but historical data is rarely archived publicly.
- **Nighttime light satellite imagery:** Used as power outage proxy in multiple studies (e.g., Texas winter storm displacement study, ScienceDirect 2024). Resolution depends on satellite (VIIRS DNB: ~750m).

**Assessment for FloodRSCT:** County-level EAGLE-I is too coarse (one county spans 50--200 ZIPs). ZIP-level power outage data requires either commercial sources (GridProfile) or satellite-derived proxies (nighttime lights).

### 4B: Road Closures (511 Systems)

- State DOTs operate 511 traveler information systems (MnDOT, Louisiana DOTD, Wisconsin DOT, etc.).
- **Nebraska CARS 511 archive:** Most directly relevant research dataset. Examined 298 roadway flooding events (2016--2021). Novel analysis of where, when, and why (meteorologically) roadway flooding occurs. Annual median of 16 flood events per year. (ScienceDirect, 2023)
- **Data availability varies by state.** No national standardized archive exists. Researchers must contact individual DOTs.
- Events are typically point-located (road segment), so aggregation to ZCTA is feasible.

### 4C: Cellular Network Disruption

- **Anatel data (Brazil, 2024):** Combined operator network records with flood polygons to classify mobile sites as affected. Research published on arXiv.
- **FCC outage reporting:** US carriers report outages to FCC, but data is not publicly available at granular geographic levels.
- No established ZIP-level cellular disruption dataset exists for the US.

### 4D: School Closures

- No systematic US database of flood-related school closures at ZIP or finer level.
- FEMA guidance (FEMA 424) documents that flooded schools close during cleanup; closure length depends on damage severity and health hazards.
- Relevant studies are predominantly international (Indonesia, Pakistan, South Africa, India, Puerto Rico).
- **Assessment:** Not viable as a systematic data source for ZCTA-level US flood analysis.

**Source quality:** EAGLE-I is a primary federal source (strong for county level). 511 data is primary (state DOT) but access is fragmented. Cellular and school closure data have thin coverage.

---

## Sub-question 5: Displacement and Demographic Signals

**Finding:** Multiple administrative and commercial data sources capture post-flood displacement, but none are clean ZCTA-level proxies. USPS NCOA is county-level and voluntary. ACS is too temporally coarse for event-level analysis. Mobile phone location data (SafeGraph, Cuebiq, Meta Disaster Maps) offers the best spatial-temporal resolution but raises privacy and representativeness concerns.

### 5A: USPS National Change of Address (NCOA)

- Contains ~160 million permanent change-of-address records filed in last 48 months.
- **Used post-Katrina:** Census Bureau supplemented administrative data with NCOA records to track household movements and develop population estimates for affected counties. (PMC, Frey & Singer, 2010)
- **Key limitations:**
  - County-level (not ZIP)
  - Voluntary filing -- "substantially underrepresents moving households"
  - In heavily damaged areas, USPS active residence data may not be collected for months
  - Population estimates for large, heavily damaged counties are "highly uncertain" (MAPD of 21.9% for Orleans Parish)
- **Assessment for FloodRSCT:** Too coarse and too noisy. USPS NCOA is a weak displacement signal at ZCTA level.

### 5B: ACS Year-over-Year Population Changes

- ACS 5-year estimates available at ZCTA, census tract, and block group levels.
- **Temporal limitation:** 5-year pooled estimates; non-overlapping comparisons required (e.g., 2015-2019 vs. 2020-2024). Cannot isolate single-event impacts.
- 1-year estimates only available for areas with 65,000+ population -- excludes most ZCTAs.
- Census geography boundaries change every decade, complicating longitudinal analysis.
- **Assessment for FloodRSCT:** Useful for long-term trend analysis (multi-year displacement after major events like Katrina/Harvey) but NOT for event-level flood damage measurement.

### 5C: Mobile Phone Location Data (SafeGraph, Cuebiq, Meta)

**SafeGraph:**
- GPS data accurate to within a few meters.
- Used by Yabe et al. (2020c) to study Hurricane Maria impacts on 635 Puerto Rico businesses.
- Provides daily visit patterns to POIs, home/work panel data.

**Cuebiq:**
- Collects 100+ data points daily per user via Bluetooth, GPS, Wi-Fi, IoT.
- CCPA/GDPR compliant. "Data for Good" program for academic research.
- Superior combination of large scale, high accuracy, precision, and observational frequency compared to CDR and in-vehicle GPS. (Cuebiq publications)

**Meta/Facebook Disaster Maps:**
- Displacement Maps classify user as "displaced" if most common night location changed after crisis event.
- Validated during California mega-fires: even with age-biased user base, maps revealed trends, magnitude, and spatial clustering of displacement with adequate representativeness. (Jia et al., 2020)
- Used for Hurricane Maria displacement in Puerto Rico. (Acosta et al., 2020)
- Presented at ACM SIGKDD 2019.
- **Resolution:** ZIP code level (confirmed: "each point represents one zip code" in California fire study).

**Assessment for FloodRSCT:** Meta Disaster Maps and SafeGraph/Cuebiq are the best displacement signals at ZCTA level. Meta data is available through Data for Good program. SafeGraph and Cuebiq require data agreements.

### 5D: Evacuation Shelter Data

- Red Cross shelter locations searchable by ZIP code (text SHELTER + ZIP to 43362).
- No publicly available archive of shelter occupancy by origin ZIP code.
- FEMA provides temporary lodging reimbursement (14-day basis) but individual-level data is privacy-protected.
- **Assessment:** Shelter location data exists; shelter occupancy by evacuee origin ZIP does not.

**Source quality:** USPS/ACS are primary federal sources but too coarse. Mobile phone data has strong validation studies (Nature, SIGKDD) but access is restricted. Shelter data is operationally available but not archived for research.

---

## Sub-question 6: Health and Mortality Signals

**Finding:** Health data at ZCTA level is usable but comes primarily through Medicare claims, not CDC WONDER. CDC WONDER faces severe data suppression at fine geographic scales. Hospital admission surges and EMS call volumes have been studied at ZIP level and show significant flood-attributable health impacts.

### 6A: CDC WONDER Excess Mortality

- CDC WONDER is the primary US mortality data repository.
- **Data suppression:** Counts fewer than 10 are suppressed for confidentiality. At ZCTA level, most flood-related death counts fall below this threshold.
- County-level analysis is feasible; ZCTA-level is generally not viable for mortality due to suppression.
- Spatial Bayesian models can partially recover suppressed CDC WONDER data. (PMC, Estimating County-Level Mortality Rates, 2019)
- **Assessment for FloodRSCT:** Too coarse. CDC WONDER mortality is viable at county level, not ZCTA.

### 6B: Medicare Claims for Flood-Related Hospitalization (ZCTA Level)

This is the most promising health data source for ZCTA-level flood analysis.

- **Hurricane Sandy long-term mortality study:** Medicare FFS beneficiaries in flood-impacted ZCTAs had **9% higher all-cause mortality** up to 5 years post-event (aMRR 1.09, 95% CI 1.06--1.12). NYC ZCTAs: 8% higher risk. Connecticut ZCTAs: 19% higher risk. Flooded ZCTAs had higher Area Deprivation Index and lower median household income ($71,587 vs $89,213 in non-flooded ZCTAs). (PMC, Long-term Hurricane Sandy impacts, 2025)

- **Cause-specific hospitalization study:** Examined association between severe flood exposure and cause-specific hospitalization rates in adults 65+ across contiguous US, 2000--2016. Linked Medicare inpatient claims (admission date, ICD code, residential ZIP code) with satellite-based Global Flood Database flood maps by ZIP code. Investigated rate changes during and for 1 month following flood exposure, including differential impacts on marginalized communities. (PMC/arXiv, 2025)

### 6C: EMS Call Data

- **Hurricane Harvey study:** Analyzed 8,233 short-term and 15,519 long-term medical visits from 652 ZIP codes in SE Texas. Socioeconomic vulnerability and flooding were significantly associated with emergency medical care seeking. (PMC, Social Vulnerability and Access, 2021)

- **RapidSOS Data Lab:** Examined 911 call volumes at Florida ECCs before, during, and after Hurricanes Milton and Helene. Developed predictive model for timing and severity of call volume surges. Data includes timestamps, geographic information, and demographic/health/medical information. (RapidSOS, 2024)

- **Limitation:** EMS/911 call data is not publicly archived at ZIP level. Access requires data-sharing agreements with individual counties or commercial providers (RapidSOS).

### 6D: UK Evidence (for context)

- UK nested case-control study: all-cause mortality increased 6.7% per unit increase in flood index after controlling for confounders. Health consequences vary across different post-event periods and population profiles. (PMC, Floods and cause-specific mortality UK, 2024)

**Source quality:** Medicare claims studies are peer-reviewed and published in high-quality journals (strong). CDC WONDER is primary federal data but suppression limits utility. EMS data access is fragmented.

---

## Confidence & Gaps

| Sub-question | Confidence | Key Gap |
|---|---|---|
| 1. 311 Data | **HIGH** | Under-reporting bias correction methods are still nascent (1 Bayesian model paper). No national 311 standard for flood categories. |
| 2. Property Values | **HIGH** | ZTRAX was discontinued 2023, restored at ICPSR 2026 but restricted. CoreLogic is commercial. Time lag (months to years) between flood and transaction makes this a lagging indicator. |
| 3. Social Media/Crowdsourced | **MEDIUM** | Twitter/X API access effectively closed for research. Geotagging rates below 8%. Waze validation promising but single study. CoCoRaHS measures precipitation not damage. |
| 4. Infrastructure Disruption | **LOW-MEDIUM** | EAGLE-I is county-level only. No national road closure archive. Cellular disruption data not publicly available for US. |
| 5. Displacement/Demographic | **MEDIUM** | Mobile phone data (Meta, SafeGraph, Cuebiq) is best option but requires data agreements. USPS NCOA too coarse. ACS too temporally slow. |
| 6. Health/Mortality | **MEDIUM** | Medicare claims studies are strong at ZCTA level but access requires CMS data use agreement. CDC WONDER suppressed at ZCTA. EMS data fragmented. |

---

## Synthesis: Recommended Data Sources for FloodRSCT

### Tier 1: Ready for Integration (publicly available, ZCTA-aggregable, validated)

| Source | Resolution | Temporal | Damage Signal | Access |
|--------|-----------|----------|---------------|--------|
| **311 Service Requests** | Point (lat/lon) | Event-level | Direct flood reports, sewer backup, water damage | Open data portals (NYC, Houston, Chicago, Boston, SF) |
| **USGS STN High Water Marks** | Point (GPS) | Event-level | Peak water levels | REST API, HyRiver Python |
| **CoCoRaHS** | Point (station) | Daily | Precipitation (exposure, not outcome) | Public download |

### Tier 2: High Value, Restricted Access

| Source | Resolution | Temporal | Damage Signal | Access |
|--------|-----------|----------|---------------|--------|
| **Meta Disaster Maps** | ZIP code | Daily | Population displacement | Data for Good program |
| **Medicare Claims** | ZIP code | Monthly | Hospitalization surges | CMS Data Use Agreement |
| **Zillow ZTRAX** | Property-level | Transaction dates | Price depression | ICPSR (restricted) |
| **SafeGraph/Cuebiq** | Point (GPS) | Hourly | Mobility disruption, displacement | Data agreements |

### Tier 3: Supplementary (partial coverage or coarse resolution)

| Source | Resolution | Temporal | Damage Signal | Access |
|--------|-----------|----------|---------------|--------|
| **EAGLE-I Power Outages** | County | 15-minute | Infrastructure disruption | OSTI.gov |
| **Waze Flood Reports** | Road segment | Real-time | Road-level flooding | Waze for Cities |
| **511 Road Closures** | Road segment | Event-level | Transportation disruption | State DOTs (fragmented) |
| **USPS NCOA** | County | Monthly | Displacement | Census Bureau |
| **ACS Population** | ZCTA | 5-year pools | Long-term displacement | Census Bureau |

### Tier 4: Not Currently Viable for ZCTA-Level Analysis

| Source | Reason |
|--------|--------|
| CDC WONDER mortality | Suppressed below 10 counts at ZCTA |
| School closures | No systematic US database |
| Cellular network disruption | Not publicly available for US |
| Red Cross shelter occupancy | Not archived by evacuee origin ZIP |

---

## Key Citations Table

| # | Source | Year | Type | Key Finding |
|---|--------|------|------|-------------|
| 1 | Agonafir et al., J. Hydrology 605:127300 | 2022 | Peer-reviewed | NYC 311 street flooding complaints detect flood-prone ZIPs via drainage/precip variables |
| 2 | Lee & Maron, Nature HSSC | 2022 | Peer-reviewed | Houston 311 water calls at census-tract level reveal disparate burst-pipe impacts by income/race |
| 3 | MDPI Water 17(21):3178 | 2025 | Peer-reviewed | Norfolk 311 flood reports driven by environmental + socioeconomic factors over 5 years |
| 4 | Nolte et al., Land Economics 100(1):200 | 2024 | Peer-reviewed | Best practices for ZTRAX hedonic analysis; data is "patchy and dirty" |
| 5 | Pollack et al., J. Housing Economics | 2023 | Peer-reviewed | SFHA remapping benefits estimated via ZTRAX repeat-sales |
| 6 | Frontiers Environmental Economics | 2025 | Peer-reviewed | Triple-difference hedonic: 21% discount for inundated floodplain properties |
| 7 | FHFA Economic Summit | 2023 | Government | Flood zone prices recover above risk-adjusted level within years ("flood amnesia") |
| 8 | Nature Scientific Reports (Waze validation) | 2022 | Peer-reviewed | Waze VGI detects flash floods aligned with NOAA LSRs; user-base bias identified |
| 9 | J. Hydrology (Waze hybrid ML) | 2024 | Peer-reviewed | 73% correct prediction of road-level flood risk combining Waze + physics |
| 10 | EAGLE-I dataset, DOE/ORNL | 2014-2024 | Government dataset | County-level power outage data, 15-min intervals, 92% US coverage |
| 11 | Nebraska CARS 511, ScienceDirect | 2023 | Peer-reviewed | 298 roadway floods 2016-2021; precursor soil moisture critical |
| 12 | Frey & Singer, PMC | 2010 | Peer-reviewed | USPS NCOA for post-Katrina displacement; county-level, voluntary, uncertain |
| 13 | Jia et al., arXiv/ResearchGate | 2020 | Peer-reviewed | Meta Disaster Maps reveal displacement trends at ZIP level despite age bias |
| 14 | Facebook Disaster Maps, ACM SIGKDD | 2019 | Conference | Displacement Maps classify user as displaced when night location changes |
| 15 | PMC (Hurricane Sandy mortality) | 2025 | Peer-reviewed | 9% higher 5-year mortality in flood-exposed ZCTAs (Medicare FFS, 65+) |
| 16 | PMC/arXiv (Cause-specific hospitalization) | 2025 | Peer-reviewed | Medicare claims linked to Global Flood Database by ZIP; cause-specific rates |
| 17 | PMC (Hurricane Harvey EMS) | 2021 | Peer-reviewed | 8,233 medical visits from 652 ZIP codes; SVI + flooding drive care-seeking |
| 18 | ICPSR (ZTRAX restoration) | 2026 | Data repository | ZTRAX restored at ICPSR; semi-annual updates; restricted but free to members |
| 19 | Cuebiq publications | Various | Commercial/research | 100+ data points/day/user; superior accuracy vs CDR; CCPA/GDPR compliant |
| 20 | RapidSOS Data Lab | 2024 | Commercial research | 911 call volume prediction model for hurricane timing/severity |

---

## Implications for FloodRSCT Framework

### Why 311 Data Outperformed NFIP Claims

The experimental finding that 311 data outperformed NFIP claims in Houston and NYC is consistent with the literature:

1. **Coverage bias in NFIP:** Only ~30% of properties in SFHAs carry flood insurance nationally; outside SFHAs, rates are even lower. NFIP claims capture only the insured population's losses.

2. **311 captures the uninsured:** 311 complaints come from anyone experiencing flooding, regardless of insurance status. In low-income communities where NFIP participation is lowest, 311 may be the only damage signal.

3. **Temporal immediacy:** 311 complaints are filed during/immediately after events. NFIP claims involve adjuster visits, processing delays (weeks to months).

4. **Spatial precision:** 311 records include lat/lon coordinates; NFIP claims are reported at census-tract level with truncated coordinates.

### Recommended Multi-Source Damage Index

For FloodRSCT, consider constructing a composite damage index combining:

1. **311 flood complaints** (event-level, point data, multiple cities)
2. **USGS STN high water marks** (event-level, physical measurement)
3. **FEMA Individual Assistance registrations** (event-level, ZIP, captures uninsured)
4. **Meta Disaster Maps displacement** (daily, ZIP, if Data for Good access obtained)
5. **Medicare hospitalization surges** (monthly, ZIP, if CMS agreement obtained)

This multi-source approach addresses the single-source biases: 311 under-reporting in low-income areas, NFIP non-participation, USGS sparse spatial coverage, and Meta demographic skew.
