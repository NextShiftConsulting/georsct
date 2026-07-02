# Socioeconomic, Crowdsourced, and Market-Based Flood Damage Proxies

## 1. Municipal 311 Data as Flood Proxy

### Published Validation

The foundational paper is **Agonafir et al. (2022), "Understanding New York City Street Flooding through 311 Complaints,"** published in *Journal of Hydrology*. It builds an inference model linking climate and infrastructure variables to 311 flood complaints (catch basin, manhole, sewer backup categories). The key contribution: because each complaint records time, date, and exact location, 311 data can serve as spatiotemporal validation for urban flood models. Kelleher and McPhillips extended this by correlating NYC 311 complaints with topographic wetness indices for pluvial flooding.

Kontokosta et al. evaluated socio-spatial patterns in complaint propensity, incorporating socioeconomic and infrastructure factors -- critical for understanding reporting bias.

For Houston, **a study on Hurricane Harvey** used 311 complaints with address and zip code to build heatmaps for resource allocation (ResearchGate, 2021). The approach used complaint density as a proxy for damage severity at the ZIP level.

### Coverage Across the 5 Study Areas

| Area | 311 System | Open Data | Flood Categories | Suitability |
|------|-----------|-----------|-------------------|-------------|
| **Houston** | Yes (24/7) | Yes -- houstontx.gov/heatmaps, nightly refresh | "Inside Structure Flooding," "Outside Structure Flooding," "Storm Debris" under Traffic/Streets/Drainage | HIGH |
| **NYC** | Yes (24/7) | Yes -- NYC Open Data portal | "Flooding," "Catch Basin," "Sewer Backup," "Manhole" | HIGH (already using obs_has_311) |
| **New Orleans** | Yes (NOLA 311) | Yes -- datadriven.nola.gov/open-data/featured/311-calls/ | "Street Flooding and Drainage Issues" via dedicated reporting page | MEDIUM-HIGH |
| **Riverside/Coachella** | **No dedicated 311** | No open 311 portal found | County uses phone lines (951.955.1200); unincorporated areas lack system entirely | LOW -- gap |
| **SW Florida (Lee/Collier)** | **No 311 open data found** | No flood-specific 311 portal | Reporting through county emergency management | LOW -- gap |

### Known Biases

- **Income/demographic reporting bias**: Lower-income areas may underreport (lack of awareness, language barriers, distrust of government) or overreport (worse infrastructure, more flooding). Kontokosta et al. found significant socio-spatial variation in complaint propensity.
- **Unincorporated areas**: No 311 service at all -- this is fatal for Riverside/Coachella (much of the Coachella Valley is unincorporated) and parts of SW Florida.
- **Temporal bias**: Complaints spike during/after events but not proportional to damage -- a flooded basement generates one call regardless of depth.

### Best Practices for ZCTA-Level Conversion

1. Geocode complaints, spatial-join to ZCTA polygons.
2. Normalize by ZCTA population or housing units (complaints per 1,000 units).
3. Use event-window filtering (e.g., 72 hours post-storm) to isolate flood-related surges from routine drainage complaints.
4. Consider binary (any flood complaint = 1) vs. count-based targets -- binary is more robust to reporting bias.

### Houston Category Codes
- **Inside Structure Flooding**: water entered the structure from street, canal, creek, bayou.
- **Outside Structure Flooding**: water reached fence/driveway/backyard but did not enter.
- **Storm Debris**: debris/trees in streets or on sidewalks.

These map well to a severity gradient. NYC uses categories like "Flooding," "Catch Basin Clogged," "Sewer Backup," and "Manhole Overflow."

---

## 2. Property Value Impacts (Hedonic Approach)

### Key Papers and Authors

- **Bin & Polasky (2004)**, *Land Economics*: Foundational hedonic study using 8,000 sales in Pitt County, NC (1992-2002). Found **-5.7% floodplain discount**. Before Hurricane Floyd the discount was below insurance cost; after Floyd it exceeded insurance cost -- demonstrating flood events as information shocks.

- **Bin & Landry (2013)**: Extended the analysis to coastal properties.

- **Atreya, Ferreira & Kriesel (2013)**: Showed floodplain discount increases after flood events and decays over time as memory fades -- "flood risk salience" effect.

- **Kousky (2010)**: Demonstrated that SFHA designation itself (via insurance mandates) acts as a property value proxy for perceived flood risk.

- **Freddie Mac (2020)** study on Harris County, TX after Hurricane Harvey: Examined home price changes in Harvey-affected areas, finding measurable post-flood declines.

The literature consensus: **4-12% floodplain discount**, widening to **21-25% immediately post-flood**, recovering over **4-7 years**.

### Persistence of Price Declines

A major repeat-sales study (England) found:
- Inland flooding: 24.9% immediate decline, recovery after ~5 years.
- Coastal flooding: 21.1% immediate decline, recovery after ~4 years.
- **Crucially heterogeneous by income**: lowest price-quartile properties take 6-7 years to recover; highest quartile recovers in 1 year.
- A 2025 study using 8M+ US transactions found the negative impact can **persist and intensify** in areas with population decline.

### Data Sources

- **Zillow ZTRAX**: 400M+ public records, 30+ years, nearly all US counties. Spatially explicit deed transfers and sale prices. **However, ZTRAX raw data access ended September 30, 2023** -- researchers were required to destroy raw data. Post-sunset alternatives include CoreLogic, county assessor records, and ATTOM Data.
- **County assessor records**: Publicly available in most jurisdictions but require individual county scraping.
- **CoreLogic**: Commercial, comprehensive, but expensive.

### Suitability as ZCTA-Level ML Target

**MEDIUM**. Challenges:
- **Lag**: Property transactions are sparse per ZCTA per quarter. Need multi-year windows to get enough sales, which blurs the flood signal.
- **Resolution**: Works well in urban areas (many transactions per ZCTA); poor in rural/suburban ZCTAs with few sales.
- **Confounders**: Post-flood price changes conflate damage, risk perception, insurance mandate effects, and market trends. Hard to isolate actual damage.
- **Best use**: Compute delta in median assessed value (or median sale price per sqft) in a ZCTA pre- vs. post-event window. Binary target (>X% decline = damaged) is more tractable than continuous.

---

## 3. Social Media / Crowdsourced Flood Data

### Twitter/X Geotagged Reports

**Key researchers and papers:**

- **de Bruijn et al. (2019, 2020)**: Created a **global database of flood events from Twitter**, using multilingual keyword filtering. Published keyword tables in multiple languages. Demonstrated feasibility of real-time flood event detection from social media at global scale.

- **Rosser et al.**: Developed a Bayesian model fusing geotagged social media photos, remote sensing, and terrain data to estimate flood inundation probability via weights-of-evidence analysis.

- **de Albuquerque et al. (2015)**: Found that user proximity to flood zones significantly increases reliability of flood-related tweets -- proximity is a quality filter.

- **NHESS (2025)**: Content analysis of multi-year time series of flood-related Twitter/X data in Germany, using NLP to categorize 43,287 posts.

### Waze Flood Reports

Waze data has a growing academic literature:

- **Praharaj et al. (Norfolk, VA)**: Assessed trustworthiness of Waze flood reports using logistic regression -- 90.5% prediction accuracy, 71.7% of reports deemed trustworthy.
- **Harris County, TX studies**: Used Waze + traffic data as road inundation indicators during Hurricane Harvey and Tropical Storm Imelda. Random forest and AdaBoost models trained on topographic, hydrologic, and precipitation features.
- **Hampton Roads region**: Used Waze Cities data-sharing program; verified flood reports by cross-referencing with traffic jam locations within 50m buffer.
- **Waze Cities Program**: Makes aggregated user-reported incident data available to researchers and municipal partners.

### iSeeChange and CoCoRaHS

- **CoCoRaHS**: 25,000+ active stations, 14,000+ reports/day, 72M records over 26 years. Primarily **precipitation measurement**, not flood damage. Originated after the 1997 Fort Collins flood. Includes a Condition Monitoring module for weekly community impact reports. Data is freely available. **Useful as a precipitation input feature, not a flood damage target.**

- **iSeeChange**: Community-reported climate observations platform. Much smaller scale, primarily narrative/qualitative. Not suitable as a quantitative ML target.

### Privacy and Reproducibility Concerns

This is a **critical limitation**:

- **Twitter/X API restrictions**: API access has become heavily restricted and expensive since Elon Musk's acquisition. Academic API access was eliminated. A 2026 study in *Big Data & Society* documented the trajectory "from almost open to heavily restricted." Researchers can no longer share raw tweet content -- only tweet IDs, which become useless when tweets are deleted ("tweet mortality").
- **Reproducibility crisis**: Heterogeneous collection pipelines, preprocessing, and analytical approaches mean studies are rarely reproducible. Only tweet/user IDs can be shared, and platform changes render legacy code unusable.
- **Geolocation accuracy**: Users may report floods from locations distant from the actual flooding. De Bruijn found geotag locations can deviate significantly from reported flood locations.
- **Practical implication**: Social media data is essentially **non-reproducible for new research** post-2023. Historical datasets collected before API restrictions may still be usable if archived.

### Suitability as ZCTA-Level ML Target

**LOW-MEDIUM**. Problems:
- Sparse and biased toward urban, younger, tech-savvy populations.
- Geolocation unreliable for ZCTA assignment.
- Non-reproducible due to API restrictions.
- Best role: supplementary validation signal, not primary target.

---

## 4. Power Outage Data

### EAGLE-I Dataset (DOE/ORNL)

The **Environment for Analysis of Geo-Located Energy Information (EAGLE-I)** platform, developed by Oak Ridge National Laboratory for DOE CESER, is the primary public source:

- **Coverage**: 3,044 of 3,226 US counties (94.4%), representing 87.5% of US electricity customers.
- **Resolution**: County-level, 15-minute intervals.
- **Temporal span**: 2014-2022 (with annual updates).
- **Access**: Freely available for academic research via the [ORNL SMC Data Challenge portal](https://smc-datachallenge.ornl.gov/eagle/).
- **Data fields**: County, timestamp, customers without power (note: "customer" = one meter/building, not necessarily one person).

### Academic Use

- ML models have been built to predict power outage risk during extreme weather by jointly using EAGLE-I and NWS alert datasets.
- Studies on infrastructure resilience use outage data alongside flood, hurricane, and wildfire hazard data.
- Economic impact estimates: $43-62 billion/year in losses from prolonged interruptions.

### Limitations

- **County-level only**: Cannot directly map to ZCTA. Would need to use ZCTA-to-county crosswalks, losing sub-county resolution.
- **No causal attribution**: Outages are caused by wind, flooding, equipment failure, deliberate shutoffs -- cannot isolate flood-caused outages without additional data.
- **Coverage gaps**: Rural counties have intermittent/missing data. Short-duration outages (<15 min) may be missed.
- **PowerOutage.us**: Commercial (requires purchase), but provides more granular (utility-service-area level) data.

### Suitability as ZCTA-Level ML Target

**LOW-MEDIUM**. County resolution is too coarse for ZCTA-level modeling. Could serve as a **feature** (county-level outage duration during flood event) rather than a target. Sub-county utility-level data from PowerOutage.us would be needed for ZCTA resolution, but it is commercial and not reproducibly accessible.

---

## 5. Displacement and Shelter Data

### FEMA Individual Assistance (IA) Registrations -- OpenFEMA

This is the most promising dataset in this category. Multiple datasets are publicly available:

| Dataset | Geography | Key Fields | URL |
|---------|-----------|------------|-----|
| **IHP Valid Registrations v2** | County, city, ZIP code | Applicant counts, damage levels, assistance amounts | [fema.gov](https://www.fema.gov/openfema-data-page/individuals-and-households-program-valid-registrations-v2) |
| **Housing Assistance - Owners v2** | State, county, ZIP code | Inspections, damage severity, assistance provided | [fema.gov](https://www.fema.gov/openfema-data-page/housing-assistance-program-data-owners-v2) |
| **IA Housing Registrants - Large Disasters v1** | Detailed non-PII | Designed for CDBG-DR analysis | [fema.gov](https://www.fema.gov/openfema-data-page/individual-assistance-housing-registrants-large-disasters-v1) |
| **IA Multiple Loss Flood Properties v1** | Lat/Lon (1 decimal, NAD83) | Repetitive loss properties, high water marks | [fema.gov](https://www.fema.gov/openfema-data-page/individual-assistance-multiple-loss-flood-properties-v1) |

**These are aggregated at ZIP code** and can be crosswalked to ZCTA. Coverage spans DR1439 (2002) onward with weekly refresh for recent disasters.

### Red Cross / FEMA Shelter Data

- Red Cross shelter locations are published during active disasters via their online map, but **not archived as downloadable datasets**.
- FEMA's Transitional Sheltering Assistance (TSA) program data is internal; not publicly available at fine spatial resolution.
- Not suitable as a reproducible ML target.

### Census/ACS Post-Disaster Population Estimates

- The Census Bureau used **IRS change-of-address** and **USPS change-of-address** data after Katrina/Rita to estimate displacement when tax return rates dropped below usable thresholds.
- **ACS** provides annual ZCTA-level population estimates, but the 1-year lag and 5-year pooling make it too coarse temporally for event-specific damage proxy use.
- Mobile phone data (used for Hurricane Maria studies in PNAS) provides near-real-time displacement estimates but is proprietary and non-reproducible.
- The Census Household Pulse Survey reported 3.33M adults displaced by disasters in 2022, but this is national-level, not ZCTA.

### HUD CDBG-DR Allocation Data

- CDBG-DR allocations are **damage-proportional by definition**: HUD uses FEMA IA and SBA loan data to calculate "serious unmet housing need" at county and ZIP code levels.
- **Most Impacted and Distressed (MID) areas** are designated at the ZIP code level (threshold: $2M+ in unmet housing needs per ZIP) and published in Federal Register notices.
- This is a **revealed-preference damage proxy**: Congress allocates money proportional to damage, making it an externally validated signal.
- **Limitation**: Only available for Presidentially declared disasters receiving CDBG-DR appropriations (not all floods), and published with 1-2 year lag.

### Suitability as ZCTA-Level ML Target

**HIGH for OpenFEMA IA data**. ZIP-code-level damage severity from FEMA inspections is the closest public analogue to NFIP claims data and is freely available via API. The IHP Valid Registrations and Housing Assistance datasets contain damage severity categories and dollar amounts aggregated to ZIP -- these can be directly crosswalked to ZCTA.

**MEDIUM for CDBG-DR MID designations**. Binary (MID yes/no) at ZIP level. Good validation signal but too coarse for continuous targets.

**LOW for shelter/displacement data**. Not archived, not at ZCTA resolution, not reproducible.

---

## Summary Ranking for ZCTA-Level ML Target Suitability

| Proxy | Resolution | Public? | Coverage (5 areas) | Bias/Noise | **Overall** |
|-------|-----------|---------|---------------------|------------|-------------|
| **FEMA IA Registrations** | ZIP code | Yes (OpenFEMA API) | All 5 (any declared disaster) | Requires declaration; undercount of unregistered | **HIGH** |
| **311 Flood Complaints** | Point (geocoded) | Yes (varies by city) | Houston, NYC, NOLA = yes; Riverside, SWFL = no | Income/reporting bias | **HIGH where available** |
| **CDBG-DR MID Designation** | ZIP code | Yes (Federal Register) | Only CDBG-DR-funded events | Binary only; 1-2yr lag | **MEDIUM** |
| **Property Value Change** | Parcel/transaction | Partially (assessor records) | All 5 in principle | Lag, confounders, sparsity | **MEDIUM** |
| **Waze Flood Reports** | Point (road segment) | Via Waze Cities program | Urban areas only | Trustworthiness ~72%; road-only | **MEDIUM-LOW** |
| **EAGLE-I Power Outages** | County | Yes (ORNL) | All counties | Cannot isolate flood cause; too coarse | **LOW-MEDIUM** |
| **Twitter/X Flood Posts** | Approximate geotag | **No longer** (API restricted) | Historical only | Non-reproducible post-2023 | **LOW** |
| **CoCoRaHS** | Station point | Yes | Dense in some areas | Precipitation only, not damage | **Feature, not target** |
| **Shelter/Displacement** | National/county | No (not archived) | Sporadic | Not reproducible | **LOW** |

---

## Confidence and Gaps

**High confidence:**
- FEMA OpenFEMA IA datasets are the strongest alternative proxy. ZIP-level, public, API-accessible, damage-severity graded, covering all 5 study areas for declared disasters. This should be the **first dataset to pursue** after NFIP claims.
- 311 data is well-validated academically for Houston and NYC but has a hard coverage gap for Riverside/Coachella and SW Florida.

**Medium confidence:**
- Hedonic property value approaches are theoretically sound and well-published, but practical ZCTA-level implementation faces sparsity and lag problems. ZTRAX sunset makes this harder. County assessor bulk downloads remain feasible but labor-intensive.
- Waze data via the Cities program is promising for road-level flood detection but has not been validated as a damage proxy (it measures inundation extent, not damage severity).

**Key gaps:**
1. **No 311 data for 2 of 5 study areas** (Riverside, SWFL). These areas lack municipal 311 systems or do not publish open data. This is an irreducible gap -- no workaround exists.
2. **Twitter/X data is effectively dead** for new research due to API restrictions and reproducibility collapse. Historical archives (pre-2023) could still be used if available, but new collection is infeasible.
3. **Power outage data is county-level only**. Sub-county resolution would require commercial PowerOutage.us data or direct utility partnerships, neither of which is reproducible for academic work.
4. **No single proxy covers all 5 areas with ZCTA resolution except FEMA IA and NFIP**. A multi-proxy ensemble (311 where available + FEMA IA everywhere + property value change as validation) is likely the most robust path.
5. **CoCoRaHS is a feature, not a target** -- high-resolution precipitation measurements complement flood models but do not measure damage.

## Sources

- Agonafir et al. - NYC 311 Flooding, Journal of Hydrology
- NOAA repository - NYC 311 flooding study
- Predicting 311 Calls After Disaster (Hurricane Harvey) - ResearchGate
- NYC Emergency Management - 311 flood reporting
- Houston 311 Service Categories - houstontx.gov
- Houston 311 Flooding Heatmaps - houstontx.gov
- New Orleans 311 Open Data - datadriven.nola.gov
- NOLA - Report Street Flooding
- Bin & Polasky 2004 - Hedonic Flood Study, Land Economics
- Atreya, Ferreira & Kriesel 2013 - Flood risk salience
- Freddie Mac 2020 - Harris County Post-Harvey
- ZTRAX Database Info - Zillow Research
- de Bruijn et al. (2019, 2020) - Global Flood Database from Social Media
- Rosser et al. - Rapid Flood Inundation Mapping
- de Albuquerque et al. (2015) - Proximity quality filter
- NHESS 2025 - Twitter/X Flood Content Analysis
- Praharaj et al. - Waze Flood Trustworthiness Norfolk VA
- Waze Road Flooding Risk - Harris County
- CoCoRaHS Data Explorer - citizenscience.gov
- Twitter API Restrictions - Big Data & Society 2026
- EAGLE-I Dataset - ORNL
- DOE Power Outage Prediction - OSTI
- OpenFEMA IHP Valid Registrations v2
- OpenFEMA Housing Assistance Owners v2
- OpenFEMA IA Large Disasters v1
- OpenFEMA Multiple Loss Flood Properties v1
- CDBG-DR Federal Register Allocations Jan 2025
- Post-Disaster Population Estimation - PMC
- Hurricane Maria Migration - PNAS
- Kousky 2010, 2017 - Flood insurance take-up
- Kontokosta et al. - 311 socio-spatial patterns
