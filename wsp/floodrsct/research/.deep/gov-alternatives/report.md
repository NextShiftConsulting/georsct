# Government Data Sources as Alternatives/Complements to NFIP Claims for ZCTA-Level Flood Damage Measurement

**Date:** 2026-07-02
**Confidence:** HIGH for source identification, MEDIUM for bias assessment (some sources could not be fully accessed)

---

## Executive Summary

Seven government data sources were identified that can serve as alternatives or complements to NFIP claims for measuring flood damage at the ZCTA level. The strongest candidate is **FEMA Individual Assistance (IA) registrations via OpenFEMA**, which provides 6.4M registrant-level records with ZIP code resolution, damage flags, and dollar amounts -- and critically, does not require insurance participation. The second strongest is the **USACE National Structure Inventory (NSI)**, which provides per-structure replacement values and flood zone classification for every building in the US, serving as an exposure denominator.

The table below ranks sources by ZCTA suitability:

| Source | Native Resolution | ZCTA Feasible? | Records | Insurance-Independent? |
|--------|------------------|----------------|---------|----------------------|
| FEMA IA Registrations | ZIP code | YES | 6.4M | YES |
| FEMA Housing Assistance (Owners/Renters) | ZIP code | YES | 307K aggregated | YES |
| USACE NSI | Point (lat/lon) | YES (aggregate) | ~130M structures | YES (exposure) |
| SBA Disaster Loans | ZIP code (likely) | PROBABLE | Unknown publicly | NO (creditworthiness) |
| NOAA Storm Events | County/Zone | NO (too coarse) | ~2M+ events | YES |
| USGS STN High Water Marks | Point (lat/lon) | MARGINAL | ~303 events | YES (physical) |
| FEMA Public Assistance | County | NO (too coarse) | 80K+ flood | YES (infrastructure) |
| HUD CDBG-DR | Varies (grantee) | UNLIKELY | ~hundreds | YES |
| FEMA HMA Subapplications | County | NO | 5.2K | YES |

---

## 1. FEMA Individual Assistance (IA) Registrations via OpenFEMA

### Dataset: IndividualAssistanceHousingRegistrantsLargeDisasters (v1)

**API Endpoint:** `https://www.fema.gov/api/open/v1/IndividualAssistanceHousingRegistrantsLargeDisasters`

**Volume:** 6,393,012 registrant records (verified via API query, 2026-07-02)

**Spatial Resolution:** ZIP code (`damagedZipCode` field) plus Census Block ID (2010 vintage, `censusBlockId` field). This is the finest spatial resolution of any bulk damage dataset identified. Census block IDs can be mapped to ZCTAs with high precision.

**Key Fields (verified via API):**
- `damagedZipCode` -- 5-digit ZIP
- `censusBlockId` -- 15-digit census block (2010)
- `damagedCity`, `damagedStateAbbreviation`
- **Damage indicators:** `floodDamage` (boolean), `foundationDamage` (boolean + amount), `roofDamage` (boolean + amount), `destroyed` (boolean), `waterLevel`, `highWaterLocation`
- **Dollar amounts:** `rentalAssistanceAmount`, `repairAmount`, `replacementAmount`
- **Eligibility flags:** `rentalAssistanceEligible`, `repairAssistanceEligible`, `replacementAssistanceEligible`, `sbaEligible`
- **Context:** `floodInsurance` (boolean), `homeOwnersInsurance` (boolean), `ownRent`, `residenceType`, `grossIncome`, `specialNeeds`, `householdComposition`
- **Inspection:** `inspected` (boolean), `habitabilityRepairsRequired`
- `renterDamageLevel` -- categorical damage severity for renters

**Temporal Coverage:** Covers "large disasters" -- the exact cutoff is not documented in the API response, but the dataset name implies major presidential disaster declarations only. The sample record was disaster 4332 (Hurricane Harvey, 2017).

**Access:** Free REST API, JSON format, supports OData query syntax ($filter, $select, $top, $skip). Bulk CSV download also available from OpenFEMA.

**Known Biases:**
1. **Application bias:** Only includes people who applied for FEMA assistance. Non-applicants (those unaware of the program, undocumented residents, or those who self-recovered) are invisible.
2. **Declaration bias:** Only covers presidentially declared disasters. Smaller floods without declarations are excluded entirely.
3. **"Large disasters" filter:** This specific dataset is limited to large disasters, meaning moderate events may be missing. The definition of "large" is not clearly documented.
4. **Inspection completeness:** Many records show `inspected: false` with null damage fields, suggesting pre-inspection registrations are included. Only inspected records have reliable damage data.
5. **Income/demographics bias:** Lower-income households may be more likely to apply; higher-income households may self-recover without registering.
6. **Duplication with NFIP:** The `floodInsurance` boolean lets you identify overlap, but IA is "last resort" -- FEMA reduces IA grants for insured properties.

**ZCTA Suitability:** EXCELLENT. This is the single best alternative to NFIP claims. The `damagedZipCode` field maps directly to ZCTAs (with standard ZIP-to-ZCTA crosswalk caveats). The `censusBlockId` field allows even finer spatial analysis. The `floodInsurance: false` filter isolates uninsured damage that NFIP completely misses.

**Construction of a ZCTA target variable:**
- Count of IA registrations per ZCTA per disaster
- Count of inspected registrations with `floodDamage: true` per ZCTA
- Sum of `repairAmount + replacementAmount` per ZCTA
- Binary: any `destroyed: true` in ZCTA

### Dataset: HousingAssistanceOwners (v2) / HousingAssistanceRenters (v2)

**API Endpoints:**
- `https://www.fema.gov/api/open/v2/HousingAssistanceOwners` (159,433 records)
- `https://www.fema.gov/api/open/v2/HousingAssistanceRenters` (147,554 records)

**Spatial Resolution:** ZIP code, city, county, state

**Key Fields (verified via API):**
- `zipCode` -- 5-digit ZIP
- Average inspected damage amounts
- Damage categories: inspected counts by severity ($1-$10K, $10K-$25K, $25K+)
- Approved IHP amounts (repair/replacement, rental, other needs)
- Registration counts, approval counts

**Distinction from IA Registrations:** These are aggregated summaries by geographic unit and disaster, not individual registrations. They provide pre-aggregated damage statistics per ZIP/disaster combination.

**ZCTA Suitability:** GOOD. Pre-aggregated at ZIP level makes ZCTA construction straightforward. Damage severity bins ($1-$10K, $10K-$25K, $25K+) provide ordinal damage intensity.

### Dataset: RegistrationIntakeIndividualsHouseholdPrograms (v2)

**Volume:** 224,699 records

**Spatial Resolution:** ZIP code, city, county, state

**Key Fields:** `ihpAmount`, `haAmount`, `onaAmount`, registration source (call center, web, mobile), eligibility counts

**ZCTA Suitability:** MODERATE. Dollar amounts are assistance-level, not damage-level. Useful as complement.

---

## 2. SBA Disaster Loans

**Agency:** U.S. Small Business Administration

**Data availability:** SBA disaster loan data is published via data.sba.gov (CKAN portal). However, the API returned 500 errors during this research session, and the resource pages were blocked. The SBA Disaster Loan Data dataset exists at `https://data.sba.gov/dataset/sba-disaster-loan-data` but the specific resource structure could not be verified.

**What is known from SBA program documentation and FEMA cross-references:**
- SBA provides low-interest disaster loans to homeowners, renters, businesses, and nonprofits
- Homeowner loans up to $500,000 for real estate damage, $100,000 for personal property
- Business loans up to $2 million
- The `sbaEligible` field in FEMA IA data (verified above) indicates SBA referral from FEMA

**Spatial Resolution:** SBA loan applications likely contain ZIP code or address-level data. The FEMA IA data includes `sbaEligible` as a boolean, confirming the referral pipeline exists. SBA FOIA releases have historically included ZIP code-level data.

**Known Biases:**
1. **Creditworthiness filter:** SBA loans require credit approval. Low-income households and those with poor credit are systematically excluded, creating a wealth-correlated bias that is arguably worse than NFIP's insurance requirement.
2. **Application bias:** Requires active application and often FEMA referral.
3. **Repayment obligation:** As loans (not grants), they attract different populations than FEMA IA grants.
4. **Business vs. residential mix:** Includes both business and residential damage, which may need separation.

**ZCTA Suitability:** PROBABLE but UNVERIFIED. If ZIP code-level data is available (as expected from FOIA releases), it could complement FEMA IA. However, the creditworthiness bias makes it a poor proxy for total flood damage -- it systematically undercounts damage to low-income areas.

**Action needed:** Download and verify the SBA disaster loan CSV from data.sba.gov when the portal is available.

---

## 3. USGS Short-Term Network (STN) High Water Marks

**API Base:** `https://stn.wim.usgs.gov/STNServices/`

**Volume:** 303 events documented (verified via Events.json endpoint, 2026-07-02). Events span from 1888 to 2020.

**Spatial Resolution:** Point data (latitude/longitude per high water mark). Individual HWMs are GPS-located physical marks (debris lines, stain marks, seed lines) on structures and terrain.

**Event Types:** Floods, hurricanes/tropical storms, extratropical cyclones, exercises

**Data Access Issues:** During this session, filtered HWM queries returned empty arrays for multiple event/state combinations (Events 7/IA, 183/TX, 281/FL). The STN API appears to require specific content-type negotiation that did not resolve via standard JSON requests. The web application (STN Flood Event Viewer) works interactively but bulk API access proved unreliable.

**Known Characteristics:**
- HWMs are physical measurements, not damage assessments. They record water surface elevation, not damage dollars.
- Deployment is event-driven and concentrated along major river reaches and coastlines
- Spatial density varies enormously by event -- major hurricanes may have hundreds of HWMs, minor floods may have fewer than 10
- Quality ratings exist per HWM (excellent, good, fair, poor)

**Known Biases:**
1. **Deployment bias:** USGS deploys teams to areas of interest, typically along pre-identified reaches. Rural areas and minor tributaries are underrepresented.
2. **Physical not economic:** HWMs measure water elevation, not damage. Converting to damage requires coupling with depth-damage functions and structure inventory (NSI).
3. **Sparse point data:** Even for major events, HWM density is far too low for ZCTA-level target variables. Many ZCTAs will have zero HWMs in any given event.
4. **Temporal gaps:** 303 events over 130+ years means coverage is highly episodic.

**ZCTA Suitability:** POOR as a standalone target variable. Spatial density is too low for ZCTA-level regression targets. However, HWMs are EXCELLENT as validation data -- they provide ground-truth water depths that can validate flood models, which can then be used with NSI to estimate damage. The appropriate use is as a feature/validation source, not a target variable.

---

## 4. NOAA Storm Events Database

**Data Source:** NCEI bulk CSV files at `https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/`

**Volume:** Files available from 1950-2026 (verified: StormEvents_details files through d2026_c20260625.csv.gz). Three file types per year: details, fatalities, locations.

**Spatial Resolution (verified from actual data):**
- `CZ_TYPE`: "C" = county, "Z" = NWS forecast zone
- `CZ_FIPS`: County or zone FIPS code
- `BEGIN_LAT`, `BEGIN_LON`, `END_LAT`, `END_LON`: Point coordinates for event begin/end locations
- `BEGIN_LOCATION`, `END_LOCATION`: Named place descriptions
- No ZIP code field exists

**Flood Event Types (from column `EVENT_TYPE`):** "Flash Flood", "Flood", "Coastal Flood", "Lakeshore Flood"

**Damage Fields (verified):**
- `DAMAGE_PROPERTY`: Dollar amount with K/M suffix (e.g., "1.00K", "500.00K")
- `DAMAGE_CROPS`: Dollar amount with K/M suffix
- `FLOOD_CAUSE`: e.g., "Heavy Rain"
- `INJURIES_DIRECT`, `INJURIES_INDIRECT`, `DEATHS_DIRECT`, `DEATHS_INDIRECT`

**Narrative Fields:**
- `EPISODE_NARRATIVE`: Multi-paragraph event description
- `EVENT_NARRATIVE`: Per-event description with specific impacts
- `SOURCE`: Who reported (Emergency Manager, Law Enforcement, NWS, etc.)

**Known Biases and Issues:**
1. **County/zone resolution is too coarse for ZCTA analysis.** While lat/lon coordinates exist for event locations, they represent a single point (often the reporting location), not the spatial extent of flooding.
2. **Damage estimates are unreliable.** NWS warning coordination meteorologists compile damage numbers from disparate sources with no standardized methodology. Values are frequently rounded (e.g., "500.00K") and often represent initial estimates never corrected.
3. **Reporting inconsistency.** Different NWS Weather Forecast Offices have different reporting practices. Some offices systematically report more events than others for equivalent phenomena.
4. **No damage to individual structures.** All damage is aggregated to the event level at county resolution.
5. **Known undercounting of flood damage.** The NOAA Billion-Dollar Disasters methodology acknowledges that Storm Events property damage estimates are not comprehensive -- they are "the cost estimate as reported by the NWS" and do not include uninsured or unreported losses.

**ZCTA Suitability:** POOR. County/zone resolution is too coarse. The begin/end lat/lon points could theoretically be geocoded to ZCTAs, but they represent reporting locations, not damage extent. Damage dollar estimates lack rigor. This source is better used as an event catalog (identifying which disasters affected which counties) than as a damage quantification tool.

---

## 5. HUD CDBG-DR Allocations

**Agency:** U.S. Department of Housing and Urban Development

**Program:** Community Development Block Grant - Disaster Recovery

**Data Access:** The HUD Exchange site (hudexchange.info) was blocked during this session. HUD User data portals were also blocked.

**What is known from program structure:**
- CDBG-DR allocations are made by Congress to specific disasters, then administered by HUD to grantees (typically states or local governments)
- Grantees submit Action Plans describing how funds will be used
- Quarterly Performance Reports (QPR) are submitted by grantees
- Data Resolution varies by grantee -- some report at address level, others at county or city level

**Known Biases:**
1. **Allocation =/= damage.** CDBG-DR funds are allocated based on "unmet need" calculations that consider FEMA IA, NFIP, and SBA data. They are a derivative measure, not independent.
2. **Political allocation.** Congressional appropriation involves political processes; allocations do not perfectly correlate with damage severity.
3. **Extreme temporal lag.** CDBG-DR funds are often appropriated 1-2 years after a disaster, with expenditure spanning 5-10+ years.
4. **Grantee-level reporting.** Most data is at the grantee (state/city) level, not at ZCTA level.

**ZCTA Suitability:** POOR. Data is typically at grantee level (state or large city), not ZCTA. Individual grantees may have address-level data in their QPRs, but these are PDF reports with no standardized spatial identifiers. The derivation from FEMA/SBA/NFIP data makes it circular for our purposes.

---

## 6. Additional Federal Sources

### 6a. USACE National Structure Inventory (NSI)

**API:** `https://nsi.sec.usace.army.mil/nsiapi/structures?fips={county_fips}`

**Volume:** Estimated ~130 million structures nationwide (based on county-level queries returning tens of thousands per county)

**Spatial Resolution:** Point data (latitude/longitude per structure). Verified fields include:
- `x`, `y` -- longitude/latitude
- `cbfips` -- 15-digit census block FIPS
- `firmzone` -- FEMA FIRM flood zone (e.g., "X", "AE", "VE")
- `zone_sub` -- zone description
- `val_struct` -- structure replacement value ($)
- `val_cont` -- contents replacement value ($)
- `val_vehic` -- vehicle value ($)
- `fullrep` -- full replacement cost ($)
- `occtype` -- occupancy type (RES1-1SNB, COM, AGR1, etc.)
- `num_story`, `sqft`, `found_type`, `found_ht` -- structural characteristics
- `grnd_elv_m` -- ground elevation (meters)
- `static_bfe` -- base flood elevation
- `med_yr_blt` -- median year built
- `pop2amu65`, `pop2amo65` -- population estimates

**ZCTA Suitability:** EXCELLENT as an exposure denominator, NOT as a damage measure. NSI does not record actual damage -- it records what exists and what it would cost to replace. Combined with flood depth data (from hydraulic models or USGS HWMs), NSI enables synthetic damage estimation via depth-damage functions. This is how USACE's own Hydrologic Engineering Center (HEC-FIA) works.

**Bias:** Structure inventory is modeled from tax assessor data, Census, and building footprints. Values are estimated, not assessed. Rural areas may have less accurate inventories.

### 6b. FEMA Public Assistance Funded Projects

**API:** `https://www.fema.gov/api/open/v2/PublicAssistanceFundedProjectsDetails`

**Volume:** 80,381 flood-related projects (verified via API)

**Spatial Resolution:** County level only. No sub-county identifiers.

**Dollar Fields:** Project cost, federal share obligated, mitigation amount. Damage categories (A-G) classify infrastructure type.

**ZCTA Suitability:** POOR. County-level only. PA funds go to government entities and nonprofits for public infrastructure repair, not individual property damage. Not comparable to residential flood damage.

### 6c. FEMA Disaster Declarations

**API:** `https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries`

**Volume:** 11,288 flood-related declarations (verified)

**Use:** Event identification catalog. Maps disaster numbers to geographic areas (state + county FIPS) and dates. Essential for joining IA, PA, and NFIP data across datasets. Not a damage measure itself.

### 6d. FEMA NFIP Policies

**API:** `https://www.fema.gov/api/open/v2/FimaNfipPolicies`

**Volume:** 73,601,800 policy records (verified)

**Use:** Penetration rate calculation. By counting active policies per ZCTA, you can estimate what fraction of structures are insured, enabling bias correction of NFIP claims. Fields include lat/lon, flood zone, coverage amounts, and building characteristics.

### 6e. FEMA HMA Subapplications

**Volume:** 5,206 records. County-level. Contains benefit-cost analysis data for mitigation projects. Too sparse and too coarse for ZCTA targets.

### 6f. NOAA Billion-Dollar Weather and Climate Disasters

The NCEI Billion-Dollar Disasters tracker provides event-level totals for disasters exceeding $1B (CPI-adjusted). Spatial resolution is state/region level. Useful for event identification but not ZCTA-level damage measurement.

### 6g. USDA Disaster Designations

USDA Farm Service Agency (FSA) disaster designations cover agricultural damage at county level. Emergency Conservation Program (ECP) and Emergency Loans provide assistance, but data access is limited. Spatial resolution is county at best. Not suitable for ZCTA-level residential flood damage.

---

## 7. Sources Not Verified (Access Blocked or Unavailable)

The following sources are known to exist but could not be fully verified during this session:

1. **SBA Disaster Loan Data (data.sba.gov)** -- Portal returned 500 errors. Likely contains ZIP-level loan data.
2. **HUD CDBG-DR QPRs (hudexchange.info)** -- Site blocked. Grantee reports may contain sub-county data.
3. **SHELDUS (ASU)** -- Site blocked. Known county-level hazard loss database derived from Storm Events + other sources. Requires license.
4. **Census Post-Disaster Household Pulse Survey** -- May contain self-reported flood damage but spatial resolution is likely state/MSA only.
5. **State-level damage assessment databases** -- Some states (FL, TX, NC, LA) maintain their own damage assessment databases at address level. These are not federated and access varies.
6. **FEMA Preliminary Damage Assessments (PDAs)** -- Used internally to justify disaster declarations. Not publicly available in structured form.

---

## 8. Recommended Strategy for FloodRSCT

### Primary Target Variable: FEMA IA Registrations (floodDamage = true, inspected = true)

**Rationale:** 6.4M records at ZIP/census-block resolution. Captures uninsured damage. Boolean and dollar-amount damage fields. Can be filtered to inspected-only for quality.

**Construction:**
```
For each ZCTA x Disaster:
  target = count(registrations WHERE floodDamage=true AND inspected=true)
  target_dollar = sum(repairAmount + replacementAmount)
  target_severe = count(registrations WHERE destroyed=true)
```

### Bias Correction: NFIP Policies as Penetration Denominator

Use NFIP policy counts per ZCTA to estimate insurance penetration. Where penetration is high, NFIP claims are more reliable. Where penetration is low, IA registrations provide better coverage.

### Exposure Denominator: USACE NSI

Aggregate NSI structure values by ZCTA to compute total exposure at risk. Normalize damage metrics as fraction-of-exposure to enable cross-ZCTA comparison.

### Event Catalog: Disaster Declarations + Storm Events

Use FEMA declarations for major events (linking to IA data) and NOAA Storm Events for sub-declaration flood occurrence identification.

### Validation: USGS STN High Water Marks

Where available, use HWM elevations to validate hydraulic model predictions of flood depth, which feed into depth-damage functions applied to NSI.

### Secondary/Future: SBA Loans

When accessible, SBA loan data can provide a third independent damage signal at ZIP level, though creditworthiness bias must be documented.

---

## 9. Key Findings and Caveats

1. **No single source captures total flood damage.** Every source has systematic biases. The research design should explicitly model the selection process of each source.

2. **FEMA IA is the strongest alternative to NFIP claims,** but it has its own biases (declaration requirement, application behavior, inspection completion). It is NOT a census of flood damage -- it is a census of people who asked for help after a declared disaster.

3. **ZIP code vs. ZCTA is a real mapping issue.** ZIP codes are mail delivery routes; ZCTAs are Census Bureau approximations. The crosswalk is imperfect, especially in rural areas. The census block ID in IA data provides a cleaner path to ZCTA mapping via Census geographic crosswalks.

4. **The "large disasters" filter in the IA dataset is concerning.** If the dataset excludes moderate disasters, the damage distribution is right-censored in a way that biases toward catastrophic events. This needs investigation -- compare disaster numbers in the IA dataset against the full declarations list.

5. **Temporal alignment matters.** NFIP claims can be filed months after an event. IA registrations have strict deadlines (typically 60 days from declaration). SBA loans may be approved months later. Comparing across sources requires careful temporal windowing.

6. **The USACE NSI is underutilized in flood damage research.** Per-structure replacement values with flood zone classifications and ground elevations enable synthetic damage estimation that is independent of both insurance and government assistance programs.

---

## Sources Consulted (with access status)

| Source | URL | Status |
|--------|-----|--------|
| OpenFEMA API - IA Housing Registrants | fema.gov/api/open/v1/IndividualAssistanceHousingRegistrantsLargeDisasters | FULLY READ |
| OpenFEMA API - Housing Assistance Owners | fema.gov/api/open/v2/HousingAssistanceOwners | FULLY READ |
| OpenFEMA API - Housing Assistance Renters | fema.gov/api/open/v2/HousingAssistanceRenters | FULLY READ |
| OpenFEMA API - Registration Intake IHP | fema.gov/api/open/v2/RegistrationIntakeIndividualsHouseholdPrograms | FULLY READ |
| OpenFEMA API - NFIP Claims | fema.gov/api/open/v2/FimaNfipClaims | FULLY READ |
| OpenFEMA API - NFIP Policies | fema.gov/api/open/v2/FimaNfipPolicies | FULLY READ |
| OpenFEMA API - PA Funded Projects | fema.gov/api/open/v2/PublicAssistanceFundedProjectsDetails | FULLY READ |
| OpenFEMA API - PA Summaries | fema.gov/api/open/v1/PublicAssistanceFundedProjectsSummaries | FULLY READ |
| OpenFEMA API - Disaster Declarations | fema.gov/api/open/v2/DisasterDeclarationsSummaries | FULLY READ |
| OpenFEMA API - Disaster Summaries | fema.gov/api/open/v1/FemaWebDisasterSummaries | FULLY READ |
| OpenFEMA API - HMA Subapplications | fema.gov/api/open/v2/HmaSubapplications | FULLY READ |
| OpenFEMA API - Mission Assignments | fema.gov/api/open/v2/MissionAssignments | FULLY READ |
| USGS STN Events | stn.wim.usgs.gov/STNServices/Events.json | FULLY READ |
| USGS STN HWMs (filtered) | stn.wim.usgs.gov/STNServices/HWMs/FilteredHWMs.json | EMPTY RESPONSES |
| USACE NSI API | nsi.sec.usace.army.mil/nsiapi/structures | FULLY READ |
| NOAA Storm Events CSV (2026) | ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | FULLY READ (sample) |
| NOAA Storm Events file listing | ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ | FULLY READ |
| SBA Disaster Loan Data | data.sba.gov/dataset/sba-disaster-loan-data | BLOCKED (500) |
| FEMA.gov dataset pages | fema.gov/about/openfema/data-sets | BLOCKED (403) |
| HUD Exchange CDBG-DR | hudexchange.info | BLOCKED |
| SHELDUS (ASU) | cemhs.asu.edu/sheldus | BLOCKED |
| NOAA Billion-Dollar Disasters | ncei.noaa.gov/access/billions/ | BLOCKED |
| HUD Disaster Compendium | disastercompendium.hud.gov | BLOCKED |
| GAO-20-508 | gao.gov/products/gao-20-508 | PARTIAL (summary only) |
