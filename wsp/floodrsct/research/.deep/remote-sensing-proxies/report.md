# Remote Sensing and Modeled Flood Data Products as NFIP Alternatives

**Research Date:** 2026-07-02
**Scope:** Data products that could serve as alternatives or complements to NFIP claims for measuring flood damage/extent at the ZCTA level in the United States.

---

## Executive Summary

Twelve data products were evaluated across satellite SAR, satellite optical, hydrological models, and commercial/hybrid categories. No single product replaces NFIP claims as a damage proxy, but several can provide independent flood *extent* and *depth* measurements that, when aggregated to the ZCTA level, substantially reduce dependence on insurance-reported losses. The top-tier candidates for FloodRSCT integration are:

1. **NASA OPERA DSWx-S1** (Sentinel-1 SAR, 30 m, all-weather, free, CONUS, 2016-present)
2. **NOAA NWM Retrospective + FIM** (modeled streamflow + HAND inundation, 1979-2023, free)
3. **First Street Foundation** (3 m probabilistic flood model, zip-code summary stats free on AWS)
4. **Sentinel-1 RAPID Archive** (10 m SAR flood maps, CONUS, 2016-present, free)

These four together provide complementary coverage: SAR for observed extent regardless of weather, NWM for modeled depth/flow, and First Street for probabilistic risk context.

---

## 1. Sentinel-1 SAR Flood Mapping

### 1a. RAPID (Radar Produced Inundation Diary)

**What it is:** Fully automated 10 m resolution flood inundation archive over CONUS, generated from the entire Sentinel-1 SAR archive (January 2016 to present). Combines radar statistics and machine learning with zero manual postprocessing.

**Spatial resolution:** 10 m
**Temporal coverage:** January 2016 -- present (CONUS)
**Temporal resolution:** 6-day repeat over Europe; 12-day repeat elsewhere. With Sentinel-1C (launched Dec 2024) and Sentinel-1D (launched Nov 2025), the constellation is being restored to 6-day global revisit.
**Latency:** Archive product, not NRT. Maps generated from archived SAR imagery.
**Access:** Free. Available through NASA ASF DAAC (Vertex, asf_search, SearchAPI, Earthdata Search). Requires Earthdata Login.
**Accuracy:** Validated against USGS DSWE: overall agreement 99.06%, user accuracy 87.63%, producer accuracy 91.76%, critical success index 81.23%.

**Limitations:**
- 6-12 day revisit means rapidly evolving floods may not be captured at peak
- Urban areas remain challenging for SAR flood detection
- Binary flood/no-flood classification -- no depth information

**ZCTA suitability:** HIGH. 10 m resolution aggregates cleanly to ZCTA polygons. Provides observed flood extent area per ZCTA per event. The CONUS-wide archive from 2016 overlaps well with recent NFIP claims.

**Key source:** Shen et al. (2021), "A High-Resolution Flood Inundation Archive (2016-the Present) from Sentinel-1 SAR Imagery over CONUS," *Bulletin of the American Meteorological Society*, 102(5). [Link](https://journals.ametsoc.org/view/journals/bams/102/5/BAMS-D-19-0319.1.xml)

### 1b. Copernicus Global Flood Monitoring (GFM)

**What it is:** Fully automatic, near-real-time processing of all Sentinel-1 land images acquired in VV polarization. Uses ensemble of three independent algorithms (DLR, LIST, TU Wien). Launched 2021 as part of Copernicus EMS.

**Spatial resolution:** 20 m (from Sentinel-1 IW mode)
**Temporal coverage:** 2015 -- present (global archive); NRT ongoing
**Latency:** Typically within 5 hours of image acquisition
**Access:** Free. Full, free, and open availability. Visualization on GloFAS website and GFM Portal (registration required). Programmatic access via openEO platform.
**Output layers:** Observed flood extent, reference water mask, exclusion mask, likelihood values.

**Limitations:**
- Good accuracy for larger-scale floods in temperate/tropical zones
- Lower accuracy for smaller-scale floods and arid environments
- Coverage gaps due to Sentinel-1 orbit patterns
- No depth estimation

**ZCTA suitability:** HIGH. 20 m resolution is sufficient for ZCTA aggregation. Global coverage means it works for US territories (PR, USVI) where NFIP data is sparse. The 2015-present archive predates RAPID by one year.

**Key source:** Wendleder et al. (2025), "The fully-automatic Sentinel-1 Global Flood Monitoring service: Scientific challenges and future directions," *Remote Sensing of Environment*. [Link](https://www.sciencedirect.com/science/article/pii/S0034425725005127)

### 1c. Copernicus EMS Rapid Mapping (On-Demand)

**What it is:** On-demand rapid mapping service for emergency situations. Activated by authorized users only. Uses CCM portfolio (~30 missions) plus Sentinel data.

**Spatial resolution:** Variable (depends on tasked satellite)
**Temporal coverage:** April 2012 -- present; 491 activations through Dec 2020; ~35% are floods
**Latency:** Satellite tasking+acquisition covers ~76% of workflow time. First situational report within 4 hours of activation. Full product delivery: hours to days.
**Access:** Free. Products published on CEMS website unless classified as sensitive. Archive accessible.

**Limitations:**
- Activation-based, not continuous -- only triggered events are mapped
- Not suitable as a systematic ZCTA-level proxy because coverage is sporadic
- US events are a minority of activations

**ZCTA suitability:** LOW. Too sporadic for systematic use. Useful for validation of other products during major US flood events.

---

## 2. NASA OPERA DSWx (Dynamic Surface Water Extent)

### 2a. DSWx-S1 (Sentinel-1 SAR)

**What it is:** Level-3 near-global surface water extent product derived from Sentinel-1. Maps open inland water bodies >3 hectares and >200 m width. All-weather, day-and-night capability.

**Spatial resolution:** 30 m (MGRS grid)
**Temporal coverage:** Operational (recent). Used during Hurricane Helene/Milton (2024) and Texas floods (July 2025).
**Temporal resolution:** 6-12 day revisit
**Access:** Free. Hosted at PO.DAAC. Also on AWS Registry of Open Data. Jupyter notebook tutorials available.

**Limitations:**
- 30 m resolution misses small water bodies (<3 ha)
- Minimum width threshold of 200 m misses narrow inundation corridors
- No depth information

**ZCTA suitability:** HIGH. Standardized 30 m product designed for systematic use. Already integrated into FEMA disaster response workflows (used during Helene/Milton). The systematic, operational nature makes it ideal for consistent ZCTA-level aggregation.

### 2b. DSWx-HLS (Harmonized Landsat-Sentinel-2)

**What it is:** Optical-based surface water extent from HLS (Landsat 8/9 + Sentinel-2A/B/C).

**Spatial resolution:** 30 m
**Temporal resolution:** Nearly daily (combined constellation)
**Access:** Free. PO.DAAC.

**Limitations:**
- Cannot see through clouds -- critical limitation during flood-producing storms
- Daytime only
- Complements DSWx-S1 but cannot substitute for it during active flooding

**ZCTA suitability:** MODERATE. Useful for pre/post-event comparison and cloud-free conditions. Not reliable during the storm itself.

---

## 3. NOAA National Water Model (NWM) Retrospective

**What it is:** Multi-decade continental-scale hydrological simulation using the National Water Model. Provides modeled streamflow, soil moisture, and land surface outputs.

**Versions and temporal coverage:**
- v3.0: Feb 1979 -- Jan 2023 (44 years); includes AK, HI, PR/USVI
- v2.1: Feb 1979 -- Dec 2020 (42 years); CONUS only
- v2.0: Jan 1993 -- Dec 2018 (26 years)
- v1.2: Jan 1993 -- Dec 2017 (25 years)

**Temporal resolution of outputs:** Hourly streamflow, 3-hourly land surface
**Spatial resolution:** NWM operates on the NHDPlus network (~2.7 million stream reaches in CONUS)
**Access:** Free. AWS Open Data Registry (`noaa-nwm-retro-v2.0-pds` and similar buckets). NetCDF and Zarr formats. THREDDS Data Server.

**Critical distinction:** The retrospective simulations contain NO data assimilation -- they are purely model-driven. This is both a strength (consistent methodology over 44 years) and a weakness (no correction from observed data).

### NWM + Flood Inundation Mapping (FIM) with HAND

**What it is:** NOAA OWP's operational system that translates NWM discharge into flood inundation extent using the Height Above Nearest Drainage (HAND) approach. Creates synthetic rating curves using Manning's equation.

**Coverage:** Expanded to serve 60% of US population (up from 30% the prior year)
**Resolution:** High-resolution "street-level" visualizations
**Access:** Free. Open-source code on GitHub (NOAA-OWP/inundation-mapping). FIM hydrofabric data on ESIP S3 bucket (~1.7 TB). Visualization via National Water Prediction Service.
**Tool:** FIMserv v1.0 enables retrospective FIM generation from NWM discharge, including GeoGLOWS discharge.

**Limitations:**
- HAND assumes all inundated areas drain toward nearby flow paths -- fails to capture cross-floodplain spillover
- Relies on Manning's equation which is sensitive to reach-averaged slope values
- Generates an inundation *proxy*, not direct observation
- Urban flooding (pluvial) is not well represented
- Coverage is still incomplete (40% of US population not covered)

**ZCTA suitability:** HIGH for fluvial flooding. The 44-year retrospective is uniquely valuable for building long baselines. NWM streamflow at reach level can be spatially joined to ZCTAs via NHDPlus catchments. The FIM component adds flood depth/extent, though HAND limitations must be acknowledged. The lack of pluvial flood modeling is a significant gap for urban ZCTAs.

**Key source:** Johnson et al. (2023), "Restructuring and serving web-accessible streamflow data from the NOAA National Water Model historic simulations," *Scientific Data*. [Link](https://www.nature.com/articles/s41597-023-02316-7)

---

## 4. First Street Foundation Flood Model

**What it is:** Nationwide probabilistic flood model covering pluvial, fluvial, and coastal flooding. Built in collaboration with 80+ scientists. Uses 3DEP high-resolution elevation.

**Spatial resolution:** 3 m (native computation); aggregated to property, zip code, county, state levels
**Coverage:** Continental US, PR, HI, most of AK. 145+ million properties.
**Return periods:** Probabilistic (multiple return periods modeled)
**Flood types:** Pluvial, fluvial, coastal surge

**Access:**
- **Free (non-commercial):** ZIP code-level summary statistics on AWS Open Data Registry. CC BY-NC-SA 4.0 license. CSVs at congressional district, county, and zip code level. Includes FEMA vs. First Street risk comparison.
- **Paid (commercial/property-level):** Bulk data via API or download. Pricing not publicly listed; contact required.

**Updates:** Flood model updated annually; data updated quarterly.

**Limitations:**
- Free data is aggregate summary statistics (count of at-risk properties), not event-specific flood extent/depth
- Property-level data requires commercial license
- Model-based, not observed -- probabilistic risk, not actual flood events
- CC BY-NC-SA 4.0 restricts commercial use of free data
- No validation against actual event footprints in the free dataset

**ZCTA suitability:** MODERATE-HIGH. The free zip-code-level data provides a useful risk benchmark (how many properties are at risk per zip) but does not provide event-specific flood observations. For FloodRSCT, the free data serves as a static risk covariate rather than a dynamic flood damage proxy. The paid property-level data would be more useful but creates a cost/licensing dependency.

**Key source:** [AWS Marketplace listing](https://aws.amazon.com/marketplace/pp/prodview-g7s2ug7z3t4ue); [First Street Public Data Access](https://firststreet.org/data-access/public-access)

---

## 5. MODIS/VIIRS Near Real-Time Flood Products

### MCDWD (MODIS) and VCDWD (VIIRS)

**What it is:** Daily, near-global ~250 m resolution flood products from MODIS (Terra/Aqua) and VIIRS (NOAA-20/21). Both at Release 1.1. Use Modified Normalized Difference Water Index (MNDWI).

**Spatial resolution:** ~250 m
**Temporal coverage:**
- MODIS: 2012 -- present (NRT); 23-year archive (2003-2025) released April 2026 via LAADS DAAC
- VIIRS: New product released April 2026 to ensure continuity as MODIS instruments approach end-of-life (Terra/Aqua retirement expected 2026-27)

**Compositing:** 1-day, 2-day, and 3-day composites to reduce cloud-shadow false positives
**Latency:** Within 60-125 minutes of satellite overpass
**Access:** Free. NASA LANCE. Viewable in FLOOD viewer, NASA Worldview, GIBS layers. MODIS in HDF4, VIIRS in HDF5.

**Recent enhancement (2026):** Products now distinguish between *unusual flooding* and *recurring flooding*, improving ability to isolate novel flood events.

**Limitations:**
- 250 m resolution is coarse for ZCTA-level analysis (many ZCTAs are only a few km across)
- Cloud cover blocks detection -- critical problem during active flooding
- Flash floods difficult to capture (only ~2 overpasses/day)
- 1-day composite contaminated by cloud-shadow false positives
- Optical sensors cannot operate at night

**ZCTA suitability:** LOW-MODERATE. The 250 m resolution makes ZCTA-level aggregation imprecise, especially for small/urban ZCTAs. However, the 23-year MODIS archive (2003-2025) provides the longest continuous satellite flood record available, which is valuable for establishing long-term flood frequency at the ZCTA level despite the coarse resolution. Best used as a complementary long-baseline product, not a primary proxy.

**Key source:** [NASA Earthdata NRT Global Flood Products](https://www.earthdata.nasa.gov/data/instruments/viirs/near-real-time-data/nrt-global-flood-products); [NASA blog on 23-year archive release](https://www.earthdata.nasa.gov/news/blog/nasa-enhances-global-flood-products-smarter-detection-flooding-release-23-year-archive)

---

## 6. Landsat/Sentinel-2 Optical Flood Mapping

### Harmonized Landsat-Sentinel-2 (HLS)

**What it is:** Combined Landsat 8/9 + Sentinel-2A/B/C optical imagery, harmonized to consistent radiometry and gridding. ~3-4 day combined revisit at 30 m resolution.

**Spatial resolution:** 30 m
**Temporal resolution:** ~3-4 days (combined constellation), vs. 16 days for Landsat alone
**Access:** Free. NASA LP DAAC.

### USGS Dynamic Surface Water Extent (DSWE)

**What it is:** Landsat Level-3 science product that classifies surface water inundation from cloud/shadow/snow-free pixels. Includes "partial surface water" and "wetland" detection for sub-pixel inundation.

**Spatial resolution:** 30 m (ARD tiling, AEA projection)
**Temporal coverage:** 1982 -- present (CONUS, AK, HI) using Landsat 4-9
**Access:** Free. USGS EarthExplorer. Open data policy since 2008.
**Sensors:** Landsat 4-5 TM, Landsat 7 ETM+, Landsat 8-9 OLI

**Limitations (both HLS and DSWE):**
- Cloud cover during storms is the fundamental constraint -- floods produce clouds
- Daytime only
- 16-day Landsat revisit (alone) misses ephemeral floods
- Even HLS 3-4 day revisit may miss peak flood extent
- Partial mitigation: fusion with SAR data; statistical interpolation through cloud gaps

**ZCTA suitability:** MODERATE. The 40+ year Landsat DSWE archive is unmatched for long-term water extent change analysis. For individual flood events, cloud obstruction during the storm phase makes these products unreliable as primary damage proxies. Best used for pre/post-event comparison (assessing permanent water body changes after flooding) rather than capturing event extent.

**Key source:** [USGS DSWE product page](https://www.usgs.gov/landsat-missions/landsat-dynamic-surface-water-extent-science-products)

---

## 7. FEMA Hazus

**What it is:** FEMA's standardized loss estimation software for natural hazards. Estimates building damage, economic losses, displaced households, casualties, debris. Current version: Hazus 7.2 (compatible with ArcGIS Pro 3.4-3.6).

**What it outputs:** Flood depth grids, building damage by occupancy class, economic loss estimates, displaced households, shelter needs, debris quantities.

**What is publicly available:**
- Software: Free download from FEMA Map Service Center
- Baseline inventory data: Included with software (buildings, critical facilities, utilities at census tract level)
- Technical manuals: Hazus Flood Model Technical Manual (Hazus 7.0, June 2025) freely available
- Export tools: Hazus Export Tool (open source) for extracting results to GIS formats
- FAST (Flood Assessment Structure Tool): Building-specific flood risk assessment with annualized loss calculations

**What requires local runs:**
- Actual loss estimation requires running Hazus locally with ArcGIS Pro
- Custom scenarios require user-supplied flood hazard layers (depth grids, floodplains)
- The baseline inventory datasets have "a great deal of uncertainty" (per FEMA's own documentation)

**Limitations:**
- Requires ArcGIS Pro (commercial software dependency)
- Baseline inventory is generic, not event-specific
- Results are model estimates, not observations
- No cloud-hosted pre-computed results for historical events
- Running Hazus at ZCTA scale for many events would require significant computational effort

**ZCTA suitability:** LOW-MODERATE. Hazus can produce ZCTA-level damage estimates if someone runs the model with appropriate flood hazard inputs, but there is no pre-computed archive of event-specific results. The baseline inventory (census-tract level) provides exposure data that could be aggregated to ZCTAs. More useful as a damage estimation engine (given flood extent from another source) than as an independent flood data product.

**Key source:** [FEMA Hazus page](https://www.fema.gov/flood-maps/products-tools/hazus); [Hazus 7.0 Flood Technical Manual (June 2025)](https://www.fema.gov/sites/default/files/documents/fema_rsl_hazus-7-fltm_06272025_0.pdf)

---

## 8. Emerging Products

### 8a. ICEYE Flood Insights (Commercial SAR)

**What it is:** World's largest commercial SAR constellation (50+ satellites by 2025, 70+ launched since 2018). Provides observed flood extent and depth globally. Gen4 satellites (March 2025) deliver up to 16 cm resolution.

**Products:**
- Flood Rapid Impact (FRI): Automated NRT flood extent, updated every 6-12 hours during events
- Flood Insights (FI): Peak impact capture, building-level damage assessment

**Access:** Commercial only. Case-by-case pricing. Available via web viewer, API, or partner platforms (Guidewire, Duck Creek, EigenRisk, Esri).
**Use cases:** Insurance claims triage, parametric insurance triggers, catastrophe modeling.

**Limitations:**
- Fully commercial -- no free/academic tier identified
- Proprietary algorithms with limited methodological transparency
- Cost likely prohibitive for academic/research use
- Data access tied to commercial contracts

**ZCTA suitability:** POTENTIALLY HIGH but commercially gated. If FloodRSCT had access, ICEYE's sub-meter resolution and near-real-time depth estimation would be the best available flood damage proxy. The insurance industry integration suggests the data is validated for claims-relevant applications. But the cost/access barrier makes this impractical without a commercial partnership.

### 8b. NISAR (NASA-ISRO SAR)

**What it is:** Dual-frequency SAR satellite (L-band 24 cm + S-band 9.4 cm). Launched July 30, 2025. Declared fully operational January 2026. L-band penetrates vegetation canopy -- critical advantage over Sentinel-1 C-band for detecting flooding under trees.

**Spatial resolution:** ~10 m
**Temporal resolution:** 12-day global repeat
**Access:** Free. Over 100,000 Level 1-3 L-band products released via ASF DAAC (Feb 2026).

**Current status (July 2026):** Operational and producing data. Soil moisture maps demonstrated at 100 m resolution.

**Significance for flood mapping:** L-band SAR penetrates forest canopy, addressing a systematic false-negative problem in vegetated suburban environments that plagues C-band Sentinel-1. This is particularly relevant for southeastern US flooding where riparian vegetation obscures floodwater.

**ZCTA suitability:** HIGH (prospective). Too new for historical analysis, but will become increasingly valuable for event-based flood mapping. The vegetation penetration capability makes it uniquely suited for areas where Sentinel-1 underperforms. Integration with existing Sentinel-1 products would significantly improve flood extent estimation.

**Key source:** [NASA Earthdata NISAR data expectations](https://www.earthdata.nasa.gov/news/now-that-nisar-launched-heres-what-you-can-expect-from-the-data)

### 8c. Fathom Global Flood Model

**What it is:** Probabilistic flood hazard model from University of Bristol spin-out. Covers fluvial, pluvial, and coastal flooding with depth grids at return periods from 5 to 1,000 years.

**Spatial resolution:** 30 m (native Fathom-Global 3.0); downscalable to 10 m with regional data
**Access:**
- Free (non-commercial): Available for 16 vulnerable countries via World Bank partnership. NOT available for the US in the free tier.
- Commercial: API access, bulk download. Pricing not public.

**ZCTA suitability:** LOW for our use case. US data requires commercial license. Provides probabilistic hazard (return period flood depth) rather than event-specific observations. Competes with First Street Foundation for the same niche.

### 8d. Global Flood Database (Cloud to Street / Tellman et al.)

**What it is:** Archive of 913 major flood events (2000-2018) mapped from MODIS at 250 m resolution. Includes maximum flood extent, duration of inundation, and cloud observation conditions.

**Spatial resolution:** 250 m
**Temporal coverage:** 2000-2018 (913 events globally)
**Access:** Free. Google Earth Engine (`GLOBAL_FLOOD_DB/MODIS_EVENTS/V1`), Google Cloud Storage (`gfd_v3`), HydroShare. Open-source code on GitHub.

**ZCTA suitability:** LOW-MODERATE. Same 250 m resolution limitation as MODIS NRT. But the curated event-level structure (each image = one flood event) is useful for validation. The 18-year span provides a historical baseline, though event coverage for the US is not exhaustive.

**Key source:** Tellman et al. (2021), "Satellite imaging reveals increased proportion of population exposed to floods," *Nature*, 596(7870):80-86.

### 8e. ALTIS (Automated Loss Triage from Sentinel-1)

**What it is:** Research system for property-level flood damage assessment from Sentinel-1 SAR. Addresses the specific question insurers need: which properties were flooded, to what depth, with what estimated structural damage, and in what priority order.

**Status:** Research paper (2025). Not an operational product.

**ZCTA suitability:** POTENTIAL FUTURE. Demonstrates the feasibility of SAR-to-damage estimation. If operationalized, this approach would directly address the gap between flood extent (what satellites measure) and flood damage (what NFIP claims measure).

**Key source:** ALTIS (2025), arxiv.org/pdf/2603.13803

---

## Comparative Assessment

### Ranking by Practical Utility for ZCTA-Level Flood Damage Proxy

| Rank | Product | Resolution | Temporal Span | Cost | Measures | ZCTA Utility |
|------|---------|-----------|---------------|------|----------|--------------|
| 1 | OPERA DSWx-S1 | 30 m | Recent (operational) | Free | Observed extent | HIGH |
| 2 | NWM Retro + FIM | Reach-level | 1979-2023 | Free | Modeled flow + depth | HIGH |
| 3 | Sentinel-1 RAPID | 10 m | 2016-present | Free | Observed extent | HIGH |
| 4 | First Street (free) | Zip-code agg. | Current risk | Free (NC) | Probabilistic risk | MOD-HIGH |
| 5 | Copernicus GFM | 20 m | 2015-present | Free | Observed extent | HIGH |
| 6 | NISAR | ~10 m | 2026-present | Free | Observed extent | HIGH (future) |
| 7 | MODIS 23-yr archive | 250 m | 2003-2025 | Free | Observed extent | MOD |
| 8 | USGS DSWE (Landsat) | 30 m | 1982-present | Free | Surface water class | MOD |
| 9 | DSWx-HLS | 30 m | Recent | Free | Observed extent | MOD |
| 10 | Global Flood Database | 250 m | 2000-2018 | Free | Event extent | LOW-MOD |
| 11 | Hazus | Census tract | On-demand | Free (sw) | Modeled damage | LOW-MOD |
| 12 | ICEYE | <1 m | Event-based | $$$$ | Observed extent+depth | Gated |

### Recommended Integration Strategy for FloodRSCT

**Primary proxy (observed extent):** OPERA DSWx-S1 + RAPID archive. These provide the most systematic, validated, all-weather flood extent observations at resolutions that aggregate cleanly to ZCTAs. Use DSWx-S1 for events from its operational start forward; use RAPID for the 2016-present archive.

**Primary proxy (modeled depth/flow):** NWM Retrospective v3.0. The 44-year hourly streamflow record enables long-baseline analysis. Combine with HAND-FIM to convert discharge to inundation extent/depth for fluvial events.

**Risk context layer:** First Street Foundation free zip-code data. Provides the count of at-risk properties per zip, enabling normalization of observed flood extent by exposure.

**Long-baseline validation:** MODIS 23-year archive (2003-2025). Despite 250 m coarseness, the temporal depth is unmatched. Use for establishing flood frequency at the ZCTA level over two decades.

**Future enhancement:** NISAR L-band SAR. As the archive builds (2026+), this fills the critical vegetation-obscured flooding gap that Sentinel-1 misses in forested/suburban areas.

---

## Key Gaps and Caveats

### What remote sensing measures vs. what NFIP claims measure

This is the fundamental tension. Remote sensing products measure **flood extent** (area of water) and sometimes **flood depth** (modeled or estimated). NFIP claims measure **financial loss to insured structures**. The relationship between extent/depth and dollar damage depends on:

- Building inventory (density, value, elevation, construction type)
- Contents vs. structure damage
- Basement presence
- Whether the property was actually occupied

No remote sensing product directly measures *damage*. The ALTIS research direction (SAR-to-damage) and ICEYE's building-level products are the closest, but neither is freely available at scale.

### Temporal alignment problem

Satellite revisit times (6-12 days for SAR) mean the peak flood extent may not be captured. The observed extent on a given satellite pass may underestimate the actual maximum extent. Multi-temporal compositing (as done in RAPID and GFM) partially addresses this.

### The pluvial gap

NWM and most SAR-based products focus on fluvial (riverine) flooding. Urban pluvial flooding (from rainfall exceeding drainage capacity) is poorly captured by all products except First Street (which models it) and very high-resolution optical imagery (which is cloud-blocked during rain). This is a significant gap for urban ZCTAs.

### Aggregation effects

At 250 m (MODIS), a single pixel covers ~6.25 hectares. Many ZCTAs in dense urban areas are small enough that a handful of MODIS pixels covers the entire area, making flood fraction estimates highly uncertain. Products at 10-30 m resolution (Sentinel-1, OPERA, Landsat) are much better suited for ZCTA aggregation.

### No single product spans the full NFIP claims history

NFIP claims data extends back to the 1970s. The longest satellite flood record (MODIS) starts in 2003; the highest-quality SAR record (Sentinel-1) starts in 2015/2016. NWM Retrospective v3.0 (1979-2023) provides the best temporal overlap but is model-only (no observations). Any multi-decade analysis must accept that pre-2003 flood extent is available only through models, not observation.

---

## Sources Consulted

### Government / Agency Sources
- [NASA Earthdata - Sentinel-1](https://www.earthdata.nasa.gov/data/platforms/space-based-platforms/sentinel-1)
- [NASA Earthdata - NRT Global Flood Products](https://www.earthdata.nasa.gov/data/instruments/viirs/near-real-time-data/nrt-global-flood-products)
- [NASA Earthdata - OPERA DSWx-S1](https://www.earthdata.nasa.gov/data/catalog/pocloud-opera-l3-dswx-s1-v1-1.0)
- [NASA Earthdata - OPERA flood mapping in action](https://www.earthdata.nasa.gov/learn/data-in-action/opera-mapping-flood-waters-face-storm)
- [NASA Earthdata - NISAR data expectations](https://www.earthdata.nasa.gov/news/now-that-nisar-launched-heres-what-you-can-expect-from-the-data)
- [NASA Earthdata - 23-year MODIS flood archive release](https://www.earthdata.nasa.gov/news/blog/nasa-enhances-global-flood-products-smarter-detection-flooding-release-23-year-archive)
- [NOAA NWM Retrospective - AWS Open Data Registry](https://registry.opendata.aws/nwm-archive/)
- [NOAA OWP Inundation Mapping - GitHub](https://github.com/NOAA-OWP/inundation-mapping)
- [NOAA - FIM expansion to 60% of US](https://www.noaa.gov/news-release/noaas-transformative-flood-inundation-mapping-expands-to-60-of-us)
- [FEMA Hazus](https://www.fema.gov/flood-maps/products-tools/hazus)
- [FEMA Hazus 7.0 Flood Technical Manual (June 2025)](https://www.fema.gov/sites/default/files/documents/fema_rsl_hazus-7-fltm_06272025_0.pdf)
- [USGS DSWE product page](https://www.usgs.gov/landsat-missions/landsat-dynamic-surface-water-extent-science-products)
- [Copernicus GFM Portal](https://global-flood.emergency.copernicus.eu)
- [Copernicus EMS Rapid Mapping](https://mapping.emergency.copernicus.eu/)
- [NOAA NWS FIM FAQ](https://www.weather.gov/media/owp/operations/nws_fim_faqs.pdf)

### Peer-Reviewed / Archival Sources
- Shen et al. (2021), "A High-Resolution Flood Inundation Archive," *BAMS*, 102(5). [Link](https://journals.ametsoc.org/view/journals/bams/102/5/BAMS-D-19-0319.1.xml)
- Wendleder et al. (2025), "The fully-automatic Sentinel-1 GFM service," *Remote Sensing of Environment*. [Link](https://www.sciencedirect.com/science/article/pii/S0034425725005127)
- Johnson et al. (2023), "Restructuring NWM streamflow data," *Scientific Data*. [Link](https://www.nature.com/articles/s41597-023-02316-7)
- Tellman et al. (2021), "Satellite imaging reveals increased proportion of population exposed to floods," *Nature*, 596(7870):80-86.
- Tellman (2022), "Regional Index Insurance Using Satellite-Based Fractional Flooded Area," *Earth's Future*. [Link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EF002418)
- ALTIS (2025), "Automated Loss Triage from Sentinel-1 SAR," arXiv:2603.13803.
- FIMserv v1.0 (2025), *Environmental Modelling & Software*. [Link](https://www.sciencedirect.com/science/article/pii/S1364815225002658)
- Increasing Timeliness of Satellite-Based Flood Mapping in CEMS (2021), *Remote Sensing*, 13(11):2114. [Link](https://www.mdpi.com/2072-4292/13/11/2114)

### Commercial / Foundation Sources
- [First Street Foundation - Public Data Access](https://firststreet.org/data-access/public-access)
- [First Street Foundation - AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-g7s2ug7z3t4ue)
- [First Street Foundation - Flood Model Methodology](https://firststreet.org/methodology/flood)
- [Fathom Global Flood Map](https://www.fathom.global/product/global-flood-map/)
- [Fathom 3.0 World Bank free access](https://datacatalog.worldbank.org/search/dataset/0065654/Fathom-3-0---Free-of-Charge---Non-Commercial-Use---High-Resolution-Country-Flood-Maps-Including-Climate-Scenarios)
- [ICEYE Flood Insights](https://www.iceye.com/solutions/insurance/flood-insights)
- [ICEYE SAR Data](https://www.iceye.com/sar-data)

### Archival / Database Sources
- [Global Flood Database - Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1)
- [Global Flood Database - HydroShare](https://www.hydroshare.org/resource/6461528501c14f7c9d6b10d20dd4f657/)
- [Global Flood Database - GitHub](https://github.com/cloudtostreet/MODIS_GlobalFloodDatabase)
