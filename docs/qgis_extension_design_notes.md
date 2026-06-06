# QGIS Extension Design Notes for s035-model-ladder

Study QGIS extensions for **analysis patterns**, not to build inside QGIS.
These plugins encode decades of practical GIS workflow design. Extract the
design ideas, implement in Python/S3/parquet where our stack already lives.

---

## Plugin-to-Experiment Mapping

| Plugin / Tool | GeoRSCT Mode | Stage | Idea to Borrow | Implementation in Our Codebase |
|---|---|---|---|---|
| **QGIS Processing Modeler** | (pipeline design) | R0 | Chain spatial operations into one repeatable workflow with inputs, outputs, thresholds, evidence path. Pipeline provenance: every result points back to audit evidence. | Audit manifest pattern in `_validate_contract.py` and `EXPERIMENT_CONTRACT.yaml`. Each phase is a reusable pipeline step. |
| **Hotspot Analysis v3** (Getis-Ord Gi*, Local Moran's I, Bivariate Local Moran's I) | A.1 leakage, A.2 heterogeneity, A.3 residual clusters | R0 diagnostics | LISA cluster maps of residuals (HH/HL/LH/LL quadrants). Bivariate Moran for target-vs-residual spatial association. Getis-Ord for hot/cold spot detection in prediction errors. | `diag_residual_spatial` in `compute_diagnostics.py`. Currently Global Moran's I only. **Gap: add Local Moran's I cluster classification per ZCTA to identify WHERE the model fails spatially.** |
| **WhiteboxTools / Whitebox Workflows** (~500 hydrology algorithms) | B.1 catchment, B.2 flow | R1 | Sink filling, D8/D-infinity flow direction, flow accumulation, stream network extraction, watershed delineation, HAND (Height Above Nearest Drainage), stream power index. Stochastic depression analysis for flat coastal terrain. | R1 supplement features in `build_r1_features.py`. Currently: TWI, catchment area, stream density. **Gap: stream power index, HAND, D-infinity flow (better than D8 for urban grids).** |
| **TauDEM** (hydrologic DEM analysis) | B.1 catchment | R1 | Parallel watershed delineation, contributing area, stream network from DEM. Separation of upslope vs. channel flow. | Same as WhiteboxTools. TauDEM's parallel approach relevant if scaling beyond 4 scenarios. |
| **GRASS r.watershed** | B.1, B.2 | R1 | Drainage direction, stream network, half-basins. The `r.water.outlet` tool traces downstream from any point -- useful for "what is upstream of this ZCTA" queries. | W-matrix construction in R1. **Gap: upstream/downstream asymmetry in W-matrix (current W is symmetric contiguity).** |
| **Rain to Flood Analysis / SCS Runoff** | C.1 temporal, C.2 compound | R2 | Rainfall frequency analysis, SCS curve number runoff, flood hydrographs, lag-to-peak, time of concentration. Organizes rainfall -> runoff -> inundation as sequential stages. | R2 temporal features from MRMS + tides + HURDAT2. **Gap: SCS curve number as a feature (combines land cover + soil type + antecedent moisture). Rain-surge overlap duration not yet computed.** |
| **RiverGIS / HEC-RAS** | B.3 infrastructure | R1 | Separates geometry (cross-sections), roughness (Manning's n), boundary conditions, and outputs. Teaches that hydraulic properties are not just proximity features. | Infrastructure features from HIFLD + NHDPlus. **Gap: Manning's n as a feature proxy (via NLCD land cover), levee/canal network topology (NOLA story).** |
| **FLO-2D QGIS ecosystem** | B.3 infrastructure | R1-R2 | High-resolution urban flood modeling. Grid-based 2D flow with street/building obstruction. | Not directly implementable at ZCTA scale. Design lesson: urban flood behavior differs fundamentally from riverine -- separate treatment needed. |
| **InaSAFE** (dormant but pattern valuable) | (decision support) | Floodcaster | Impact report structure: hazard layer + exposure layer + vulnerability function -> action summary. Designed for disaster managers, not modelers. | Floodcaster decision support layer. **Copy the report structure: hazard map, exposure table, audit warnings, recommended action, evidence provenance.** |
| **QuickOSM** (Overpass API queries) | B.3 infrastructure | R1 | Reproducible infrastructure extraction by bbox/scenario. Custom Overpass queries for canals, levees, drains, roads, shelters, transit stations. | Infrastructure feature acquisition. Currently HIFLD + NHDPlus. **Gap: OSM-derived canal/drain density, levee proximity (especially NOLA, Houston).** |
| **Temporal Controller** | C.1 temporal | R2 | Exposes temporal slices of timestamped spatial data. Treats event timing as a layer, not a footnote. | R2 temporal features. Design lesson: **rainfall intensity, duration, peak timing, surge duration, rain-surge overlap, lag between peak rain and peak tide should each be explicit features, not collapsed into a single "event" label.** |
| **Semi-Automatic Classification Plugin (SCP)** | (raster classification) | R3 | Supervised raster classification workflow: raster stack -> consistent CRS -> patch extraction -> model input. Built on Remotior Sensus for scripted automation. | Future CNN/raster prototype. **Lesson: raster stacks require explicit CRS, resolution, and footprint discipline before modeling.** |
| **Zonal Exact Extract / Zonify** | (aggregation) | Phase 7 (FAST) | Partial pixel coverage weighting for zonal statistics. Percentile statistics (P10, P50, P90), not just mean/sum. Batch processing across return periods. | `run_fast_zcta.py` aggregation. **Gap: add depth percentiles to FAST ZCTA features. Use exact-extract weighting for boundary ZCTAs.** |
| **Geomorphic Flood Index (GFI)** | A.2 terrain | R1 | DEM-only flood susceptibility -- no hydro model needed. Low data requirement, decent accuracy. | Cheap additional R1 feature. **Gap: GFI per ZCTA as a terrain-only flood proxy.** |
| **CanFlood** | (vulnerability) | Phase 7 | Depth-damage curves, fragility functions for dikes, vulnerability function libraries. | FAST damage estimation. Study their depth-damage function library for comparison with sphere/Hazus defaults. |
| **FloodRiskSwatPlus** | (EAD) | Phase 7b | Expected Annual Damage integration across return periods. Climate scenario comparison. | FAST validation. **Gap: compute EAD from multi-return-period FAST runs as external validation target (stronger than single-period Spearman rho).** |

---

## Design Lessons by Experiment Stage

### R0 -- Tabular ML Baseline
**QGIS inspiration:** Processing Modeler + risk dashboard patterns.

Pattern: `input layers -> standardized processing chain -> output layer/report`

For us: `assembled parquet -> audit battery -> R0 feature table -> fixed folds -> model report`

The key idea is **pipeline provenance**, not cartography. Each R0 result
should point back to audit evidence.

### R1 -- Hydrology / Infrastructure Representation
**QGIS inspiration:** TauDEM, WhiteboxTools, GRASS r.watershed, RiverGIS, QuickOSM.

Feature design ideas:
- Flow accumulation (upstream contributing area)
- Drainage direction (D-infinity, not just D8)
- Stream network membership / distance to channel
- Watershed/catchment membership
- HAND (Height Above Nearest Drainage)
- Stream power index
- Levee/canal/network proximity (from OSM or USACE)

**Design lesson: Do not treat hydrology as just another column. Treat it as a topology.**

### R2 -- Temporal / Compound Flooding Representation
**QGIS inspiration:** Temporal Controller + Rain to Flood Analysis.

Timing features to organize:
- Rainfall intensity + duration + peak timing
- Surge duration + peak timing
- Rain-surge overlap window
- Lag between peak rain and peak tide
- SCS curve number (antecedent moisture proxy)

**Design lesson: Event timing is a layer, not a footnote.**

### R3 -- Raster/CNN Prototype (future)
**QGIS inspiration:** WhiteboxTools, SCP, raster classification workflows.

Pattern: `raster stack -> consistent CRS -> patch extraction -> model input`

**Design lesson: Raster stacks require explicit CRS, resolution, and footprint
discipline before modeling.**

### Floodcaster Decision Support (future)
**QGIS inspiration:** InaSAFE + QGIS report/atlas design.

Report structure to copy:
1. Hazard map
2. Exposure table
3. Audit warnings
4. Recommended action
5. Evidence provenance

VLM/VLA are **consumers** of audit outputs, not the source of scientific evidence.

---

## Actionable Gaps Identified

| Gap | Stage | Priority | Blocked By |
|---|---|---|---|
| Local Moran's I cluster map per ZCTA | R0 diagnostics | High | Need adjacency matrix on S3 |
| Stream power index feature | R1 | Medium | Need DEM + flow accumulation rasters |
| HAND (Height Above Nearest Drainage) | R1 | Medium | Need NHDPlus + DEM alignment |
| D-infinity flow (vs D8) | R1 | Low | WhiteboxTools preprocessing |
| Rain-surge overlap duration | R2 | High | MRMS + tides temporal join |
| SCS curve number proxy | R2 | Medium | NLCD + SSURGO soil data |
| Upstream/downstream asymmetric W-matrix | R1 | Medium | Flow direction raster |
| Depth percentiles in FAST ZCTA | Phase 7 | Medium | Raster zonal stats |
| EAD from multi-return-period FAST | Phase 7b | High | Multiple FAST runs |
| GFI per ZCTA | R1 | Low | DEM-derived, no external data |

---

## Implementation Ownership: floodcaster vs rsct-geocert

### floodcaster extensions (general flood analysis — reusable beyond s035)

| Module | Capability | QGIS Inspiration | Rationale |
|---|---|---|---|
| `hydrology.py` (NEW) | stream_power_index, hand, gfi, flow_accumulation | WhiteboxTools, TauDEM, GFI | DEM-derived flood features are general-purpose, not experiment-specific |
| `aggregation.py` (extend) | Percentile stats (P10/P50/P90), exact-extract pixel weighting | Zonal Exact Extract, Zonify | Natural extension of existing aggregate_by_zcta() |
| `analysis.py` (extend) | compute_ead() across return periods | FloodRiskSwatPlus, CanFlood | Core flood economics — standard actuarial metric |
| `vulnerability.py` (NEW) | Alternative depth-damage curve sets | CanFlood | Extend beyond sphere/HAZUS defaults |
| `infrastructure.py` (future) | OSM-derived canal/levee/drain proximity | QuickOSM | General flood exposure context |

### rsct-geocert s035 jobs (experiment/audit specific)

| Script | Capability | QGIS Inspiration | Rationale |
|---|---|---|---|
| `compute_residual_lisa.py` (NEW) | Local Moran's I cluster maps on model residuals | Hotspot Analysis v3 | Model diagnostic, not flood analysis |
| `validate_spatial_block_size.py` (NEW) | Variogram-based block size validation | blockCV | CV methodology QA, not flood analysis |
| `build_compound_timing.py` (NEW) | Rain-surge overlap, lag-to-peak, duration features | Rain to Flood Analysis, Temporal Controller | R2 experiment-specific feature engineering |

### Decision rule

**If it takes raster/building inputs and produces flood-relevant outputs → floodcaster.**
**If it measures model quality or builds experiment-specific features → rsct-geocert.**

### Dependency flow

```
floodcaster (hydrology, EAD, exact zonal, infrastructure)
    ^
    |  pip install floodcaster
    |
rsct-geocert/s035 (LISA diagnostics, spatial CV validation, compound timing)
    |
    |  imports libpysal, esda directly (not via floodcaster)
    v
s035 experiment results
```

### License constraints

| Repo | License | Safe to Study | Safe to Reuse Code |
|---|---|---|---|
| HotSpotAnalysis_Plugin | GPL-3.0 | Yes | No — clean-room only |
| exactextract | Apache-2.0 | Yes | Yes with attribution |
| exactextract_qgis | GPL-3.0 | Yes | No — use `exactextract` pip package |
| blockCV | GPL-3.0 | Yes | No — clean-room only |
| CanFlood | MIT | Yes | Yes with attribution |
| FloodRiskSwatPlus | MIT | Yes | Yes with attribution |
| GFA-Geomorphic-Flood-Area | GPL-2.0+ | Yes | No — clean-room only |
| WhiteboxTools | MIT | Yes | Yes — use `whitebox` pip package |

**Pattern:** `external/geospatial_reference/` = cloned for study. Production code = clean-room.

---

## What NOT to Do

- Do NOT turn this into a QGIS plugin project
- Do NOT add QGIS as a dependency
- Do NOT reimplement QGIS algorithms -- use the design patterns only
- Use QGIS as a **design library**: "How do mature GIS tools structure this problem?"
- Implement in Python/S3/parquet where the current stack lives
