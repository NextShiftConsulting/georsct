# Hex Architecture Decision -- floodcaster / georsct

**Date**: 2026-06-03
**Status**: Deferred (after NeurIPS submission)
**Scope**: Structural refactor of data pipeline from monolith to ports-and-adapters

---

## Decision

The floodcaster/georsct data pipeline will be restructured into a hexagonal
(ports-and-adapters) architecture. This refactor is **deferred until after
NeurIPS 2026 paper submission**. Current work adds derivative features
(cropland_pct, impervious_pct, wlag variants) directly in the monolith
(`build_event_dataset.py`) to support the s035 model ladder.

**Sequencing rule**: Features now in the monolith for the paper. Hex refactor
after submission, when the feature set is frozen and the cost of restructuring
is paid once.

---

## The Architecture

```
           +-----------------------------+
           |     APPLICATION LAYER       |
           |  Scenario orchestrator      |
           |  (what to build, in order)  |
           +------+---------------+------+
                  |               |
          +-------v-------+ +----v-----------+
          |  DOMAIN LAYER | | DOMAIN LAYER   |
          |  Readiness    | | Feature compute |
          |  (R, S, N)    | | (cropland_pct, |
          |               | |  rainfall, etc) |
          +-------+-------+ +----+-----------+
                  |               |
          +-------v---------------v------+
          |         PORT LAYER           |
          |  Required-evidence contracts |
          |  (what data, what shape)     |
          +---+------+------+------+----+
              |      |      |      |
         +----v+ +---v--+ +v----+ +v--------+
         |FEMA | |NOAA  | |USGS | |PostGIS  |
         |NFHL | |MRMS  | |NED  | |spatial  |
         |     | |Storm | |SMAP | |queries  |
         +-----+ +------+ +-----+ +---------+
                  ADAPTER LAYER
```

### Design Rules

1. **Domain computes readiness.** The domain layer owns R/S/N certificate
   computation, feature derivation (cropland_pct from raster pixels), and
   validation logic. No I/O, no SQL, no HTTP.

2. **Application orchestrates scenarios.** The scenario registry
   (`build_houston`, `build_new_orleans`, ...) becomes a port-backed config
   that declares which evidence is needed, not how to fetch it. The
   application layer calls adapters through ports in the correct order.

3. **Ports define required evidence.** Each port is a contract: "I need
   ZCTA-level rainfall totals for this event window" or "I need FEMA flood
   zone percentages for these ZCTAs." The port specifies shape and schema,
   not source.

4. **Adapters fetch data and render maps.** FEMA NFHL, NOAA MRMS, USGS NED,
   NLCD rasters, PostGIS spatial queries -- each is an adapter behind a port.
   SQL lives inside the PostGIS adapter, not in domain code. HTTP lives
   inside the NOAA/USGS adapters.

5. **Scenario registry is port-backed config.** Each scenario declares its
   event window, geographic extent, and which adapters to activate. Adding
   a new scenario is config, not code -- provided the required adapters exist.

---

## What This Fixes

| Current problem | Hex solution |
|---|---|
| `build_event_dataset.py` is 1800+ lines with 5 scenario functions that duplicate adapter logic | Each adapter is independent; scenarios compose adapters via ports |
| Adding a new data source requires editing 5 places | Add one adapter, wire to port, all scenarios get it |
| S3 key resolution scattered across helper functions | Each adapter owns its own S3 key resolution |
| Silent NaN propagation when data source missing | Port contract enforces schema; adapter raises on missing data |
| Testing requires full S3 access | Ports are mockable; domain logic is pure functions |

---

## What We Do NOW (Pre-Paper)

For the NeurIPS s035 model ladder, derivative features are added directly
in the monolith:

- `cropland_pct` (NLCD 2021 classes 81+82) -- **done** (commit 61f23ce)
- `impervious_pct` (NLCD 2021) -- **done** (already in 3 scenarios, added to remaining 2)
- `wlag_cropland_pct` (spatial lag) -- **done** (added to R1_WMATRIX)
- Potential: `wetland_pct` (classes 90+95), `forest_pct` (41-43), `developed_pct` (21-24)

These features go into `build_event_dataset.py` as `build_*_features()` functions
following the established pattern (S3 cache check, raster extract, centroid buffer,
manifest write). When the hex refactor happens, each becomes an adapter.

---

## Refactor Sequencing (Post-Paper)

1. **Extract adapters** from `build_event_dataset.py`:
   - `FemaFloodZoneAdapter` (from `build_flood_zone_features`)
   - `NlcdRasterAdapter` (from `build_impervious_features`, `build_cropland_features`)
   - `MrmsRainfallAdapter` (from MRMS accumulation logic)
   - `NfipClaimsAdapter` (from `build_nfip_historical`)
   - `CensusAcsAdapter` (from ACS feature joins)
   - `WmatrixAdapter` (from `compute_w_matrix_features`)

2. **Define port contracts** as typed dataclasses:
   ```python
   @dataclass
   class RainfallEvidence:
       zcta_id: str
       rainfall_total_mm: float
       event_window: tuple[datetime, datetime]
   ```

3. **Refactor scenario builders** into declarative configs:
   ```python
   HOUSTON = ScenarioConfig(
       region="houston",
       events=["harvey2017", "imelda2019", "beryl2024"],
       adapters=[fema, nlcd, mrms, nfip, census, wmatrix],
   )
   ```

4. **Domain stays pure**: feature derivation functions take DataFrames in,
   return DataFrames out. No S3, no HTTP, no SQL.

---

## Relationship to Existing Architecture Docs

- `georsct_architecture_v2.md`: Describes the **API/serving** architecture
  (swarm-it-api endpoints, cache layers, client SDK). That doc is about
  how certificates are served. This doc is about how **training data** is
  built.

- These are complementary, not competing. The hex refactor applies to the
  offline data pipeline (`build_event_dataset.py` and its launchers), not
  to the real-time API path.
