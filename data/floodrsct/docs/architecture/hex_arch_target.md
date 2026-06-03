# Hexagonal Architecture Target State

**Status:** TARGET STATE -- current codebase is SageMaker job scripts. Migration begins post-paper.

---

## Design Rule

> Domain computes readiness. Application orchestrates scenarios. Ports define required evidence. Adapters fetch data and render maps.

---

## Package Structure

```text
src/
  floodcaster/
    domain/
      scenario.py
      hazard_evidence.py
      spatial_unit.py
      exposure.py
      residual.py
      readiness_certificate.py
      decision_action.py        # Maps RSCT outcomes to operational guidance

    application/
      use_cases/
        build_event_dataset.py
        compute_spatial_probe.py
        compute_readiness_certificate.py
        render_decision_view.py
        run_oahu_h4_probe.py

    ports/
      raster_source.py
      boundary_source.py
      claims_source.py
      exposure_source.py
      spatial_graph_source.py
      certificate_store.py
      map_renderer.py
      scenario_registry.py

    adapters/
      inbound/
        cli/
        streamlit/
        api/

      outbound/
        s3/
        postgis/
        fema/
        noaa/
        usgs/
        census/
        svg/
        deckgl/
        geojson/

  georsct/
    domain/
      quality.py
      compatibility.py
      turbulence.py
      rsn_simplex.py
      kappa.py
      certificate.py

    application/
      use_cases/
        score_quality_compatibility_turbulence.py
        classify_readiness_zone.py
        explain_certificate.py

    ports/
      certificate_repository.py
      embedding_source.py
      spatial_metric_source.py

    adapters/
      outbound/
        faiss/
        pgvector/
        json/
```

---

## Principles

### 1. Rendering is an adapter concern

The domain produces a `ReadinessView` object. The SVG/DeckGL/Streamlit adapter decides how to draw it.

```python
@dataclass
class ReadinessView:
    scenario_id: str
    quality: float
    compatibility: float
    turbulence: float
    decision: str              # EXECUTE | CAUTION | REFUSE (ADR-029/034)
    spatial_annotations: list
    residual_hotspots: list
    evidence_warnings: list
```

Adapter locations:

- `ports/map_renderer.py`
- `adapters/outbound/svg/oahu_spatial_grounding_renderer.py`
- `adapters/outbound/deckgl/interactive_map_renderer.py`

### 2. Data sources are outbound adapters

FEMA, NOAA, USGS, Census should not leak into the domain.

```text
ports/raster_source.py          # Abstract
ports/hazard_source.py          # Abstract
adapters/outbound/noaa/nwm_adapter.py
adapters/outbound/fema/nfhl_adapter.py
```

The use case calls `hazard_source.fetch_hazard_grid(scenario)` without knowing the provider.

### 3. SQL belongs inside the PostGIS adapter

```text
adapters/outbound/postgis/sql/
  00_extensions.sql
  01_load_boundaries.sql
  02_spatial_indexes.sql
  03_exposure_overlay.sql
  04_residuals.sql
  05_adjacency.sql
  06_quality_checks.sql
```

PostGIS is an implementation detail. The domain only knows `SpatialGraphSource` or `ExposureSource`.

### 4. Scenario registry is a port-backed configuration source

```yaml
scenario_id: oahu_h4_coastal_surge_10yr
hazard_type: coastal_surge
truth_source: nfip_claims
truth_event_match: low
spatial_unit: zcta
model_unit: building
diagnostic_status: illustrative_not_adjudicative
```

Accessed through:

- `ports/scenario_registry.py`
- `adapters/outbound/yaml/scenario_registry_yaml.py`

### 5. GeoRSCT is a separate reusable package

GeoRSCT theory (quality, compatibility, turbulence, kappa, certificates) is reusable across hazard types (flood, earthquake, wildfire). Keeping it separate from floodcaster prevents the theory from accumulating flood-specific assumptions.

### 6. Decision vocabulary follows RSCT governance

Per ADR-029/034:

- **Public (certificate layer):** EXECUTE | CAUTION | REFUSE
- **Internal (gatekeeper layer):** EXECUTE | REJECT | BLOCK | RE_ENCODE | REPAIR | WARN | FALLBACK

The `decision_action.py` module maps these to floodcaster-specific operational guidance at the adapter boundary, not in the domain.

---

## Feature Backlog Under Hex Arch

| Feature | Hex location | Why |
|---------|-------------|-----|
| Quality / Compatibility / Turbulence scoring | `georsct/domain/` | Core theory |
| Readiness decision mapping | `floodcaster/domain/decision_action.py` | Maps RSCT outcomes to operational guidance |
| Oahu H4 use case | `floodcaster/application/use_cases/run_oahu_h4_probe.py` | Orchestrates domain + ports |
| Spatial grounding panel | `adapters/outbound/svg/` | Rendering concern |
| Interactive map / dashboard | `adapters/outbound/deckgl/` | Rendering concern |
| FEMA NFHL connector | `adapters/outbound/fema/` | External data |
| NOAA NWM / CO-OPS / MRMS | `adapters/outbound/noaa/` | External data |
| USGS DEM / water data | `adapters/outbound/usgs/` | External data |
| Census TIGER / ACS | `adapters/outbound/census/` | External data |
| PostGIS spatial joins | `adapters/outbound/postgis/` | Persistence/query |
| S3 storage | `adapters/outbound/s3/` | Storage |
| CLI / Streamlit / API | `adapters/inbound/` | Entry points |

---

## Existing Contract Inventory

The ecosystem already has substantial contract infrastructure. The hex arch should
absorb and formalize these, not replace them.

| Contract | Location | Role in Hex Arch |
|----------|----------|------------------|
| **FEATURE_CONTRACT.yaml** | `floodrsct/FEATURE_CONTRACT.yaml` | Port contract: defines required evidence fields, temporal classes, causal boundaries. Becomes the schema for `ports/exposure_source.py` and `ports/claims_source.py`. |
| **EXPERIMENT_CONTRACT.yaml** | `floodrsct/exp/s035-model-ladder/EXPERIMENT_CONTRACT.yaml` | Application contract: declares per-phase inputs/outputs, S3 artifacts, pip deps. Validates experiment readiness before launch. |
| **Scenario configs** | `floodrsct/configs/{houston,new_orleans,...}.yaml` | Adapter config: per-scenario metadata (FIPS codes, events, USGS sites, bounding boxes). Currently consumed by `ScenarioRegistry`. Becomes `adapters/outbound/yaml/scenario_registry_yaml.py`. |
| **ScenarioRegistry** | `rsct-geocert/db/scripts/scenario_registry.py` | Already a port pattern: `ScenarioRegistry.from_db()` and `from_schema_dir()` with dual lookup (by pipeline_name and scenario_id). Maps to `ports/scenario_registry.py`. |
| **Database schema** | `rsct-geocert/db/schema/001_scenarios.sql`, `006_scenario_pipeline.sql` | Adapter implementation: PostGIS/DuckDB tables backing the ScenarioRegistry port. Maps to `adapters/outbound/postgis/`. |
| **Certificate schema** | `swarm-it-api/schemas/certificate_v1.py`, `geo_v1.py` | Domain contract: RSCTCertificateCore, GateDecision, PublicDecision, ScenarioId enums. Shared with GeoRSCT domain. |
| **Port ABCs** | `swarm-it-adk/sidecar/ports.py` | Existing hex arch: EmbeddingPort, RSNPort, CertifyPort. Pattern to follow for floodcaster ports. |
| **Suite A contract tests** | `swarm-it-api/tests/suite_a_contract.py` | Test pattern: validates API shape (simplex valid, decision is valid gate outcome) before semantic tests. |
| **3-layer validation** | `floodrsct/DATA_VALIDATION.md` + `_validate_contract.py` | Quality gate: interface -> assembly -> data lock. Enforces FEATURE_CONTRACT at build time. |
| **SCRIPT_PROVENANCE.yaml** | `floodrsct/exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml` | Audit trail: per-script generator, draft date, cleanup record. |

### Key Insight: ScenarioRegistry is Already a Port

`ScenarioRegistry` in `db/scripts/scenario_registry.py` already implements the port
pattern with `from_db()` (PostGIS adapter) and `from_schema_dir()` (YAML adapter).
The hex arch migration should promote this to `ports/scenario_registry.py` and rename
the existing implementations as adapters, not rewrite them.

### Key Insight: FEATURE_CONTRACT is a Port Schema

`FEATURE_CONTRACT.yaml` defines the evidence interface -- what features exist, their
temporal class (invariant/slow_drift/event_window/post_event), source provenance, and
output columns. This is exactly what `ports/exposure_source.py` should declare as its
schema. The 3-layer validation framework (`_validate_contract.py`) becomes the port's
runtime verification.

### What's Missing

| Gap | Description | Priority |
|-----|-------------|----------|
| **Scenario diagnostic_status** | No field marking whether a scenario comparison is adjudicative vs illustrative (e.g., Oahu H4 NFIP mismatch). Needed for dashboard and paper. | High (paper) |
| **truth_event_match** | No field quantifying how well the truth source (NFIP) matches the modeled event (10-yr surge). Oahu = low. Houston = high. | High (paper) |
| **ReadinessView dataclass** | No structured object passing domain results to rendering adapters. Dashboard currently hardcodes metrics inline. | Medium (post-paper) |
| **Map renderer port** | No abstract interface for rendering. SVG and HTML dashboard scripts are standalone jobs. | Low (post-paper) |

---

## Migration Strategy

1. Ship the paper with current SageMaker scripts (they work).
2. Add `diagnostic_status` and `truth_event_match` to scenario configs immediately (paper need).
3. Extract domain objects post-paper (`ReadinessView`, `ScenarioContract`).
4. Promote `ScenarioRegistry` to a proper port; rename existing implementations as adapters.
5. Wrap existing S3/FEMA/NOAA code as adapters around new ports.
6. Build inbound adapters (CLI first, then Streamlit/API).
