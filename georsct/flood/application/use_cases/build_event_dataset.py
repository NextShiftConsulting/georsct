"""Use case: Build event dataset for a scenario.

Orchestrates adapters through ports to assemble (unit, event) feature matrix.
Post-paper: replaces monolithic jobs/build_event_dataset.py.

Currently: this is a skeleton. The actual logic lives in
jobs/build_event_dataset.py as per the sequencing rule (features now
in monolith, hex refactor after NeurIPS submission).
"""

from georsct.flood.domain.scenario import ScenarioConfig
from georsct.flood.ports.raster_source import RasterSource
from georsct.flood.ports.boundary_source import BoundarySource
from georsct.flood.ports.claims_source import ClaimsSource
from georsct.flood.ports.exposure_source import ExposureSource


def build(
    scenario: ScenarioConfig,
    raster: RasterSource,
    boundary: BoundarySource,
    claims: ClaimsSource,
    exposure: ExposureSource,
) -> "pd.DataFrame":
    """Assemble complete feature matrix for a scenario.

    Orchestration steps:
    1. Load boundaries (ZCTA geometries) via boundary port
    2. Load demographics via exposure port
    3. Load claims/truth data via claims port
    4. Extract raster features (NLCD, DEM, MRMS) via raster port
    5. Compute derived features (cropland_pct, spatial lags) via domain functions
    6. Validate against FEATURE_CONTRACT.yaml schema
    7. Return assembled DataFrame
    """
    # TODO: Port from jobs/build_event_dataset.py after NeurIPS submission.
    raise NotImplementedError("Hex arch build_event_dataset not yet implemented")
