"""JRC surface water expert — reanalysis environmental data.

Pure-computation demo expert for the GeoRSCT-X benchmark.
Production version should live in flood/adapters/ and inject
an S3 raster source port for actual JRC Global Surface Water data.
"""

from __future__ import annotations

from typing import Any

from georsct.contracts.task_contract import TaskContract
from georsct.experts.base import ExpertResult, SpatialExpert
from georsct.provenance.trace import Artifact


class JRCSurfaceWaterExpert(SpatialExpert):
    """Enriches representation with JRC surface water persistence.

    Addresses under-supported geometry by adding satellite-observed
    historical water presence as a feature.
    """

    expert_id = "jrc_surface_water"
    tool_group = "reanalysis_environmental"
    admissible_geometries = frozenset({"prediction", "ranking"})
    addresses = frozenset({"under_supported_geometry"})
    known_failure_modes = ("autocorrelation_amplification",)

    def expected_delta(self, geometry: str) -> float:
        return 0.18  # historical average from CONUS-27 traces

    def run(
        self,
        contract: TaskContract,
        state: dict[str, Any],
    ) -> ExpertResult:
        # Production: fetch from S3 via raster_source port
        # Benchmark: return mock features
        art = Artifact(
            artifact_id=f"jrc_persistence_{contract.task_id}",
            artifact_type="geotiff",
            uri=f"s3://swarm-yrsn-checkpoints/georsct/artifacts/jrc/{contract.task_id}.tif",
            source_version="JRC-GSW-v1.4",
        )
        return ExpertResult(
            features={"surface_water_persistence": 0.42},
            compatibility_delta=+0.18,
            artifacts=(art,),
        )
