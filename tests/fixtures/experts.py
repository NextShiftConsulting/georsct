"""Demo spatial experts for testing.

These are NOT production experts. They return hardcoded features
for benchmark testing. Real flood experts belong in flood/adapters/
backed by real data/source contracts.
"""

from __future__ import annotations

from typing import Any

from georsct.contracts.task_contract import TaskContract
from georsct.ports.spatial_expert import ExpertResult, SpatialExpert
from georsct.provenance.trace import Artifact


class HWMObservationExpert(SpatialExpert):
    """Demo: enriches representation with high-water mark observation coverage."""

    expert_id = "hwm_observation_reliability"
    tool_group = "satellite_eo"
    admissible_geometries = frozenset({"prediction", "ranking", "allocation"})
    addresses = frozenset({"low_target_coverage"})
    known_failure_modes = ("sparse_coverage_overconfidence",)
    preserves = frozenset({"topology", "adjacency"})

    def expected_delta(self, geometry: str) -> float:
        return 0.06

    def run(
        self,
        contract: TaskContract,
        state: dict[str, Any],
    ) -> ExpertResult:
        art = Artifact(
            artifact_id=f"hwm_coverage_{contract.task_id}",
            artifact_type="csv",
            uri=f"s3://swarm-yrsn-checkpoints/georsct/artifacts/hwm/{contract.task_id}.csv",
            source_version="USGS-HWM-v2",
        )
        return ExpertResult(
            features={"hwm_coverage": 0.31},
            compatibility_delta=+0.06,
            artifacts=(art,),
        )


class JRCSurfaceWaterExpert(SpatialExpert):
    """Demo: enriches representation with JRC surface water persistence."""

    expert_id = "jrc_surface_water"
    tool_group = "reanalysis_environmental"
    admissible_geometries = frozenset({"prediction", "ranking"})
    addresses = frozenset({"under_supported_geometry"})
    known_failure_modes = ("autocorrelation_amplification",)
    preserves = frozenset({"topology", "area_proportional"})

    def expected_delta(self, geometry: str) -> float:
        return 0.18

    def run(
        self,
        contract: TaskContract,
        state: dict[str, Any],
    ) -> ExpertResult:
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
