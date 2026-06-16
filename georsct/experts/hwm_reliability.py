"""HWM observation reliability expert — satellite EO data.

Pure-computation demo expert for the GeoRSCT-X benchmark.
Production version should live in flood/adapters/ and inject
a claims_source port for USGS high-water mark observations.
"""

from __future__ import annotations

from typing import Any

from georsct.contracts.task_contract import TaskContract
from georsct.experts.base import ExpertResult, SpatialExpert
from georsct.provenance.trace import Artifact


class HWMObservationExpert(SpatialExpert):
    """Enriches representation with high-water mark observation coverage.

    Addresses low target coverage by adding HWM spatial coverage flags.
    """

    expert_id = "hwm_observation_reliability"
    tool_group = "satellite_eo"
    admissible_geometries = frozenset({"prediction", "ranking", "allocation"})
    addresses = frozenset({"low_target_coverage"})
    known_failure_modes = ("sparse_coverage_overconfidence",)

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
