"""Spatial expert port — ABC for DMoE-style modular experts.

Admitted on demand by the gearbox. Concrete implementations belong in
flood/adapters/ (backed by real data/source contracts) or in test
fixtures (demo/mock experts).

Import rule: this module imports only from contracts and provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from georsct.contracts.task_contract import TaskContract
from georsct.provenance.trace import Artifact, ExecutionCertificate


class ExpertResult:
    """Result from running a spatial expert.

    Attributes:
        features: New features to merge into the state.
        compatibility_delta: Signed change in kappa_coupling (estimate).
        artifacts: Evidence objects produced.
    """

    __slots__ = ("features", "compatibility_delta", "artifacts")

    def __init__(
        self,
        features: dict[str, Any],
        compatibility_delta: float = 0.0,
        artifacts: tuple[Artifact, ...] = (),
    ):
        self.features = features
        self.compatibility_delta = compatibility_delta
        self.artifacts = artifacts


class SpatialExpert(ABC):
    """Port: a modular spatial expert that can be admitted by the gearbox.

    Subclasses declare what task geometries and weaknesses they address.
    The harness calls ``admissible_for`` before ``run``.

    Concrete experts that need I/O (S3, APIs) should live in
    ``flood/adapters/`` and inject their data sources via ports.
    Pure-computation experts can live directly in test fixtures.
    """

    expert_id: str = "base"
    tool_group: str = "base"
    admissible_geometries: frozenset[str] = frozenset()
    addresses: frozenset[str] = frozenset()
    known_failure_modes: tuple[str, ...] = ()

    def admissible_for(
        self,
        contract: TaskContract,
        cert: ExecutionCertificate,
    ) -> bool:
        """Check whether this expert is admissible for the current state.

        Admissible iff:
          1. The task geometry is in admissible_geometries (or empty = all).
          2. At least one certificate weakness is in addresses.
        """
        if self.admissible_geometries and contract.geometry not in self.admissible_geometries:
            return False
        weakness_types = {w.weakness_type for w in cert.weakness_vector()}
        return bool(weakness_types & self.addresses)

    def expected_delta(self, geometry: str) -> float:
        """Expected compatibility gain for ranking (from historical traces).

        Default: 0.0.  Override in concrete experts or populate from
        GeoCertDB historical data.
        """
        return 0.0

    @abstractmethod
    def run(
        self,
        contract: TaskContract,
        state: dict[str, Any],
    ) -> ExpertResult:
        """Execute the expert and return enrichment results.

        Args:
            contract: The public task contract.
            state: Mutable execution state (scenario + accumulated features).

        Returns:
            ExpertResult with new features, estimated delta, and artifacts.
        """
