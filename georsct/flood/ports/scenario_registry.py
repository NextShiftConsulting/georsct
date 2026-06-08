"""Port: scenario configuration registry.

Contract for loading scenario definitions.
Adapter impls: YAML files, PostGIS table, DuckDB.
Existing impl: rsct-geocert/db/scripts/scenario_registry.py (already a port pattern).
"""

from abc import ABC, abstractmethod
from typing import Optional

from georsct.flood.domain.scenario import ScenarioConfig


class ScenarioRegistry(ABC):
    """Abstract port for scenario lookup."""

    @abstractmethod
    def get(self, scenario_id: str) -> Optional[ScenarioConfig]:
        """Load a single scenario by ID."""

    @abstractmethod
    def list_all(self) -> list[ScenarioConfig]:
        """List all registered scenarios."""

    @abstractmethod
    def list_by_region(self, region: str) -> list[ScenarioConfig]:
        """List scenarios for a geographic region."""
