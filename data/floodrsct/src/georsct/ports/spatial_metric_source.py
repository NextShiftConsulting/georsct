"""Port: spatial metric data for quality/compatibility/turbulence computation."""

from abc import ABC, abstractmethod

import numpy as np


class SpatialMetricSource(ABC):

    @abstractmethod
    def get_values(
        self,
        variable: str,
        unit_ids: list[str],
    ) -> np.ndarray:
        """Load variable values for spatial units."""

    @abstractmethod
    def get_coordinates(
        self,
        unit_ids: list[str],
    ) -> list[tuple[float, float]]:
        """Load (x, y) coordinates for GWR/MGWR."""
