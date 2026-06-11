"""Port: construct-specific target data for five-construct divergence.

Each flood construct has a different target column and possibly a
different source file.  This port abstracts the data loading so the
application use case does not know about S3 or file formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from georsct.domain.construct_certificate import ConstructLabel


@dataclass(frozen=True)
class ConstructData:
    """Target values for one construct, one geography.

    When ``available`` is False, ``target_values`` and ``region_ids``
    are None and ``reason`` explains why.
    """

    construct: ConstructLabel
    target_values: Optional[np.ndarray]  # (n_obs,) or None
    region_ids: Optional[np.ndarray]     # (n_obs,) str or None
    available: bool
    reason: str  # "ok" or explanation of unavailability

    class Config:
        arbitrary_types_allowed = True


class ConstructDataSource(ABC):
    """Port for loading construct-specific target data.

    Implementations handle the specifics of where each construct's
    data lives (same parquet vs separate file, S3 vs local, etc.).
    """

    @abstractmethod
    def load_construct_target(
        self,
        construct: ConstructLabel,
        scenario_id: str,
        event_id: Optional[str] = None,
    ) -> ConstructData:
        """Load target values for one construct.

        Args:
            construct: Which flood construct.
            scenario_id: Scenario key (e.g., "houston").
            event_id: Optional event filter (e.g., "harvey2017").
                Some constructs are static (JRC, Deltares, FEMA)
                and ignore this parameter.

        Returns:
            ConstructData with target values or unavailability reason.
        """

    @abstractmethod
    def available_constructs(
        self,
        scenario_id: str,
    ) -> list[ConstructLabel]:
        """List constructs with data available for a scenario.

        Returns:
            List of ConstructLabels that can be loaded.
        """
