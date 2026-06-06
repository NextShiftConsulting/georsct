"""Port: insurance claims and loss data.

Contract for NFIP claims, 311 reports, high water marks.
Adapter impls: S3 parquet, OpenFEMA API, PostGIS.
"""

from abc import ABC, abstractmethod
from typing import Optional


class ClaimsSource(ABC):
    """Abstract port for claims/loss data access."""

    @abstractmethod
    def load_claims(
        self,
        scenario_id: str,
        spatial_unit: str = "zcta",
    ) -> "pd.DataFrame":
        """Load claims aggregated to spatial unit level.

        Returns DataFrame with at minimum: unit_id, claim_count, claim_amount.
        """

    @abstractmethod
    def load_historical_frequency(
        self,
        unit_ids: list[str],
    ) -> "pd.DataFrame":
        """Load historical claim frequency/severity per unit.

        Returns DataFrame: unit_id, nfip_historical_frequency, nfip_historical_severity.
        """
