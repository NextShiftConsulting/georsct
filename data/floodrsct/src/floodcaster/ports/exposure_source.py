"""Port: exposure data (structures, population, infrastructure).

Contract for NSI structures, Census ACS, HIFLD facilities.
Adapter impls: S3, Census API, PostGIS.
Schema aligns with FEATURE_CONTRACT.yaml fields.
"""

from abc import ABC, abstractmethod


class ExposureSource(ABC):
    """Abstract port for exposure/vulnerability data."""

    @abstractmethod
    def load_demographics(
        self,
        unit_ids: list[str],
    ) -> "pd.DataFrame":
        """Load ACS demographic features per spatial unit.

        Expected columns per FEATURE_CONTRACT.yaml:
        acs_total_pop, acs_median_hh_income, acs_pct_below_poverty,
        acs_pct_renter_occupied, acs_pct_vacant, etc.
        """

    @abstractmethod
    def load_structures(
        self,
        unit_ids: list[str],
    ) -> "pd.DataFrame":
        """Load NSI structure inventory per spatial unit.

        Expected columns: unit_id, n_structures, structure_value_total.
        """

    @abstractmethod
    def load_vulnerability_index(
        self,
        unit_ids: list[str],
    ) -> "pd.DataFrame":
        """Load SVI (Social Vulnerability Index) per spatial unit.

        Expected columns per FEATURE_CONTRACT.yaml:
        svi_overall, svi_socioeconomic, svi_household_disability,
        svi_minority_language, svi_housing_transport.
        """
