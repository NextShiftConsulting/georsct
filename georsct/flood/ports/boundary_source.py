"""Port: vector boundary data access.

Contract for loading spatial boundaries (ZCTAs, counties, watersheds).
Adapter impls: S3/GeoJSON, PostGIS, Census TIGER API.
Pattern: Fiona/GeoPandas read (AGDS Ch.4, LGAP Ch.4).
"""

from abc import ABC, abstractmethod
from typing import Optional


class BoundarySource(ABC):
    """Abstract port for vector boundary access."""

    @abstractmethod
    def load_boundaries(
        self,
        boundary_type: str,
        region_filter: Optional[dict] = None,
    ) -> "gpd.GeoDataFrame":
        """Load boundary geometries.

        Args:
            boundary_type: 'zcta', 'county', 'tract', 'watershed', 'catchment'.
            region_filter: Optional dict of filter criteria
                           (e.g., {'state_fips': '48'}).
        """

    @abstractmethod
    def load_adjacency(
        self,
        boundary_type: str,
        region_filter: Optional[dict] = None,
    ) -> "pd.DataFrame":
        """Load adjacency edge list for spatial weights construction.

        Returns DataFrame with columns [source_id, target_id].
        """
