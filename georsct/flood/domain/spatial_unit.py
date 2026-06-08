"""Spatial unit domain objects.

Pure data structures for ZCTA-level spatial units.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpatialUnit:
    """A single spatial unit (ZCTA) with its attributes."""

    zcta_id: str
    geometry: Optional[object] = None  # shapely geometry (injected by adapter)
    county_fips: Optional[str] = None
    state_fips: Optional[str] = None

    @property
    def geoid(self) -> str:
        """Census GEOID construction: state + county + tract/zcta."""
        if self.state_fips and self.county_fips:
            return f"{self.state_fips}{self.county_fips}{self.zcta_id}"
        return self.zcta_id
