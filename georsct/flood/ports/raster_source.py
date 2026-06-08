"""Port: raster data access.

Contract for reading raster bands (NLCD, DEM, MRMS, satellite).
Adapter impls: S3, local file, USGS API, NOAA API.
Pattern: rasterio band access (AGDS Ch.4, LGAP Ch.4).
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class RasterSource(ABC):
    """Abstract port for raster data access."""

    @abstractmethod
    def read_band(
        self,
        dataset_id: str,
        band: int = 1,
        window: Optional[tuple] = None,
    ) -> np.ndarray:
        """Read a single raster band, optionally windowed."""

    @abstractmethod
    def get_transform(self, dataset_id: str) -> "affine.Affine":
        """Return the affine geotransform for a dataset."""

    @abstractmethod
    def get_crs(self, dataset_id: str) -> str:
        """Return the CRS string (e.g., 'EPSG:4326')."""

    @abstractmethod
    def clip_to_geometry(
        self,
        dataset_id: str,
        geometry: "shapely.geometry.base.BaseGeometry",
        band: int = 1,
    ) -> tuple[np.ndarray, "affine.Affine"]:
        """Clip raster to geometry boundary. Return (array, transform)."""
