"""Exposure and feature derivation via spatial operations.

Pure functions -- no I/O, no S3, no SQL.
Source patterns: AGDS Ch.7 (Spatial Feature Engineering -- buffer aggregation,
distance bands, proximity features).
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_buffer_fraction(
    unit_geometry: "shapely.geometry.base.BaseGeometry",
    raster_values: np.ndarray,
    raster_transform: "affine.Affine",
    target_classes: list[int],
    buffer_m: float = 0.0,
) -> float:
    """Fraction of raster pixels within a geometry matching target classes.

    Pattern: buffer + raster zonal stats (AGDS Ch.7).
    Used for: cropland_pct (classes 81,82), impervious_pct, wetland_pct.

    Args:
        unit_geometry: Polygon (ZCTA boundary), optionally buffered.
        raster_values: 2D raster array (e.g., NLCD land cover).
        raster_transform: Affine geotransform for the raster.
        target_classes: Pixel values to count as positive.
        buffer_m: Buffer distance in meters (0 = exact boundary).
    """
    from rasterio.features import geometry_mask

    if buffer_m > 0:
        unit_geometry = unit_geometry.buffer(buffer_m)

    mask = geometry_mask(
        [unit_geometry],
        out_shape=raster_values.shape,
        transform=raster_transform,
        invert=True,
    )
    pixels_in = raster_values[mask]
    if len(pixels_in) == 0:
        return 0.0

    return float(np.isin(pixels_in, target_classes).sum() / len(pixels_in))


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index.

    Source: LGAP Ch.9 B19730_09_01-ndvi.py.
    NDVI = (NIR - Red) / (NIR + Red + epsilon).
    """
    epsilon = 1e-5
    return ((nir.astype(np.float32) - red.astype(np.float32))
            / (nir.astype(np.float32) + red.astype(np.float32) + epsilon))


def compute_spatial_lag(
    values: np.ndarray,
    w: "libpysal.weights.W",
) -> np.ndarray:
    """Spatially lagged variable using weights matrix.

    Pattern: weights.spatial_lag.lag_spatial (AGDS Ch.6).
    Used for: wlag_cropland_pct, wlag_impervious_pct.
    """
    from pysal.lib.weights.spatial_lag import lag_spatial

    return lag_spatial(w, values)


def compute_distance_to_feature(
    unit_centroids: "gpd.GeoSeries",
    feature_geometry: "shapely.geometry.base.BaseGeometry",
) -> np.ndarray:
    """Distance from each unit centroid to nearest feature geometry.

    Pattern: distance band computation (AGDS Ch.7).
    Used for: coastal_distance_m, hospital_distance_km.
    """
    return np.array([c.distance(feature_geometry) for c in unit_centroids])


def compute_point_density(
    unit_geometry: "shapely.geometry.base.BaseGeometry",
    points: "gpd.GeoDataFrame",
    buffer_m: float = 1000.0,
) -> int:
    """Count of points within buffer of a spatial unit.

    Pattern: buffer + sjoin counting (AGDS Ch.7).
    Used for: hifld_n_hospitals, n_311_reports.
    """
    import geopandas as gpd

    buffered = unit_geometry.buffer(buffer_m)
    within = points[points.geometry.within(buffered)]
    return len(within)
