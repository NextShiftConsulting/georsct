"""Raster diff analysis -- delegated to rsct_spatial."""

from rsct_spatial.analysis.raster_diff import (  # noqa: F401
    DiffResult,
    compute_frame_diff,
    temporal_profile,
    change_magnitude_map,
    find_hotspots,
)
