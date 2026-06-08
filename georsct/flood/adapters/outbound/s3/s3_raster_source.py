"""S3-backed raster source adapter.

Reads raster data (NLCD, DEM, MRMS) from S3 via rasterio.
Wraps existing S3 cache-check pattern from build_event_dataset.py.

Post-paper: extract S3 raster logic from build_*_features() functions.
"""

# TODO: Extract from build_event_dataset.py after NeurIPS submission.
# Skeleton for the adapter interface:
#
# class S3RasterSource(RasterSource):
#     def __init__(self, bucket: str, profile_name: str = "nsc-swarm"):
#         ...
#     def read_band(self, dataset_id, band=1, window=None): ...
#     def get_transform(self, dataset_id): ...
#     def get_crs(self, dataset_id): ...
#     def clip_to_geometry(self, dataset_id, geometry, band=1): ...
