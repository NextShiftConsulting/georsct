#!/usr/bin/env python3
"""Diagnostic: compare rasterio vs netCDF4 reads for Deltares events."""
import subprocess, sys
_WHEELS = "/opt/ml/processing/input/wheels"
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--find-links", _WHEELS,
    "sphere-core", "sphere-data", "sphere-flood", "floodcaster",
    "planetary-computer", "pystac-client", "netCDF4",
])

import numpy as np
import planetary_computer
import urllib.request
import netCDF4
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
import tempfile, os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVENTS_BASE = (
    "https://deltaresfloodssa.blob.core.windows.net/floods/v2021.06/events"
)
NO_BBOX = (-90.2, 29.8, -89.8, 30.1)  # xmin, ymin, xmax, ymax

# Download Katrina 90m file
url = f"{EVENTS_BASE}/NASADEM_90m-wm_final/Katrina_2005_masked.nc"
href = planetary_computer.sign(url)

tmpfile = tempfile.mktemp(suffix=".nc")
print("Downloading Katrina event file...", flush=True)
with urllib.request.urlopen(href, timeout=120) as resp:
    size = int(resp.headers.get("Content-Length", 0))
    print(f"  File size: {size / 1024 / 1024:.1f} MB", flush=True)
    with open(tmpfile, "wb") as f:
        f.write(resp.read())
print("Downloaded.", flush=True)

# ===================================================================
# Method A: netCDF4 direct read (known good from Part 2)
# ===================================================================
print("\n=== Method A: netCDF4 direct array slicing ===", flush=True)
ds = netCDF4.Dataset(tmpfile, "r")
lats = ds.variables["lat"][:]
lons = ds.variables["lon"][:]

lat_mask = (lats >= NO_BBOX[1]) & (lats <= NO_BBOX[3])
lon_mask = (lons >= NO_BBOX[0]) & (lons <= NO_BBOX[2])
lat_idx = np.where(lat_mask)[0]
lon_idx = np.where(lon_mask)[0]

r0_nc, r1_nc = lat_idx[0], lat_idx[-1] + 1
c0_nc, c1_nc = lon_idx[0], lon_idx[-1] + 1
print(f"  lat range used: [{float(lats[r0_nc]):.4f}, {float(lats[r1_nc-1]):.4f}]", flush=True)
print(f"  lon range used: [{float(lons[c0_nc]):.4f}, {float(lons[c1_nc-1]):.4f}]", flush=True)
print(f"  row slice: [{r0_nc}:{r1_nc}], col slice: [{c0_nc}:{c1_nc}]", flush=True)

data_nc = ds.variables["inun"][0, r0_nc:r1_nc, c0_nc:c1_nc]
nodata_val = getattr(ds.variables["inun"], '_FillValue', -9999.0)
data_nc = np.where(data_nc == nodata_val, np.nan, data_nc)
ds.close()

valid_nc = data_nc[np.isfinite(data_nc)]
wet_nc = data_nc[data_nc > 0]
print(f"  shape: {data_nc.shape}", flush=True)
print(f"  valid: {len(valid_nc)}/{data_nc.size} ({len(valid_nc)/data_nc.size*100:.1f}%)", flush=True)
print(f"  wet: {len(wet_nc)}/{data_nc.size} ({len(wet_nc)/data_nc.size*100:.1f}%)", flush=True)
if len(wet_nc) > 0:
    print(f"  depth (m): median={np.median(wet_nc):.3f}, max={np.max(wet_nc):.3f}", flush=True)

# ===================================================================
# Method B: rasterio with from_bounds transform
# ===================================================================
print("\n=== Method B: rasterio windowed read (local file) ===", flush=True)
env = rasterio.Env(GDAL_HTTP_UNSAFESSL="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".nc")
with env:
    with rasterio.open(tmpfile) as src:
        h, w = src.height, src.width
        print(f"  rasterio shape: {w}x{h}", flush=True)
        print(f"  rasterio transform: {src.transform}", flush=True)
        print(f"  rasterio nodata: {src.nodata}", flush=True)
        print(f"  rasterio CRS: {src.crs}", flush=True)
        print(f"  rasterio dtypes: {src.dtypes}", flush=True)
        print(f"  rasterio subdatasets: {src.subdatasets}", flush=True)

        # Build transform from known coordinate bounds
        lat_min = float(lats.min())
        lat_max = float(lats.max())
        lon_min = float(lons.min())
        lon_max = float(lons.max())
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
        print(f"  computed transform: {transform}", flush=True)

        # Compute pixel window from bbox
        inv = ~transform
        col_min_px, row_min_px = inv * (NO_BBOX[0], NO_BBOX[3])  # xmin, ymax
        col_max_px, row_max_px = inv * (NO_BBOX[2], NO_BBOX[1])  # xmax, ymin
        print(f"  pixel coords: rows=[{row_min_px:.1f}, {row_max_px:.1f}], "
              f"cols=[{col_min_px:.1f}, {col_max_px:.1f}]", flush=True)

        r0 = max(0, int(row_min_px))
        r1 = min(h, int(row_max_px) + 1)
        c0 = max(0, int(col_min_px))
        c1 = min(w, int(col_max_px) + 1)
        print(f"  clamped window: rows=[{r0}:{r1}], cols=[{c0}:{c1}]", flush=True)

        window = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
        data_rio = np.squeeze(src.read(1, window=window)).astype(np.float64)
        nodata_rio = src.nodata
        if nodata_rio is not None:
            data_rio[data_rio == nodata_rio] = np.nan

print(f"  shape: {data_rio.shape}", flush=True)
valid_rio = data_rio[np.isfinite(data_rio)]
wet_rio = data_rio[data_rio > 0]
print(f"  valid: {len(valid_rio)}/{data_rio.size} ({len(valid_rio)/data_rio.size*100:.1f}%)", flush=True)
print(f"  wet: {len(wet_rio)}/{data_rio.size} ({len(wet_rio)/data_rio.size*100:.1f}%)", flush=True)
if len(wet_rio) > 0:
    print(f"  depth (m): median={np.median(wet_rio):.3f}, max={np.max(wet_rio):.3f}", flush=True)

# ===================================================================
# Method C: Check if lat direction matters (N-to-S vs S-to-N)
# ===================================================================
print("\n=== Coordinate direction check ===", flush=True)
print(f"  lat[0] = {float(lats[0]):.4f} (first = {'south' if lats[0] < lats[-1] else 'north'})", flush=True)
print(f"  lat[-1] = {float(lats[-1]):.4f}", flush=True)
print(f"  lat direction: {'ascending (S-to-N)' if lats[0] < lats[-1] else 'descending (N-to-S)'}", flush=True)
print(f"  lon[0] = {float(lons[0]):.4f}", flush=True)
print(f"  lon[-1] = {float(lons[-1]):.4f}", flush=True)
print(f"  lon direction: {'ascending (W-to-E)' if lons[0] < lons[-1] else 'descending (E-to-W)'}", flush=True)

# netCDF4 row 0 = lat[0]; rasterio row 0 = ?
# Check what rasterio reads at specific pixel coords
print("\n=== Spot checks: rasterio vs netCDF4 at same pixel ===", flush=True)
with env:
    with rasterio.open(tmpfile) as src:
        # Read corners of the full grid
        for label, row, col in [
            ("top-left (0,0)", 0, 0),
            ("top-right (0,11999)", 0, 11999),
            ("bottom-left (11999,0)", 11999, 0),
            ("bottom-right (11999,11999)", 11999, 11999),
            ("center (6000,6000)", 6000, 6000),
        ]:
            w_1px = Window(col_off=col, row_off=row, width=1, height=1)
            rio_val = float(src.read(1, window=w_1px).ravel()[0])
            print(f"  {label}: rasterio={rio_val:.4f}", flush=True)

# Same from netCDF4
ds = netCDF4.Dataset(tmpfile, "r")
inun = ds.variables["inun"]
for label, row, col in [
    ("top-left (0,0)", 0, 0),
    ("top-right (0,11999)", 0, 11999),
    ("bottom-left (11999,0)", 11999, 0),
    ("bottom-right (11999,11999)", 11999, 11999),
    ("center (6000,6000)", 6000, 6000),
]:
    nc_val = float(inun[0, row, col])
    print(f"  {label}: netCDF4={nc_val:.4f}", flush=True)
ds.close()

# ===================================================================
# Method D: rasterio with NETCDF subdataset path
# ===================================================================
print("\n=== Method D: rasterio via NETCDF:path:inun subdataset ===", flush=True)
netcdf_sd = f'NETCDF:"{tmpfile}":inun'
try:
    with rasterio.open(netcdf_sd) as src:
        h2, w2 = src.height, src.width
        print(f"  shape: {w2}x{h2}", flush=True)
        print(f"  transform: {src.transform}", flush=True)
        print(f"  nodata: {src.nodata}", flush=True)
        print(f"  CRS: {src.crs}", flush=True)

        # Build transform + window
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w2, h2)
        inv = ~transform
        col_min_px, row_min_px = inv * (NO_BBOX[0], NO_BBOX[3])
        col_max_px, row_max_px = inv * (NO_BBOX[2], NO_BBOX[1])
        r0 = max(0, int(row_min_px))
        r1 = min(h2, int(row_max_px) + 1)
        c0 = max(0, int(col_min_px))
        c1 = min(w2, int(col_max_px) + 1)
        window = Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
        data_sd = np.squeeze(src.read(1, window=window)).astype(np.float64)
        nd = src.nodata
        if nd is not None:
            data_sd[data_sd == nd] = np.nan
        valid_sd = data_sd[np.isfinite(data_sd)]
        wet_sd = data_sd[data_sd > 0]
        print(f"  window shape: {data_sd.shape}", flush=True)
        print(f"  valid: {len(valid_sd)}/{data_sd.size}", flush=True)
        print(f"  wet: {len(wet_sd)}/{data_sd.size}", flush=True)
        if len(wet_sd) > 0:
            print(f"  depth (m): median={np.median(wet_sd):.3f}, max={np.max(wet_sd):.3f}", flush=True)
except Exception as e:
    print(f"  ERROR: {e}", flush=True)

os.unlink(tmpfile)
print("\n=== Done ===", flush=True)
