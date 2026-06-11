#!/usr/bin/env python3
"""Diagnostic: dump ALL GDAL metadata from Deltares event NetCDF."""
import subprocess, sys
_WHEELS = "/opt/ml/processing/input/wheels"
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--find-links", _WHEELS,
    "sphere-core", "sphere-data", "sphere-flood", "floodcaster",
    "planetary-computer", "pystac-client", "netCDF4", "h5netcdf",
])

import numpy as np
import planetary_computer
import rasterio
from rasterio.windows import Window

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("=== Deltares Event Metadata Dump ===", flush=True)

# Sign Katrina event URL
href = planetary_computer.sign(
    "https://deltaresfloodssa.blob.core.windows.net/floods/v2021.06/"
    "events/NASADEM_90m-wm_final/Katrina_2005_masked.nc"
)

env = rasterio.Env(GDAL_HTTP_UNSAFESSL="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".nc")
with env:
    with rasterio.open(href) as src:
        print(f"Shape: {src.width}x{src.height}", flush=True)
        print(f"CRS: {src.crs}", flush=True)
        print(f"Transform: {src.transform}", flush=True)
        print(f"Nodata: {src.nodata}", flush=True)
        print(f"Dtypes: {src.dtypes}", flush=True)
        print(f"Subdatasets: {src.subdatasets}", flush=True)
        print(f"\nDefault tags:", flush=True)
        for k, v in sorted(src.tags().items()):
            print(f"  {k} = {v[:200] if len(str(v)) > 200 else v}", flush=True)
        # Check all GDAL metadata domains
        for domain in ["", "IMAGE_STRUCTURE", "SUBDATASETS",
                        "GEOLOCATION", "DERIVED_SUBDATASETS", "xml:NC_GLOBAL"]:
            tags = src.tags(ns=domain)
            if tags:
                print(f"\nTags (ns={domain!r}):", flush=True)
                for k, v in sorted(tags.items()):
                    vstr = str(v)
                    print(f"  {k} = {vstr[:300]}", flush=True)

# Try netCDF4 directly via HTTP
print("\n=== netCDF4 direct read ===", flush=True)
try:
    import netCDF4
    # netCDF4 can open URLs via OPeNDAP or local files
    # For HTTP, we need to download first or use a different approach
    # Try reading via GDAL's virtual filesystem
    print("netCDF4 installed, trying via vsicurl...", flush=True)

    # Download just the header to get coordinates
    import urllib.request
    import tempfile, os

    # Download the first 10 MB which should contain coords
    print("Downloading partial file for coordinate extraction...", flush=True)
    tmpfile = "/tmp/katrina_partial.nc"
    req = urllib.request.Request(href)
    # Read the whole file (event files are small ~= regional crop)
    with urllib.request.urlopen(req, timeout=120) as resp:
        size = int(resp.headers.get("Content-Length", 0))
        print(f"  File size: {size / 1024 / 1024:.1f} MB", flush=True)
        with open(tmpfile, "wb") as f:
            f.write(resp.read())

    ds = netCDF4.Dataset(tmpfile, "r")
    print(f"\nDimensions:", flush=True)
    for name, dim in ds.dimensions.items():
        print(f"  {name}: size={len(dim)}, unlimited={dim.isunlimited()}", flush=True)
    print(f"\nVariables:", flush=True)
    for name, var in ds.variables.items():
        print(f"  {name}: dims={var.dimensions}, shape={var.shape}, dtype={var.dtype}", flush=True)
        if name in ("lat", "lon", "latitude", "longitude", "x", "y"):
            vals = var[:]
            print(f"    range: [{vals.min():.4f}, {vals.max():.4f}]", flush=True)
            print(f"    first 5: {vals[:5]}", flush=True)
            print(f"    last 5: {vals[-5:]}", flush=True)
    ds.close()
    os.unlink(tmpfile)

except Exception as e:
    print(f"netCDF4 error: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n=== Done ===", flush=True)
