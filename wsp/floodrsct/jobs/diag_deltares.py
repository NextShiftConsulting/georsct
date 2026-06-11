#!/usr/bin/env python3
"""Diagnostic: enumerate Deltares file variants and test coverage."""
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVENTS_BASE = (
    "https://deltaresfloodssa.blob.core.windows.net/floods/v2021.06/events"
)

STORMS = [
    ("Katrina", 2005), ("Sandy", 2012), ("Irma", 2017), ("Ike", 2008),
    ("Harvey", 2017), ("Ian", 2022), ("Ida", 2021), ("Michael", 2018),
]

# DEM directories to probe
DEM_DIRS = [
    "NASADEM_90m-wm_final",
    "NASADEM_1km-wm_final",
    "MERITDEM_90m-wm_final",
    "MERITDEM_1km-wm_final",
    "LIDAR_90m-wm_final",
    "LIDAR_1km-wm_final",
    # Try without -wm suffix
    "NASADEM_90m_final",
    "NASADEM_1km_final",
]

# File suffixes to probe
SUFFIXES = ["_masked.nc", ".nc", "_unmasked.nc"]


def probe_url(url):
    """Check if a URL exists (HTTP HEAD). Returns (exists, size_mb)."""
    signed = planetary_computer.sign(url)
    req = urllib.request.Request(signed, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            size = int(resp.headers.get("Content-Length", 0))
            return True, size / 1024 / 1024
    except urllib.error.HTTPError:
        return False, 0
    except Exception:
        return False, 0


# ===================================================================
# Part 1: Enumerate available event file variants
# ===================================================================
print("=" * 70, flush=True)
print("PART 1: Enumerate Deltares event file variants", flush=True)
print("=" * 70, flush=True)

found_files = []
for storm, year in STORMS:
    print(f"\n--- {storm} {year} ---", flush=True)
    for dem_dir in DEM_DIRS:
        for suffix in SUFFIXES:
            filename = f"{storm}_{year}{suffix}"
            url = f"{EVENTS_BASE}/{dem_dir}/{filename}"
            exists, size_mb = probe_url(url)
            if exists:
                print(f"  FOUND: {dem_dir}/{filename} ({size_mb:.1f} MB)", flush=True)
                found_files.append((storm, year, dem_dir, suffix, size_mb))
            # Don't print misses to keep output short

print(f"\n\nTotal files found: {len(found_files)}", flush=True)
for storm, year, dem_dir, suffix, size_mb in found_files:
    print(f"  {storm}_{year}: {dem_dir}{suffix} ({size_mb:.1f} MB)", flush=True)


# ===================================================================
# Part 2: For files that exist, check if UNMASKED has more coverage
# ===================================================================
print("\n" + "=" * 70, flush=True)
print("PART 2: Compare masked vs unmasked coverage (New Orleans bbox)", flush=True)
print("=" * 70, flush=True)

from floodcaster.sources import BBox
from floodcaster.stac import get_deltares_event_depth

NO_BBOX = BBox(-90.2, 29.8, -89.8, 30.1)

# If unmasked Katrina exists, test it
for storm, year, dem_dir, suffix, size_mb in found_files:
    if storm == "Katrina":
        print(f"\n--- {storm}_{year} ({dem_dir}, suffix={suffix}) ---", flush=True)
        # Build the URL manually for non-standard variants
        filename = f"{storm}_{year}{suffix}"
        url = f"{EVENTS_BASE}/{dem_dir}/{filename}"
        href = planetary_computer.sign(url)

        # Download and check with netCDF4
        import netCDF4, tempfile, os
        tmpfile = tempfile.mktemp(suffix=".nc")
        try:
            with urllib.request.urlopen(href, timeout=120) as resp:
                with open(tmpfile, "wb") as f:
                    f.write(resp.read())
            ds = netCDF4.Dataset(tmpfile, "r")

            # Find the data variable
            data_var = None
            for name in ds.variables:
                if name not in ("lat", "lon", "latitude", "longitude",
                                "time", "projection", "x", "y", "crs"):
                    data_var = name
                    break

            if data_var:
                var = ds.variables[data_var]
                print(f"  data var: {data_var}, shape={var.shape}", flush=True)

                # Get lat/lon bounds
                lat_var = ds.variables.get("lat") or ds.variables.get("latitude")
                lon_var = ds.variables.get("lon") or ds.variables.get("longitude")
                if lat_var and lon_var:
                    lat_min, lat_max = float(lat_var[:].min()), float(lat_var[:].max())
                    lon_min, lon_max = float(lon_var[:].min()), float(lon_var[:].max())
                    print(f"  lat: [{lat_min:.2f}, {lat_max:.2f}]", flush=True)
                    print(f"  lon: [{lon_min:.2f}, {lon_max:.2f}]", flush=True)

                # Extract New Orleans window
                if lat_var is not None and lon_var is not None:
                    lats = lat_var[:]
                    lons = lon_var[:]

                    # Find indices for NO bbox
                    lat_mask = (lats >= NO_BBOX.ymin) & (lats <= NO_BBOX.ymax)
                    lon_mask = (lons >= NO_BBOX.xmin) & (lons <= NO_BBOX.xmax)
                    lat_idx = np.where(lat_mask)[0]
                    lon_idx = np.where(lon_mask)[0]

                    if len(lat_idx) > 0 and len(lon_idx) > 0:
                        r0, r1 = lat_idx[0], lat_idx[-1] + 1
                        c0, c1 = lon_idx[0], lon_idx[-1] + 1
                        # Read the data variable
                        if len(var.shape) == 3:
                            data = var[0, r0:r1, c0:c1]
                        else:
                            data = var[r0:r1, c0:c1]
                        nodata = getattr(var, '_FillValue', -9999.0)
                        data = np.where(data == nodata, np.nan, data)

                        valid = data[np.isfinite(data)]
                        wet = data[data > 0]
                        total = data.size
                        print(f"  window: {data.shape}", flush=True)
                        print(f"  valid: {len(valid)}/{total} "
                              f"({len(valid)/max(total,1)*100:.1f}%)", flush=True)
                        print(f"  wet: {len(wet)}/{total} "
                              f"({len(wet)/max(total,1)*100:.1f}%)", flush=True)
                        if len(wet) > 0:
                            print(f"  depth (m): median={np.median(wet):.3f}, "
                                  f"max={np.max(wet):.3f}", flush=True)
                    else:
                        print("  NO bbox outside file coverage", flush=True)

            ds.close()
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)


# ===================================================================
# Part 3: STAC catalog - what query properties are available?
# ===================================================================
print("\n" + "=" * 70, flush=True)
print("PART 3: STAC catalog query properties (deltares-floods)", flush=True)
print("=" * 70, flush=True)

import pystac_client

client = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search with broad bbox covering New Orleans
search = client.search(
    collections=["deltares-floods"],
    bbox=(-90.2, 29.8, -89.8, 30.1),
    max_items=200,
)
items = list(search.items())
print(f"\nTotal STAC items for NO bbox: {len(items)}", flush=True)

# Collect unique property values
prop_values = {}
for item in items:
    for key, val in item.properties.items():
        if key.startswith("deltares:"):
            prop_values.setdefault(key, set()).add(str(val))

print("\nAvailable STAC query properties:", flush=True)
for key in sorted(prop_values.keys()):
    vals = sorted(prop_values[key])
    print(f"  {key}: {vals}", flush=True)

# Show asset keys from first item
if items:
    item0 = items[0]
    print(f"\nAsset keys in first item ({item0.id}):", flush=True)
    for ak, asset in item0.assets.items():
        print(f"  {ak}: {asset.title or ''} "
              f"({asset.media_type or 'unknown type'})", flush=True)
    print(f"\nAll properties:", flush=True)
    for k, v in sorted(item0.properties.items()):
        vstr = str(v)
        print(f"  {k}: {vstr[:120]}", flush=True)

# Test: how many items per return period?
print("\n\nItems per return period:", flush=True)
for rp in [2, 5, 10, 25, 50, 100, 250]:
    count = sum(1 for i in items
                if i.properties.get("deltares:return_period") == rp)
    print(f"  RP {rp:3d}: {count} items", flush=True)

# Test: how many items per DEM?
print("\nItems per DEM:", flush=True)
dems = {}
for i in items:
    d = i.properties.get("deltares:dem_name", "unknown")
    dems[d] = dems.get(d, 0) + 1
for d, c in sorted(dems.items()):
    print(f"  {d}: {c} items", flush=True)

# Test: how many items per scenario?
print("\nItems per sea_level_year:", flush=True)
years = {}
for i in items:
    y = i.properties.get("deltares:sea_level_year", "unknown")
    years[y] = years.get(y, 0) + 1
for y, c in sorted(years.items(), key=lambda x: str(x[0])):
    print(f"  {y}: {c} items", flush=True)

print("\n=== Done ===", flush=True)
