#!/usr/bin/env python3
"""Diagnostic: test Deltares event depth extraction with netCDF4 strategy."""
import subprocess, sys
_WHEELS = "/opt/ml/processing/input/wheels"
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--find-links", _WHEELS,
    "sphere-core", "sphere-data", "sphere-flood", "floodcaster",
    "planetary-computer", "pystac-client", "netCDF4",
])

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("=== Deltares Event Depth Extraction Test ===", flush=True)

from floodcaster.sources import BBox
from floodcaster.stac import get_deltares_event_depth

# Test events with bboxes that should be inside the event coverage
TEST_CASES = [
    # (storm_name, year, bbox_description, bbox)
    ("Katrina", 2005, "New Orleans metro",
     BBox(-90.2, 29.8, -89.8, 30.1)),
    ("Sandy", 2012, "NYC metro",
     BBox(-74.1, 40.5, -73.7, 40.9)),
    ("Irma", 2017, "SW Florida (Fort Myers)",
     BBox(-82.0, 26.4, -81.6, 26.8)),
    ("Ike", 2008, "Houston/Galveston",
     BBox(-95.0, 29.2, -94.6, 29.6)),
]

for storm, year, desc, bbox in TEST_CASES:
    print(f"\n--- {storm} {year} ({desc}) ---", flush=True)
    print(f"  bbox: ({bbox.xmin:.2f}, {bbox.ymin:.2f}, "
          f"{bbox.xmax:.2f}, {bbox.ymax:.2f})", flush=True)
    try:
        depth_ft, transform = get_deltares_event_depth(
            bbox, storm_name=storm, storm_year=year,
        )
        valid = depth_ft[np.isfinite(depth_ft)]
        wet = depth_ft[depth_ft > 0]
        print(f"  shape: {depth_ft.shape}", flush=True)
        print(f"  transform: {transform}", flush=True)
        print(f"  valid pixels: {len(valid)}/{depth_ft.size}", flush=True)
        print(f"  wet pixels: {len(wet)}/{depth_ft.size} "
              f"({len(wet)/max(depth_ft.size,1)*100:.1f}%)", flush=True)
        if len(wet) > 0:
            print(f"  depth (ft): median={np.median(wet):.2f}, "
                  f"max={np.max(wet):.2f}", flush=True)
        else:
            print("  depth: no inundated pixels", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\n=== Done ===", flush=True)
