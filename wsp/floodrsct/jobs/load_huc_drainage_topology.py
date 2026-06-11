#!/usr/bin/env python3
"""
load_huc_drainage_topology.py

Build pluggable reference topologies for GeoRSCT / FloodRSCT kappa_reconstruct.

Purpose
-------
kappa_reconstruct needs a reference spatial support. For general GeoRSCT, shared
border or centroid distance is often acceptable. For FloodRSCT, the more faithful
reference is hydrological topology: HUC/catchment co-membership and upstream /
downstream drainage connectivity.

This script emits:
  1. reference_edges.csv   columns: src,dst,weight,relation
  2. region_coords.csv     columns: region_id,x,y
  3. topology_metadata.json

The output can be passed to the kappa_reconstruct registry or the s035
adversarial_reconstruct adapter as the reference edge graph.

Supported topology modes
------------------------
shared_border:
    Queen-contiguity between region polygons.

shared_huc:
    Regions are connected when they share dominant or overlapping HUC membership.

drainage:
    Regions are connected when their dominant HUCs are connected by a drainage
    edge table.

centroid_distance:
    Symmetric k-NN graph on projected centroids with Gaussian distance weights.
    Geometry-only null: preserves Tobler's first law but does not encode
    border geometry or hydrological structure. Use for convergent validity
    and to pre-register reconstruct_floor.

combined:
    Union of shared_border, shared_huc, and drainage when available.

Expected input files
--------------------
--regions:
    GeoPackage/Shapefile/GeoJSON/Parquet readable by geopandas with region
    polygons and a region id column.

--huc-polygons:
    Optional HUC/catchment polygons readable by geopandas.

--drainage-edges:
    Optional CSV with [from_huc, to_huc, weight] or configured column names.

Examples
--------
# Base shared-border topology
python load_huc_drainage_topology.py \\
  --regions processed/houston/zcta_boundaries.gpkg \\
  --region-id zcta \\
  --topology shared_border \\
  --out-dir artifacts/topology_houston_shared_border

# Centroid-distance null baseline
python load_huc_drainage_topology.py \\
  --regions processed/houston/zcta_boundaries.gpkg \\
  --region-id zcta \\
  --topology centroid_distance --centroid-k 8 \\
  --out-dir artifacts/topology_houston_centroid_null

# Flood-faithful HUC/drainage topology
python load_huc_drainage_topology.py \\
  --regions processed/houston/zcta_boundaries.gpkg \\
  --region-id zcta \\
  --huc-polygons processed/houston/huc12_catchments.gpkg \\
  --huc-id huc12 \\
  --drainage-edges processed/houston/huc12_flow_edges.csv \\
  --topology combined \\
  --out-dir artifacts/topology_houston_huc_drainage
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "requires geopandas: pip install geopandas pyogrio"
    ) from exc


TOPOLOGY_CHOICES = [
    "shared_border", "shared_huc", "drainage", "centroid_distance", "combined",
]


@dataclass
class TopologyMetadata:
    topology: str
    n_regions: int
    n_edges: int
    region_id: str
    huc_id: Optional[str]
    regions_path: str
    huc_polygons_path: Optional[str]
    drainage_edges_path: Optional[str]
    crs: str
    notes: str


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build HUC/drainage reference topology for kappa_reconstruct.",
    )
    p.add_argument("--regions", required=True, type=Path)
    p.add_argument("--region-id", default="region_id")
    p.add_argument("--out-dir", required=True, type=Path)

    p.add_argument("--topology", choices=TOPOLOGY_CHOICES, default="combined")

    p.add_argument("--huc-polygons", type=Path, default=None)
    p.add_argument("--huc-id", default="huc_id")
    p.add_argument("--min-overlap-frac", type=float, default=0.05)

    p.add_argument("--drainage-edges", type=Path, default=None)
    p.add_argument("--from-huc-col", default="from_huc")
    p.add_argument("--to-huc-col", default="to_huc")
    p.add_argument("--weight-col", default="weight")

    p.add_argument("--centroid-k", type=int, default=8,
                   help="k for centroid_distance kNN graph")
    p.add_argument("--centroid-sigma", type=float, default=None,
                   help="Gaussian sigma for centroid_distance weights "
                        "(default: median kNN distance)")

    p.add_argument("--project-crs", default=None,
                   help="Projected CRS for area/centroid work, e.g. EPSG:5070.")
    p.add_argument("--simplify-tolerance", type=float, default=0.0)
    return p.parse_args()


# =========================================================================
# Geometry helpers
# =========================================================================

def read_geodata(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No features read from {path}")
    if "geometry" not in gdf.columns:
        raise ValueError(f"{path} has no geometry column.")
    return gdf


def normalize_region_ids(gdf: gpd.GeoDataFrame, region_id: str) -> gpd.GeoDataFrame:
    if region_id not in gdf.columns:
        raise ValueError(f"Missing region id column: {region_id}")
    out = gdf.copy()
    out[region_id] = out[region_id].astype(str)
    out = out[[region_id, "geometry"]].dropna(subset=["geometry"])
    out = out.dissolve(by=region_id, as_index=False)
    return out


def choose_projected_crs(gdf: gpd.GeoDataFrame, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if gdf.crs is None:
        raise ValueError("Input CRS is missing. Provide --project-crs.")
    return "EPSG:5070"  # CONUS Albers default


def make_region_coords(
    regions_proj: gpd.GeoDataFrame,
    region_id: str,
) -> pd.DataFrame:
    cent = regions_proj.geometry.centroid
    return pd.DataFrame({
        region_id: regions_proj[region_id].astype(str).to_numpy(),
        "x": cent.x.to_numpy(dtype=float),
        "y": cent.y.to_numpy(dtype=float),
    })


# =========================================================================
# Edge helpers
# =========================================================================

def _canonical_edge(src: str, dst: str) -> tuple[str, str]:
    return tuple(sorted((str(src), str(dst))))


def _edge_frame(edge_weights: dict[tuple[str, str, str], float]) -> pd.DataFrame:
    rows = []
    for (src, dst, relation), weight in edge_weights.items():
        if src == dst:
            continue
        rows.append({
            "src": src, "dst": dst,
            "weight": float(weight), "relation": relation,
        })
    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight", "relation"])
    return (
        pd.DataFrame(rows)
        .sort_values(["relation", "src", "dst"])
        .reset_index(drop=True)
    )


# =========================================================================
# Topology builders
# =========================================================================

def build_shared_border_edges(
    regions_proj: gpd.GeoDataFrame,
    region_id: str,
) -> pd.DataFrame:
    """Queen contiguity between region polygons."""
    sindex = regions_proj.sindex
    edge_weights: dict[tuple[str, str, str], float] = {}

    for i, geom in enumerate(regions_proj.geometry):
        rid_i = str(regions_proj.iloc[i][region_id])
        candidates = list(sindex.query(geom, predicate="intersects"))
        for j in candidates:
            if j <= i:
                continue
            rid_j = str(regions_proj.iloc[j][region_id])
            other = regions_proj.iloc[j].geometry

            if not geom.touches(other) and not geom.intersects(other):
                continue

            inter = geom.boundary.intersection(other.boundary)
            weight = float(inter.length) if not inter.is_empty else 1.0
            if weight <= 0:
                weight = 1.0

            src, dst = _canonical_edge(rid_i, rid_j)
            key = (src, dst, "shared_border")
            edge_weights[key] = max(edge_weights.get(key, 0.0), weight)

    return _edge_frame(edge_weights)


def build_centroid_distance_edges(
    coords: pd.DataFrame,
    region_id: str,
    k: int = 8,
    sigma: Optional[float] = None,
) -> pd.DataFrame:
    """Symmetric k-NN graph on projected centroids with Gaussian weights.

    Geometry-only null topology. Preserves Tobler's first law (nearby things
    are related) but does not encode border geometry or hydrological
    structure. Use for convergent validity. Expected ordering under
    flood-faithful targets: huc_drainage >= shared_border >= centroid_distance.
    If centroid_distance wins, the target may be driven by broad geographic
    proximity rather than drainage topology, or the HUC loader is too coarse.
    """
    ids = coords[region_id].astype(str).to_numpy()
    P = coords[["x", "y"]].to_numpy(dtype=float)
    n = len(ids)
    if n < 2:
        return pd.DataFrame(columns=["src", "dst", "weight", "relation"])

    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    kk = min(k, n - 1)

    nbr_d = np.sort(D, axis=1)[:, :kk]
    if sigma is None:
        sigma = float(np.median(nbr_d[np.isfinite(nbr_d)]))
        sigma = sigma if sigma > 0 else 1.0

    weights: dict[tuple[str, str], float] = {}
    nbr_idx = np.argsort(D, axis=1)[:, :kk]
    for i in range(n):
        for j in nbr_idx[i]:
            a, b = _canonical_edge(ids[i], ids[int(j)])
            if a == b:
                continue
            w = float(np.exp(-(D[i, int(j)] ** 2) / (2 * sigma ** 2)))
            weights[(a, b)] = max(weights.get((a, b), 0.0), w)

    rows = [
        {"src": a, "dst": b, "weight": w, "relation": "centroid_distance"}
        for (a, b), w in weights.items()
    ]
    return (
        pd.DataFrame(rows, columns=["src", "dst", "weight", "relation"])
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
    )


def compute_region_huc_membership(
    regions_proj: gpd.GeoDataFrame,
    hucs_proj: gpd.GeoDataFrame,
    region_id: str,
    huc_id: str,
    min_overlap_frac: float,
) -> pd.DataFrame:
    """Compute fractional overlap of each region with each HUC."""
    if huc_id not in hucs_proj.columns:
        raise ValueError(f"Missing HUC id column: {huc_id}")

    hucs = hucs_proj[[huc_id, "geometry"]].dropna(subset=["geometry"]).copy()
    hucs[huc_id] = hucs[huc_id].astype(str)

    regions_area = regions_proj[[region_id, "geometry"]].copy()
    regions_area["region_area"] = regions_area.geometry.area

    overlay = gpd.overlay(
        regions_area, hucs, how="intersection", keep_geom_type=False,
    )
    if overlay.empty:
        return pd.DataFrame(
            columns=[region_id, huc_id, "overlap_area", "overlap_frac", "is_dominant"]
        )

    overlay["overlap_area"] = overlay.geometry.area
    overlay = overlay.merge(
        regions_area[[region_id, "region_area"]],
        on=region_id, how="left", suffixes=("", "_r"),
    )
    if "region_area_r" in overlay.columns:
        overlay["region_area"] = overlay["region_area"].fillna(overlay["region_area_r"])

    overlay["overlap_frac"] = (
        overlay["overlap_area"] / overlay["region_area"].replace(0, np.nan)
    )
    overlay = overlay[overlay["overlap_frac"] >= min_overlap_frac].copy()

    if overlay.empty:
        return pd.DataFrame(
            columns=[region_id, huc_id, "overlap_area", "overlap_frac", "is_dominant"]
        )

    overlay["rank"] = overlay.groupby(region_id)["overlap_frac"].rank(
        method="first", ascending=False,
    )
    overlay["is_dominant"] = overlay["rank"] == 1

    return overlay[
        [region_id, huc_id, "overlap_area", "overlap_frac", "is_dominant"]
    ].reset_index(drop=True)


def build_shared_huc_edges(
    membership: pd.DataFrame,
    region_id: str,
    huc_id: str,
) -> pd.DataFrame:
    """Connect regions that share HUC membership."""
    edge_weights: dict[tuple[str, str, str], float] = {}
    if membership.empty:
        return _edge_frame(edge_weights)

    for huc, group in membership.groupby(huc_id):
        regs = group[[region_id, "overlap_frac"]].drop_duplicates()
        if len(regs) < 2:
            continue
        frac = dict(zip(
            regs[region_id].astype(str), regs["overlap_frac"].astype(float),
        ))
        for a, b in combinations(sorted(frac), 2):
            src, dst = _canonical_edge(a, b)
            weight = min(frac[a], frac[b])
            key = (src, dst, "shared_huc")
            edge_weights[key] = max(edge_weights.get(key, 0.0), float(weight))

    return _edge_frame(edge_weights)


def build_drainage_edges(
    membership: pd.DataFrame,
    drainage_edges_path: Path,
    region_id: str,
    huc_id: str,
    from_huc_col: str,
    to_huc_col: str,
    weight_col: str,
) -> pd.DataFrame:
    """Connect regions whose dominant HUCs have a drainage relationship."""
    edge_weights: dict[tuple[str, str, str], float] = {}
    if membership.empty:
        return _edge_frame(edge_weights)

    drainage = pd.read_csv(drainage_edges_path)
    missing = [c for c in [from_huc_col, to_huc_col] if c not in drainage.columns]
    if missing:
        raise ValueError(f"drainage-edges missing columns: {missing}")
    if weight_col not in drainage.columns:
        drainage[weight_col] = 1.0

    dominant = membership[membership["is_dominant"]].copy()
    huc_to_regions = (
        dominant.groupby(huc_id)[region_id]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )

    for row in drainage.itertuples(index=False):
        h1 = str(getattr(row, from_huc_col))
        h2 = str(getattr(row, to_huc_col))
        weight = float(getattr(row, weight_col))
        for a in huc_to_regions.get(h1, []):
            for b in huc_to_regions.get(h2, []):
                if a == b:
                    continue
                src, dst = _canonical_edge(a, b)
                key = (src, dst, "drainage")
                edge_weights[key] = max(edge_weights.get(key, 0.0), weight)

    return _edge_frame(edge_weights)


def combine_edges(*frames: pd.DataFrame) -> pd.DataFrame:
    """Union edge frames, normalizing weights within each relation."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["src", "dst", "weight", "relation"])

    pieces = []
    for g_name, g in pd.concat(frames, ignore_index=True).groupby("relation"):
        g = g.copy()
        max_w = g["weight"].max()
        if max_w > 0:
            g["weight"] = g["weight"] / max_w
        pieces.append(g)

    return (
        pd.concat(pieces, ignore_index=True)
        .groupby(["src", "dst", "relation"], as_index=False)["weight"]
        .max()
        .sort_values(["relation", "src", "dst"])
        .reset_index(drop=True)
    )


# =========================================================================
# Main
# =========================================================================

def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    regions = read_geodata(args.regions)
    regions = normalize_region_ids(regions, args.region_id)

    if args.simplify_tolerance > 0:
        regions["geometry"] = regions.geometry.simplify(
            args.simplify_tolerance, preserve_topology=True,
        )

    project_crs = choose_projected_crs(regions, args.project_crs)
    regions_proj = regions.to_crs(project_crs)

    coords = make_region_coords(regions_proj, args.region_id)
    coords.to_csv(args.out_dir / "region_coords.csv", index=False)

    shared_border = pd.DataFrame()
    shared_huc = pd.DataFrame()
    drainage = pd.DataFrame()
    centroid_dist = pd.DataFrame()
    membership = pd.DataFrame()

    # ── shared_border ──
    if args.topology in {"shared_border", "combined"}:
        shared_border = build_shared_border_edges(regions_proj, args.region_id)
        shared_border.to_csv(args.out_dir / "edges_shared_border.csv", index=False)

    # ── centroid_distance ──
    if args.topology == "centroid_distance":
        centroid_dist = build_centroid_distance_edges(
            coords, args.region_id,
            k=args.centroid_k, sigma=args.centroid_sigma,
        )
        centroid_dist.to_csv(args.out_dir / "edges_centroid_distance.csv", index=False)

    # ── HUC membership ──
    hucs_proj = None
    if args.huc_polygons:
        hucs = read_geodata(args.huc_polygons)
        if hucs.crs is None:
            raise ValueError("HUC polygons CRS is missing.")
        hucs_proj = hucs.to_crs(project_crs)
        membership = compute_region_huc_membership(
            regions_proj=regions_proj,
            hucs_proj=hucs_proj,
            region_id=args.region_id,
            huc_id=args.huc_id,
            min_overlap_frac=args.min_overlap_frac,
        )
        membership.to_csv(args.out_dir / "region_huc_membership.csv", index=False)

    # ── shared_huc ──
    if args.topology in {"shared_huc", "combined"}:
        if membership.empty:
            raise ValueError(
                "--topology shared_huc/combined requires "
                "--huc-polygons with non-empty overlap."
            )
        shared_huc = build_shared_huc_edges(membership, args.region_id, args.huc_id)
        shared_huc.to_csv(args.out_dir / "edges_shared_huc.csv", index=False)

    # ── drainage ──
    if args.topology in {"drainage", "combined"}:
        if args.drainage_edges is None:
            if args.topology == "drainage":
                raise ValueError("--topology drainage requires --drainage-edges.")
        else:
            if membership.empty:
                raise ValueError(
                    "--drainage-edges requires --huc-polygons to map HUCs to regions."
                )
            drainage = build_drainage_edges(
                membership=membership,
                drainage_edges_path=args.drainage_edges,
                region_id=args.region_id,
                huc_id=args.huc_id,
                from_huc_col=args.from_huc_col,
                to_huc_col=args.to_huc_col,
                weight_col=args.weight_col,
            )
            drainage.to_csv(args.out_dir / "edges_drainage.csv", index=False)

    # ── Assemble final edge set ──
    if args.topology == "shared_border":
        final_edges = shared_border
    elif args.topology == "shared_huc":
        final_edges = shared_huc
    elif args.topology == "drainage":
        final_edges = drainage
    elif args.topology == "centroid_distance":
        final_edges = centroid_dist
    else:
        final_edges = combine_edges(shared_border, shared_huc, drainage)

    final_edges.to_csv(args.out_dir / "reference_edges.csv", index=False)

    metadata = TopologyMetadata(
        topology=args.topology,
        n_regions=int(len(regions_proj)),
        n_edges=int(len(final_edges)),
        region_id=args.region_id,
        huc_id=args.huc_id if args.huc_polygons else None,
        regions_path=str(args.regions),
        huc_polygons_path=str(args.huc_polygons) if args.huc_polygons else None,
        drainage_edges_path=str(args.drainage_edges) if args.drainage_edges else None,
        crs=project_crs,
        notes=(
            "Use reference_edges.csv for kappa_reconstruct reference topology. "
            "For FloodRSCT, combined/shared_huc/drainage are preferred over "
            "centroid adjacency when hydrological structure is the generator. "
            "centroid_distance is the geometry-only null for convergent validity."
        ),
    )

    with open(args.out_dir / "topology_metadata.json", "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)

    print(json.dumps(asdict(metadata), indent=2))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
