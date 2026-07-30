"""Composition root: five-construct divergence for one scenario.

Wires the concrete data-source and model-fitter adapters into the
``compute_five_construct_divergence`` use case and serializes the resulting
5x5 divergence matrix (plus a data-quality block) to JSON. This is the only
layer that performs I/O and adapter construction; the use case and domain
stay pure (P4).

Cross-cutting geo-data utilities are CALLED from the central ``rsct_spatial``
package -- spatial weights (``weights.builders``), centroid recovery
(``geometry``), and the dedup/recover/require-finite/coverage process
(``quality``). This job orchestrates them; it does not reimplement them.

The shared inputs (feature matrix, folds, coordinates, spatial weights) are
built ONCE in the canonical id order owned by the data source, so every
per-construct target the use case reads back is row-aligned with them.

Usage:
    python -m georsct.jobs.compute_construct_divergence \
        --scenario houston \
        --table processed/houston/houston_scenario.parquet \
        --centroid-table data/geocert/v24/zcta_features_labels.parquet \
        --out results/houston_divergence.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rsct_spatial.geometry import ParquetCentroidSource
from rsct_spatial.quality import coverage_report
from rsct_spatial.weights.builders import build_row_normalized_knn

from georsct.adapters.parquet_construct_data_source import ParquetConstructDataSource
from georsct.adapters.sklearn_model_fitter import SklearnModelFitter
from georsct.application.use_cases.certify_constructs import (
    compute_five_construct_divergence,
)
from georsct.domain.construct_certificate import CONSTRUCT_TARGET_COLUMNS
from georsct.domain.construct_divergence_matrix import (
    DivergenceMatrix,
    summarize_divergence,
)
from georsct.loaders.georsct import feature_columns

log = logging.getLogger(__name__)

_CONSTRUCT_TARGET_COLS = set(CONSTRUCT_TARGET_COLUMNS.values())


@dataclass(frozen=True)
class DivergenceRun:
    """The two artifacts of a run: the matrix and its data-quality block."""

    matrix: DivergenceMatrix
    data_quality: dict


def select_feature_columns(df) -> list[str]:
    """Prefix-selected features minus construct targets (anti-leakage).

    ``feature_columns`` includes the ``flood_`` prefix, which also matches
    the FEMA construct target ``flood_pct_zone_a``; a target must never
    enter its own construct's feature matrix, so all construct target
    columns are removed here regardless of prefix.
    """
    return [c for c in feature_columns(df) if c not in _CONSTRUCT_TARGET_COLS]


def run(
    scenario: str,
    table: str | Path,
    centroid_table: str | Path | None = None,
    id_column: str = "zcta_id",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    n_folds: int = 5,
    knn_k: int = 8,
    n_baseline_trials: int = 20,
    n_mantel_perms: int = 99,
    min_finite: int = 30,
    event_column: str | None = "event",
    event_id: str | None = None,
) -> DivergenceRun:
    """Compute the divergence matrix + data-quality block for one scenario.

    If the scenario table is replicated per event, pass ``event_id`` (with
    ``event_column``, default ``"event"``) to slice one event so the analysis
    is one row per region. Without it, a replicated table raises rather than
    silently collapsing regions.
    """
    # Authoritative centroid source (central): recover missing coordinates
    # instead of dropping the rows -- a missing centroid is a broken join.
    centroid_source = None
    if centroid_table is not None:
        centroid_source = ParquetCentroidSource(
            centroid_table, id_column=id_column,
            lat_column=lat_column, lon_column=lon_column,
        )

    data_source = ParquetConstructDataSource(
        scenario_paths={scenario: table},
        id_column=id_column,
        min_finite=min_finite,
        centroid_source=centroid_source,
        lat_column=lat_column,
        lon_column=lon_column,
        event_column=event_column,
        event_id=event_id,
        # Coordinates are mandatory for the spatial analysis; only rows that
        # remain coordinate-less after recovery are dropped, once, aligned.
        require_finite=(lat_column, lon_column),
    )
    df = data_source.scenario_frame(scenario)
    region_ids = data_source.canonical_order(scenario)
    region_order = tuple(region_ids)

    feat_cols = select_feature_columns(df)
    if not feat_cols:
        raise ValueError(
            f"scenario '{scenario}' has no usable feature columns "
            f"(checked georsct feature prefixes minus construct targets)"
        )
    features = df[feat_cols].to_numpy(dtype=float)

    # coords2d as (lon, lat) -> planar (x, y) ordering.
    coords2d = df[[lon_column, lat_column]].to_numpy(dtype=float)

    # Frozen, target-independent folds (ADR-014): deterministic assignment
    # over the canonical order, so replays are reproducible.
    n_obs = features.shape[0]
    fold_ids = np.arange(n_obs) % int(max(n_folds, 1))

    W_geo = build_row_normalized_knn(coords2d, knn_k)  # central rsct_spatial
    model_fitter = SklearnModelFitter()

    log.info(
        "scenario=%s n_obs=%d n_features=%d n_folds=%d knn_k=%d",
        scenario, n_obs, len(feat_cols), n_folds, knn_k,
    )

    dm = compute_five_construct_divergence(
        scenario_id=scenario,
        data_source=data_source,
        model_fitter=model_fitter,
        features=features,
        fold_ids=fold_ids,
        region_ids=region_ids,
        region_order=region_order,
        coords2d=coords2d,
        W_geo=W_geo,
        event_id=event_id,
    )

    present_targets = sorted(c for c in _CONSTRUCT_TARGET_COLS if c in df.columns)
    data_quality = {
        "n_obs": int(n_obs),
        "n_features": len(feat_cols),
        "cleanup": data_source.cleanup_report(scenario),
        "construct_coverage": coverage_report(df, present_targets, id_column=id_column),
    }
    return DivergenceRun(matrix=dm, data_quality=data_quality)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the five-construct divergence matrix for a scenario.",
    )
    parser.add_argument("--scenario", required=True, help="Scenario id, e.g. houston")
    parser.add_argument("--table", required=True, help="Path to the scenario parquet")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--centroid-table", default=None,
        help="Authoritative (id, lat, lon) parquet for centroid recovery",
    )
    parser.add_argument("--id-column", default="zcta_id")
    parser.add_argument("--lat-column", default="latitude")
    parser.add_argument("--lon-column", default="longitude")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--n-baseline-trials", type=int, default=20)
    parser.add_argument("--n-mantel-perms", type=int, default=99)
    parser.add_argument("--min-finite", type=int, default=30)
    parser.add_argument("--event-column", default="event",
                        help="Column identifying the event when a table is per-event replicated")
    parser.add_argument("--event-id", default=None,
                        help="Slice the scenario to this event (one row per region)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = run(
        scenario=args.scenario,
        table=args.table,
        centroid_table=args.centroid_table,
        id_column=args.id_column,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
        n_folds=args.n_folds,
        knn_k=args.knn_k,
        n_baseline_trials=args.n_baseline_trials,
        n_mantel_perms=args.n_mantel_perms,
        min_finite=args.min_finite,
        event_column=args.event_column,
        event_id=args.event_id,
    )
    dm = result.matrix

    summary = summarize_divergence(dm)
    summary["data_quality"] = result.data_quality
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ASCII-only stdout (cp1252-safe on Windows/SageMaker).
    rec = result.data_quality["cleanup"].get("coordinate_recovery", {})
    print(f"scenario={dm.geography_id} n_available={dm.n_available}/5")
    print(f"mean_distance={_fmt(dm.mean_distance)} max_distance={_fmt(dm.max_distance)}")
    print(f"max_pair={dm.max_pair[0].value} <-> {dm.max_pair[1].value}")
    if rec:
        print(
            f"coords recovered={rec.get('n_recovered', 0)} "
            f"unrecovered={rec.get('n_unrecovered', 0)} "
            f"unrecovered_ids={rec.get('unrecovered_ids', [])}"
        )
    print(f"wrote {out_path}")
    return 0


def _fmt(v: float) -> str:
    return "nan" if (v != v) else f"{v:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
