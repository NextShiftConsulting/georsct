#!/usr/bin/env python3
"""run_five_construct_divergence.py -- Five-construct divergence harness.

Adapter script implementing the ModelFitter and ConstructDataSource ports.
Certifies the same geography under each of five flood constructs, then
computes pairwise certificate distance to produce a 5x5 divergence matrix.

Hex arch: this is the outbound adapter layer.  Domain math lives in
georsct.domain.construct_certificate and georsct.domain.construct_divergence_matrix.
Orchestration lives in georsct.application.use_cases.certify_constructs.
This file handles S3 I/O, pandas wrangling, sklearn model fitting, and CLI.

Usage:
    # S3 mode (recommended)
    python run_five_construct_divergence.py --scenario houston --upload

    # Dry run (no S3 access, prints plan)
    python run_five_construct_divergence.py --scenario houston --dry-run

    # With event filter (NFIP/FAST become event-specific)
    python run_five_construct_divergence.py --scenario houston --event harvey2017 --upload
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Wire domain + ports + use case
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from georsct.domain.construct_certificate import (
    CONSTRUCT_TARGET_COLUMNS,
    ConstructLabel,
)
from georsct.domain.construct_divergence_matrix import summarize_divergence
from georsct.application.use_cases.certify_constructs import (
    compute_five_construct_divergence,
)
from georsct.ports.model_fitter import EmbedResult, FitPredictResult, ModelFitter
from georsct.ports.construct_data_source import ConstructData, ConstructDataSource

# S3 infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

# Naming convention: results/s035/doe_c1/{artifact}_{scenario}.json
# Matches PAPER_CONTRACT.yaml artifact paths and pipeline convention
# ({phase}_{scenario}.json for per-scenario, {artifact}.json for aggregate).
RESULTS_PREFIX = "results/s035/doe_c1"

# Metric-level cache: flat parquets for cross-scenario aggregation in QA.
# One row per construct per scenario, columns = certificate axes + metadata.
CACHE_PREFIX = "results/s035/doe_c1/cache"

# FAST S3 key pattern per scenario and return period
FAST_RP_MAP = {
    "houston": "100yr",
    "nyc": "100yr",
    "southwest_florida": "cat4",
}

# Event-to-return-period mapping for FAST data
EVENT_RP_MAP = {
    "harvey2017": "1000yr",
    "imelda2019": "200yr",
    "beryl2024": "25yr",
    "ida2021": "100yr",
    "henri2021": "1yr",
    "ian2022": "cat4",
    "helene2024": "cat4",
    "milton2024": "cat3",
}

# Frozen HistGBDT hyperparameters (ADR-014: no runtime tuning).
# Matches adversarial_reconstruct.py for cross-script consistency.
HGBDT_PARAMS = dict(
    loss="squared_error",
    max_iter=300,
    max_depth=6,
    min_samples_leaf=10,
    learning_rate=0.05,
)


# =========================================================================
# Port implementations
# =========================================================================

class HistGBDTModelFitter(ModelFitter):
    """Concrete ModelFitter using HistGradientBoostingRegressor.

    ADR-014: frozen hyperparameters, frozen folds.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed

    def fit_predict(
        self,
        features: np.ndarray,
        target: np.ndarray,
        fold_ids: np.ndarray,
        task_type: str,
    ) -> FitPredictResult:
        X = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        y = target.astype(float)
        folds = sorted(set(fold_ids))
        pred = np.full(len(y), np.nan, dtype=float)

        for fold in folds:
            train = fold_ids != fold
            test = ~train
            if train.sum() < 20 or test.sum() < 5:
                continue

            model = HistGradientBoostingRegressor(
                random_state=self._seed, **HGBDT_PARAMS,
            )
            model.fit(X[train], y[train])
            pred[test] = model.predict(X[test])

        valid = np.isfinite(pred)
        if valid.sum() < 3:
            return FitPredictResult(
                predictions=pred,
                forward_score=float("nan"),
                task_type=task_type,
            )

        score = float(r2_score(y[valid], pred[valid]))
        return FitPredictResult(
            predictions=pred,
            forward_score=score,
            task_type=task_type,
        )

    def aggregate_embeddings(
        self,
        features: np.ndarray,
        region_ids: np.ndarray,
        region_order: tuple[str, ...],
    ) -> EmbedResult:
        X = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        n_regions = len(region_order)
        n_features = X.shape[1]
        idx_map = {r: i for i, r in enumerate(region_order)}

        sums = np.zeros((n_regions, n_features), dtype=float)
        counts = np.zeros(n_regions, dtype=float)

        for k in range(len(region_ids)):
            r = str(region_ids[k])
            i = idx_map.get(r)
            if i is not None:
                sums[i] += X[k]
                counts[i] += 1

        counts[counts == 0] = 1  # avoid division by zero
        means = sums / counts[:, np.newaxis]

        # Global mean fill for regions with no data
        global_mean = means[counts > 1].mean(axis=0) if (counts > 1).any() else np.zeros(n_features)
        for i in range(n_regions):
            if counts[i] <= 1:
                means[i] = global_mean

        Z = StandardScaler().fit_transform(means) if n_regions > 1 else means
        return EmbedResult(embeddings=Z, region_order=region_order)


class S3ConstructDataSource(ConstructDataSource):
    """Concrete ConstructDataSource loading from S3 parquets.

    JRC, Deltares, FEMA, NFIP: from event_features parquet.
    FAST: from separate fast_zcta_{rp}.parquet (requires merge).
    """

    def __init__(self, s3, event_df: pd.DataFrame, scenario_id: str):
        self._s3 = s3
        self._event_df = event_df
        self._scenario_id = scenario_id

    def load_construct_target(
        self,
        construct: ConstructLabel,
        scenario_id: str,
        event_id: Optional[str] = None,
    ) -> ConstructData:
        target_col = CONSTRUCT_TARGET_COLUMNS[construct]
        df = self._event_df

        if event_id:
            if "event" in df.columns:
                df = df[df["event"] == event_id]
            if df.empty:
                return ConstructData(
                    construct=construct,
                    target_values=None,
                    region_ids=None,
                    available=False,
                    reason=f"no rows for event {event_id}",
                )

        # FAST requires loading a separate parquet
        if construct == ConstructLabel.FAST:
            return self._load_fast_target(df, scenario_id, event_id)

        if target_col not in df.columns:
            return ConstructData(
                construct=construct,
                target_values=None,
                region_ids=None,
                available=False,
                reason=f"column {target_col} not in event_features",
            )

        vals = df[target_col].to_numpy(dtype=float)
        n_finite = int(np.isfinite(vals).sum())
        if n_finite < 10:
            return ConstructData(
                construct=construct,
                target_values=None,
                region_ids=None,
                available=False,
                reason=f"insufficient finite values (n={n_finite})",
            )

        region_ids = df["zcta_id"].astype(str).to_numpy()
        return ConstructData(
            construct=construct,
            target_values=vals,
            region_ids=region_ids,
            available=True,
            reason="ok",
        )

    def _load_fast_target(
        self,
        df: pd.DataFrame,
        scenario_id: str,
        event_id: Optional[str],
    ) -> ConstructData:
        rp = EVENT_RP_MAP.get(event_id, "") if event_id else FAST_RP_MAP.get(scenario_id, "")
        if not rp:
            return ConstructData(
                construct=ConstructLabel.FAST,
                target_values=None,
                region_ids=None,
                available=False,
                reason=f"no FAST RP mapping for scenario={scenario_id}, event={event_id}",
            )

        key = f"processed/{scenario_id}/{scenario_id}_fast_zcta_{rp}.parquet"
        try:
            resp = self._s3.get_object(Bucket=BUCKET, Key=key)
            fast_df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        except Exception as exc:
            return ConstructData(
                construct=ConstructLabel.FAST,
                target_values=None,
                region_ids=None,
                available=False,
                reason=f"FAST parquet not found: {key} ({exc})",
            )

        # Normalize zcta_id
        if fast_df.index.name == "zcta":
            fast_df = fast_df.reset_index().rename(columns={"zcta": "zcta_id"})
        elif "zcta" in fast_df.columns:
            fast_df = fast_df.rename(columns={"zcta": "zcta_id"})
        fast_df["zcta_id"] = fast_df["zcta_id"].astype(str)

        target_col = CONSTRUCT_TARGET_COLUMNS[ConstructLabel.FAST]
        if target_col not in fast_df.columns:
            return ConstructData(
                construct=ConstructLabel.FAST,
                target_values=None,
                region_ids=None,
                available=False,
                reason=f"column {target_col} not in FAST parquet",
            )

        # Merge with event df to align on shared ZCTAs
        merged = pd.merge(
            df[["zcta_id"]].drop_duplicates(),
            fast_df[["zcta_id", target_col]],
            on="zcta_id",
            how="inner",
        )

        if len(merged) < 10:
            return ConstructData(
                construct=ConstructLabel.FAST,
                target_values=None,
                region_ids=None,
                available=False,
                reason=f"insufficient FAST overlap (n={len(merged)})",
            )

        return ConstructData(
            construct=ConstructLabel.FAST,
            target_values=merged[target_col].to_numpy(dtype=float),
            region_ids=merged["zcta_id"].astype(str).to_numpy(),
            available=True,
            reason="ok",
        )

    def available_constructs(
        self,
        scenario_id: str,
    ) -> list[ConstructLabel]:
        available = []
        for construct in ConstructLabel:
            cd = self.load_construct_target(construct, scenario_id)
            if cd.available:
                available.append(construct)
        return available


# =========================================================================
# Data loading helpers
# =========================================================================

def _load_event_features(s3, scenario: str) -> pd.DataFrame:
    """Load scenario event features from S3."""
    from _coverage_common import OUTPUT_KEYS
    key = OUTPUT_KEYS[scenario]
    log.info("Loading s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    df["zcta_id"] = df["zcta_id"].astype(str)
    log.info("Loaded %d rows x %d cols", len(df), len(df.columns))
    return df


def _merge_shared_layers(s3, df: pd.DataFrame) -> pd.DataFrame:
    """Merge JRC and Deltares shared parquets into event_features.

    These are ZCTA-level static layers stored in processed/shared/ that
    are NOT scenario-specific.  The construct harness needs them as columns
    in the event_features dataframe so S3ConstructDataSource can find them.
    """
    shared_layers = {
        "processed/shared/zcta_jrc_water_occurrence_pct.parquet": [
            "jrc_occurrence_mean",
        ],
        "processed/shared/zcta_deltares_depth.parquet": [
            "deltares_depth_ft_rp100",
        ],
    }
    for key, cols in shared_layers.items():
        # Skip if columns already present
        if all(c in df.columns for c in cols):
            log.info("Shared columns %s already in event_features, skipping %s", cols, key)
            continue
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            layer = pd.read_parquet(io.BytesIO(resp["Body"].read()))
            layer["zcta_id"] = layer["zcta_id"].astype(str)
            merge_cols = ["zcta_id"] + [c for c in cols if c in layer.columns]
            if len(merge_cols) > 1:
                df = df.merge(layer[merge_cols], on="zcta_id", how="left")
                log.info("Merged %d cols from %s (%d rows matched)",
                         len(merge_cols) - 1, key, df[cols[0]].notna().sum())
            else:
                log.warning("Expected columns %s not found in %s (has: %s)",
                            cols, key, layer.columns.tolist())
        except Exception as exc:
            log.warning("Could not load shared layer %s: %s", key, exc)
    return df


def _load_folds(s3, scenario: str) -> pd.DataFrame:
    """Load fold assignments from S3."""
    key = f"folds/{scenario}_folds.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception:
        log.warning("Folds not found at %s, will use hash-based folds", key)
        return pd.DataFrame()


def _load_coords(s3) -> pd.DataFrame:
    """Load ZCTA centroids."""
    key = "raw/geocertdb2026/zcta_features_labels.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        df["zcta_id"] = df["zcta_id"].astype(str)
        return df[["zcta_id", "lat", "lon"]]
    except Exception:
        log.warning("Centroids not found at %s", key)
        return pd.DataFrame()


def _load_adjacency(s3) -> Optional[sparse.csr_matrix]:
    """Load ZCTA adjacency as CSR matrix."""
    from _coverage_common import load_adjacency
    try:
        adj_df = load_adjacency(s3)
        return _adjacency_df_to_csr(adj_df)
    except Exception as exc:
        log.warning("Adjacency not loaded: %s", exc)
        return None


def _adjacency_df_to_csr(adj_df: pd.DataFrame) -> sparse.csr_matrix:
    """Convert adjacency edge list to row-normalized CSR."""
    ids = sorted(set(adj_df.iloc[:, 0].astype(str)) | set(adj_df.iloc[:, 1].astype(str)))
    idx = {z: i for i, z in enumerate(ids)}
    n = len(ids)
    rows, cols = [], []
    for _, row in adj_df.iterrows():
        a = idx.get(str(row.iloc[0]))
        b = idx.get(str(row.iloc[1]))
        if a is not None and b is not None and a != b:
            rows.extend([a, b])
            cols.extend([b, a])
    data = np.ones(len(rows), dtype=float)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Row-normalize
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W = W.multiply(1.0 / row_sums[:, np.newaxis]).tocsr()
    return W


def _select_features(df: pd.DataFrame, target: str) -> list[str]:
    """Select numeric feature columns, excluding reserved columns."""
    reserved = {
        target, "zcta_id", "event", "fold", "lat", "lon",
        "obs_nfip_event_claims", "nfip_event_claim_count",
        "nfip_event_total_loss", "obs_has_311", "obs_has_hwm",
    }
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in reserved and not c.startswith("_fs_")
    ]


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Five-construct divergence harness"
    )
    p.add_argument("--scenario", required=True, choices=SCENARIOS,
                   help="Scenario to certify")
    p.add_argument("--event", default=None,
                   help="Optional event filter (e.g., harvey2017)")
    p.add_argument("--upload", action="store_true",
                   help="Upload results to S3")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and exit")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run:
        log.info("DRY RUN: five-construct divergence for %s", args.scenario)
        log.info("Constructs: %s", [c.name for c in ConstructLabel])
        log.info("Targets: %s", dict(CONSTRUCT_TARGET_COLUMNS))
        return 0

    s3 = get_s3_client()

    # Load data
    log.info("Loading data for scenario: %s", args.scenario)
    event_df = _load_event_features(s3, args.scenario)
    event_df = _merge_shared_layers(s3, event_df)
    folds_df = _load_folds(s3, args.scenario)
    coords_df = _load_coords(s3)
    W_geo = _load_adjacency(s3)

    # Merge folds if available
    if not folds_df.empty and "fold" not in event_df.columns:
        folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
        event_df = event_df.merge(folds_df[["zcta_id", "fold"]], on="zcta_id", how="left")

    if "fold" not in event_df.columns:
        log.info("No folds found, creating hash-based folds")
        event_df["fold"] = event_df["zcta_id"].apply(lambda z: hash(z) % 5)

    # Feature selection (use NFIP target as baseline for exclusion)
    feature_cols = _select_features(event_df, "obs_nfip_event_claims")
    log.info("Selected %d features", len(feature_cols))

    features = (
        event_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    fold_ids = event_df["fold"].to_numpy()
    region_ids = event_df["zcta_id"].astype(str).to_numpy()

    # Region ordering and coordinates
    region_order = tuple(sorted(set(region_ids)))

    if not coords_df.empty:
        coords_df = coords_df[coords_df["zcta_id"].isin(region_order)]
        coord_map = {
            str(r.zcta_id): (r.lat, r.lon)
            for _, r in coords_df.iterrows()
        }
        coords2d = np.array([coord_map.get(r, (0.0, 0.0)) for r in region_order])
    else:
        log.warning("No coordinates available, using zeros (kappa_reconstruct will be NaN)")
        coords2d = np.zeros((len(region_order), 2))

    if W_geo is None:
        log.warning("No adjacency matrix, creating identity (kappa_spatial will be NaN)")
        W_geo = sparse.eye(len(region_order), format="csr")

    # Port implementations
    model_fitter = HistGBDTModelFitter(seed=args.seed)
    data_source = S3ConstructDataSource(s3, event_df, args.scenario)

    # Run the use case
    dm = compute_five_construct_divergence(
        scenario_id=args.scenario,
        data_source=data_source,
        model_fitter=model_fitter,
        features=features,
        fold_ids=fold_ids,
        region_ids=region_ids,
        region_order=region_order,
        coords2d=coords2d,
        W_geo=W_geo,
        event_id=args.event,
    )

    # Serialize (P9: replayable audit trail)
    summary = summarize_divergence(dm)
    summary["doe_id"] = "DOE-C1"
    summary["phase"] = "five_construct_divergence"
    summary["scenario"] = args.scenario
    summary["event"] = args.event
    summary["n_features"] = len(feature_cols)
    summary["seed"] = args.seed
    summary["model"] = "HistGradientBoostingRegressor"
    summary["model_params"] = HGBDT_PARAMS

    # Print results
    print()
    print("=" * 72)
    print("Five-Construct Divergence: %s" % args.scenario)
    print("=" * 72)
    for cert in dm.certificates:
        status = "OK" if cert.target_available else "MISSING"
        fwd = "%.3f" % cert.forward_score if cert.target_available else "N/A"
        ks = "%.3f" % cert.kappa_spatial if cert.target_available else "N/A"
        kr = "%.3f" % cert.kappa_reconstruct if cert.target_available else "N/A"
        print("  %-12s  %s  forward=%s  ks=%s  kr=%s" % (
            cert.construct.name, status, fwd, ks, kr,
        ))
    print()
    print("  n_available: %d / 5" % dm.n_available)
    if not np.isnan(dm.mean_distance):
        print("  mean_distance: %.4f" % dm.mean_distance)
        print("  max_distance:  %.4f  (%s vs %s)" % (
            dm.max_distance, dm.max_pair[0].name, dm.max_pair[1].name,
        ))
    print("=" * 72)

    # ---------------------------------------------------------------
    # Output naming convention:
    #   results/s035/doe_c1/five_construct_{scenario}.json      (full result)
    #   results/s035/doe_c1/cache/certificates_{scenario}.parquet (metric cache)
    #   results/s035/doe_c1/cache/pairwise_{scenario}.parquet    (pair cache)
    # ---------------------------------------------------------------
    scenario_tag = args.scenario

    # Write local copy
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results" / "doe_c1"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_file = out_dir / f"five_construct_{scenario_tag}.json"
    with open(local_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Written local: %s", local_file)

    # Build metric-level cache parquets for cross-scenario QA aggregation.
    # certificates cache: one row per construct, columns = certificate axes.
    cert_rows = []
    for c in dm.certificates:
        cert_rows.append({
            "scenario": scenario_tag,
            "construct": c.construct.value,
            "target_available": c.target_available,
            "target_column": c.target_column,
            "forward_score": float(c.forward_score) if np.isfinite(c.forward_score) else None,
            "kappa_spatial": float(c.kappa_spatial) if np.isfinite(c.kappa_spatial) else None,
            "kappa_reconstruct": float(c.kappa_reconstruct) if np.isfinite(c.kappa_reconstruct) else None,
            "morans_i": float(c.morans_i) if np.isfinite(c.morans_i) else None,
            "n_regions": c.n_regions,
            "n_observations": c.n_observations,
            "n_finite_targets": c.n_finite_targets,
        })
    cert_cache_df = pd.DataFrame(cert_rows)

    # pairwise cache: one row per construct pair, columns = distance axes.
    pair_rows = []
    for p in dm.pairwise:
        pair_rows.append({
            "scenario": scenario_tag,
            "construct_a": p.construct_a.value,
            "construct_b": p.construct_b.value,
            "euclidean_distance": float(p.euclidean_distance) if np.isfinite(p.euclidean_distance) else None,
            "forward_delta": float(p.forward_delta) if np.isfinite(p.forward_delta) else None,
            "kappa_spatial_delta": float(p.kappa_spatial_delta) if np.isfinite(p.kappa_spatial_delta) else None,
            "kappa_reconstruct_delta": float(p.kappa_reconstruct_delta) if np.isfinite(p.kappa_reconstruct_delta) else None,
            "both_available": p.both_available,
        })
    pair_cache_df = pd.DataFrame(pair_rows)

    # Write local cache
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cert_cache_df.to_parquet(cache_dir / f"certificates_{scenario_tag}.parquet", index=False)
    pair_cache_df.to_parquet(cache_dir / f"pairwise_{scenario_tag}.parquet", index=False)
    log.info("Written local cache: %s/certificates_%s.parquet, pairwise_%s.parquet",
             cache_dir, scenario_tag, scenario_tag)

    # Upload to S3
    if args.upload:
        # Full result JSON
        json_key = f"{RESULTS_PREFIX}/five_construct_{scenario_tag}.json"
        upload_json_result(s3, BUCKET, json_key, summary)
        log.info("Uploaded s3://%s/%s", BUCKET, json_key)

        # Certificate cache parquet
        cert_buf = io.BytesIO()
        cert_cache_df.to_parquet(cert_buf, index=False)
        cert_buf.seek(0)
        cert_key = f"{CACHE_PREFIX}/certificates_{scenario_tag}.parquet"
        s3.put_object(Bucket=BUCKET, Key=cert_key, Body=cert_buf.getvalue())
        log.info("Uploaded s3://%s/%s", BUCKET, cert_key)

        # Pairwise cache parquet
        pair_buf = io.BytesIO()
        pair_cache_df.to_parquet(pair_buf, index=False)
        pair_buf.seek(0)
        pair_key = f"{CACHE_PREFIX}/pairwise_{scenario_tag}.parquet"
        s3.put_object(Bucket=BUCKET, Key=pair_key, Body=pair_buf.getvalue())
        log.info("Uploaded s3://%s/%s", BUCKET, pair_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
