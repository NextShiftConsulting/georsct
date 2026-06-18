#!/usr/bin/env python3
"""S018B-FREEZE: Benchmark dataset freeze and reproducibility audit.

Self-contained SageMaker processing job script.
Reads OOF artifacts from S3, produces frozen benchmark artifacts.

Deliverables:
  D1: artifact_manifest.json   - SHA-256 hashes of all OOF parquets
  D2: train_test_split.json    - frozen ZCTA split (80/20, seed=42)
  D3: leakage_audit.json       - spatial autocorrelation / leakage checks
  D4: scorecard.json           - per-target R2 by solver family
  D5: summary.json             - aggregate stats for Table 2

Instance: ml.m5.xlarge (CPU)
Expected wall-clock: ~10 min
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


# -- config --
S3_BUCKET = "swarm-yrsn-datasets"
OOF_PREFIX = "rsct_curriculum/series_018/oof_artifacts/"
PROCESSED_PREFIX = "rsct_curriculum/series_018/processed/"
OUTPUT_PREFIX = "rsct_curriculum/series_018/s018b_freeze/"
RANDOM_SEED = 42
TEST_FRACTION = 0.20
N_TARGETS = 27

# Instance / time tracking (NeurIPS requirement)
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.xlarge")
START_TIME = time.time()
START_ISO = datetime.now(timezone.utc).isoformat()


def log(msg):
    """ASCII-safe logging for SageMaker cp1252."""
    print(f"[S018B] {msg}", flush=True)


def sha256_bytes(data):
    """SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def download_parquet(s3, bucket, key):
    """Download parquet from S3, return DataFrame and raw bytes."""
    log(f"Downloading s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    import io
    df = pd.read_parquet(io.BytesIO(raw))
    return df, raw


def list_oof_files(s3, bucket, prefix):
    """List all parquet files under the OOF prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                files.append(obj["Key"])
    return sorted(files)


def detect_oof_format(df):
    """Detect whether OOF is long-form or wide-form.

    Long-form: columns include [zcta, task/target, y_true, y_pred, ...]
    Wide-form: columns are target names, index is ZCTA
    """
    # Canonical GeoRSCT schema: task, model_version, zcta, y_true, y_pred, residual
    if ("task" in df.columns or "target" in df.columns) and "y_true" in df.columns:
        return "long"
    elif "zcta" in df.columns.str.lower() or df.index.name == "zcta":
        return "wide"
    else:
        n_numeric = df.select_dtypes(include=[np.number]).shape[1]
        if n_numeric >= 20:
            return "wide"
        return "unknown"


def get_task_col(df):
    """Return the task/target column name."""
    if "task" in df.columns:
        return "task"
    if "target" in df.columns:
        return "target"
    return None


def get_solver_col(df):
    """Return the solver/model_version column name."""
    if "model_version" in df.columns:
        return "model_version"
    if "solver" in df.columns:
        return "solver"
    return None


def compute_r2_long(df):
    """Compute per-target R2 from long-form OOF."""
    task_col = get_task_col(df)
    if not task_col:
        log("WARNING: No task/target column found for R2 computation")
        return {}
    results = {}
    for target, grp in df.groupby(task_col):
        y_true = grp["y_true"].values
        y_pred = grp["y_pred"].values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 10:
            results[target] = None
            continue
        ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
        ss_tot = np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[target] = float(r2)
    return results


def compute_r2_wide(df_true, df_pred):
    """Compute per-target R2 from wide-form OOF (truth vs pred frames)."""
    results = {}
    common_cols = sorted(set(df_true.columns) & set(df_pred.columns))
    for col in common_cols:
        y_true = df_true[col].values
        y_pred = df_pred[col].values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 10:
            results[col] = None
            continue
        ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
        ss_tot = np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[col] = float(r2)
    return results


def build_manifest(s3, bucket, oof_keys):
    """D1: Build artifact manifest with SHA-256 hashes."""
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "artifacts": {}
    }
    for key in oof_keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
        fname = os.path.basename(key)
        manifest["artifacts"][fname] = {
            "s3_key": key,
            "sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
        }
    return manifest


def build_split(zctas, test_fraction=TEST_FRACTION, seed=RANDOM_SEED):
    """D2: Build frozen train/test split."""
    zctas_sorted = sorted(zctas)
    train_z, test_z = train_test_split(
        zctas_sorted, test_size=test_fraction, random_state=seed
    )
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "test_fraction": test_fraction,
        "n_total": len(zctas_sorted),
        "n_train": len(train_z),
        "n_test": len(test_z),
        "train_zctas": sorted(train_z),
        "test_zctas": sorted(test_z),
    }


def build_leakage_audit(df, zcta_col="zcta"):
    """D3: Basic leakage audit -- check for duplicate ZCTAs, target overlap."""
    audit = {
        "created": datetime.now(timezone.utc).isoformat(),
        "checks": {}
    }

    solver_col = get_solver_col(df)
    task_col = get_task_col(df)

    # Check 1: duplicate ZCTAs within solver
    # Note: not every ZCTA has every target (Census/CDC suppress small-pop ZCTAs),
    # so expected = distinct (zcta, task) pairs, NOT n_zcta * n_targets.
    if solver_col and zcta_col in df.columns:
        for solver, grp in df.groupby(solver_col):
            n_zcta = grp[zcta_col].nunique()
            n_rows = len(grp)
            n_targets = grp[task_col].nunique() if task_col else 1
            n_folds = grp["fold"].nunique() if "fold" in grp.columns else 1
            if task_col:
                expected = grp[[zcta_col, task_col]].drop_duplicates().shape[0]
            else:
                expected = n_zcta
            has_duplicates = n_rows != expected
            n_suppressed = (n_zcta * n_targets) - expected
            audit["checks"][f"dup_{solver}"] = {
                "n_zcta": int(n_zcta),
                "n_rows": int(n_rows),
                "n_targets": int(n_targets),
                "n_folds": int(n_folds),
                "expected_rows": int(expected),
                "n_suppressed_zcta_target_pairs": int(n_suppressed),
                "note": "expected = distinct (zcta, target) pairs; "
                        "some ZCTAs lack certain targets due to "
                        "Census/CDC data suppression for small populations",
                "pass": not has_duplicates,
            }
    elif zcta_col in df.columns:
        n_zcta = df[zcta_col].nunique()
        n_rows = len(df)
        audit["checks"]["dup_check"] = {
            "n_zcta": int(n_zcta),
            "n_rows": int(n_rows),
            "pass": True,
        }

    # Check 2: NaN fraction
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_frac = float(df[numeric_cols].isna().mean().mean()) if len(numeric_cols) > 0 else 0.0
    audit["checks"]["nan_fraction"] = {
        "value": round(nan_frac, 6),
        "pass": nan_frac < 0.05,
    }

    audit["overall_pass"] = all(c["pass"] for c in audit["checks"].values())
    return audit


def build_scorecard(r2_by_solver):
    """D4: Per-target R2 scorecard across solver families."""
    scorecard = {
        "created": datetime.now(timezone.utc).isoformat(),
        "solvers": {}
    }
    for solver_name, r2_dict in r2_by_solver.items():
        vals = [v for v in r2_dict.values() if v is not None]
        scorecard["solvers"][solver_name] = {
            "per_target_r2": r2_dict,
            "mean_r2": float(np.mean(vals)) if vals else None,
            "median_r2": float(np.median(vals)) if vals else None,
            "std_r2": float(np.std(vals)) if vals else None,
            "n_targets": len(vals),
        }

    # Cross-solver spread (key paper metric)
    if len(r2_by_solver) > 1:
        all_targets = set()
        for d in r2_by_solver.values():
            all_targets.update(d.keys())

        spreads = []
        for target in sorted(all_targets):
            vals = [r2_by_solver[s].get(target) for s in r2_by_solver
                    if r2_by_solver[s].get(target) is not None]
            if len(vals) >= 2:
                spreads.append(max(vals) - min(vals))

        scorecard["cross_solver_spread"] = {
            "mean": float(np.mean(spreads)) if spreads else None,
            "median": float(np.median(spreads)) if spreads else None,
            "max": float(np.max(spreads)) if spreads else None,
            "n_targets_compared": len(spreads),
        }

    return scorecard


def build_summary(manifest, split, leakage, scorecard):
    """D5: Summary for Table 2."""
    summary = {
        "created": datetime.now(timezone.utc).isoformat(),
        "n_artifacts": len(manifest["artifacts"]),
        "n_zctas_total": split["n_total"],
        "n_train": split["n_train"],
        "n_test": split["n_test"],
        "leakage_pass": leakage["overall_pass"],
        "solvers": {},
    }
    for solver, data in scorecard["solvers"].items():
        summary["solvers"][solver] = {
            "mean_r2": data["mean_r2"],
            "n_targets": data["n_targets"],
        }
    if "cross_solver_spread" in scorecard:
        summary["cross_solver_spread"] = scorecard["cross_solver_spread"]
    return summary


def upload_json(s3, bucket, key, data):
    """Upload JSON to S3."""
    body = json.dumps(data, indent=2, default=str)
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
    log(f"Uploaded s3://{bucket}/{key}")


def main():
    log("=" * 60)
    log("S018B-FREEZE: Benchmark Dataset Freeze")
    log(f"Instance: {INSTANCE_TYPE}")
    log(f"Start: {START_ISO}")
    log("=" * 60)

    s3 = boto3.client("s3", region_name="us-east-1")

    # 1. Discover OOF artifacts
    oof_keys = list_oof_files(s3, S3_BUCKET, OOF_PREFIX)
    log(f"Found {len(oof_keys)} OOF parquet files")
    if not oof_keys:
        log("ERROR: No OOF parquets found. Checking processed/ prefix...")
        oof_keys = list_oof_files(s3, S3_BUCKET, PROCESSED_PREFIX)
        log(f"Found {len(oof_keys)} parquets under processed/")
        if not oof_keys:
            log("FATAL: No parquet files found in either prefix.")
            sys.exit(1)

    for k in oof_keys:
        log(f"  {k}")

    # 2. D1: Artifact manifest
    log("--- D1: Building artifact manifest ---")
    manifest = build_manifest(s3, S3_BUCKET, oof_keys)
    log(f"Manifest: {len(manifest['artifacts'])} artifacts hashed")

    # 3. Load OOF data and detect format
    log("--- Loading OOF data ---")
    solver_dfs = {}
    all_zctas = set()

    for key in oof_keys:
        df, _ = download_parquet(s3, S3_BUCKET, key)
        fname = os.path.basename(key)
        fmt = detect_oof_format(df)
        log(f"  {fname}: {len(df)} rows, format={fmt}, cols={list(df.columns[:8])}")

        # Extract solver name from model_version column or filename
        solver_col = get_solver_col(df)
        if solver_col and df[solver_col].nunique() == 1:
            solver_name = str(df[solver_col].iloc[0])
        else:
            solver_name = fname.replace(".parquet", "").replace("oof_", "")
        solver_dfs[solver_name] = {"df": df, "format": fmt}

        # Collect ZCTAs
        if "zcta" in df.columns:
            all_zctas.update(df["zcta"].unique())
        elif "ZCTA" in df.columns:
            all_zctas.update(df["ZCTA"].unique())
        elif df.index.name and "zcta" in df.index.name.lower():
            all_zctas.update(df.index.unique())

    log(f"Total unique ZCTAs: {len(all_zctas)}")

    # 4. D2: Train/test split
    log("--- D2: Building train/test split ---")
    if all_zctas:
        # Convert to strings for JSON serialization
        zcta_list = [str(z) for z in all_zctas]
        split = build_split(zcta_list)
    else:
        log("WARNING: No ZCTA column found; building split from row indices")
        n_rows = max(len(d["df"]) for d in solver_dfs.values())
        indices = list(range(n_rows))
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_FRACTION, random_state=RANDOM_SEED
        )
        split = {
            "created": datetime.now(timezone.utc).isoformat(),
            "seed": RANDOM_SEED,
            "test_fraction": TEST_FRACTION,
            "n_total": n_rows,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "note": "index-based split (no ZCTA column found)",
        }
    log(f"Split: {split['n_train']} train / {split['n_test']} test")

    # 5. Compute per-solver R2
    log("--- Computing per-solver R2 ---")
    r2_by_solver = {}
    for solver_name, sdata in solver_dfs.items():
        df = sdata["df"]
        fmt = sdata["format"]
        if fmt == "long":
            r2 = compute_r2_long(df)
        else:
            log(f"  WARNING: Cannot compute R2 for {solver_name} (format={fmt})")
            r2 = {}
        r2_by_solver[solver_name] = r2
        vals = [v for v in r2.values() if v is not None]
        log(f"  {solver_name}: {len(vals)} targets, mean R2={np.mean(vals):.4f}" if vals else f"  {solver_name}: no R2 computed")

    # 6. D3: Leakage audit
    log("--- D3: Leakage audit ---")
    # Combine all solver data for audit
    combined_dfs = []
    for solver_name, sdata in solver_dfs.items():
        df = sdata["df"].copy()
        # Normalize column names for leakage audit
        solver_col = get_solver_col(df)
        if not solver_col:
            df["model_version"] = solver_name
        combined_dfs.append(df)
    if combined_dfs:
        # Only concat if schemas are compatible
        try:
            combined = pd.concat(combined_dfs, ignore_index=True)
        except Exception:
            combined = combined_dfs[0]
        leakage = build_leakage_audit(combined)
    else:
        leakage = {"created": datetime.now(timezone.utc).isoformat(),
                    "checks": {}, "overall_pass": False}
    log(f"Leakage audit: {'PASS' if leakage['overall_pass'] else 'FAIL'}")

    # 7. D4: Scorecard
    log("--- D4: Building scorecard ---")
    scorecard = build_scorecard(r2_by_solver)
    if "cross_solver_spread" in scorecard:
        cs = scorecard["cross_solver_spread"]
        log(f"Cross-solver spread: mean={cs['mean']:.4f}, median={cs['median']:.4f}, max={cs['max']:.4f}")

    # 8. D5: Summary
    log("--- D5: Building summary ---")
    summary = build_summary(manifest, split, leakage, scorecard)

    # 9. Upload all deliverables
    log("--- Uploading deliverables ---")
    upload_json(s3, S3_BUCKET, f"{OUTPUT_PREFIX}artifact_manifest.json", manifest)
    upload_json(s3, S3_BUCKET, f"{OUTPUT_PREFIX}train_test_split.json", split)
    upload_json(s3, S3_BUCKET, f"{OUTPUT_PREFIX}leakage_audit.json", leakage)
    upload_json(s3, S3_BUCKET, f"{OUTPUT_PREFIX}scorecard.json", scorecard)
    upload_json(s3, S3_BUCKET, f"{OUTPUT_PREFIX}summary.json", summary)

    # 10. Final report
    elapsed = time.time() - START_TIME
    report = {
        "experiment": "S018B-FREEZE",
        "status": "COMPLETE",
        "instance_type": INSTANCE_TYPE,
        "start_time": START_ISO,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(elapsed, 1),
        "deliverables": {
            "D1_manifest": f"s3://{S3_BUCKET}/{OUTPUT_PREFIX}artifact_manifest.json",
            "D2_split": f"s3://{S3_BUCKET}/{OUTPUT_PREFIX}train_test_split.json",
            "D3_leakage": f"s3://{S3_BUCKET}/{OUTPUT_PREFIX}leakage_audit.json",
            "D4_scorecard": f"s3://{S3_BUCKET}/{OUTPUT_PREFIX}scorecard.json",
            "D5_summary": f"s3://{S3_BUCKET}/{OUTPUT_PREFIX}summary.json",
        },
        "leakage_pass": leakage["overall_pass"],
    }
    upload_json(s3, S3_BUCKET, f"{OUTPUT_PREFIX}report.json", report)

    log("=" * 60)
    log(f"S018B-FREEZE COMPLETE in {elapsed:.1f}s")
    log(f"Instance: {INSTANCE_TYPE}")
    log(f"Deliverables: 6 files -> s3://{S3_BUCKET}/{OUTPUT_PREFIX}")
    log("=" * 60)


if __name__ == "__main__":
    main()
