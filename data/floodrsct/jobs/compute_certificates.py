#!/usr/bin/env python3
"""
compute_certificates.py -- Phase 4.5: RSCT certificates per experiment cell.

Thin CLI wrapper around rsct.experiment_cert.certify_experiment_cell().
Loads model results + diagnostics from S3, certifies each cell, uploads
certificate parquet + JSON.

Usage:
    python compute_certificates.py --level r0 --upload
    python compute_certificates.py --level r1 --upload
    python compute_certificates.py --level r2 --upload
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client, level_prefix
from _s3_result import upload_json_result

# rsct service layer -- the actual certification logic
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rsct.experiment_cert import certify_experiment_cell

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]

PRIMARY_METRIC = {
    "regression": "r2",
    "classification": "roc_auc",
}


# ---------------------------------------------------------------------------
# Metric extraction from results JSON
# ---------------------------------------------------------------------------

def _extract_fold_metrics(
    results: dict, target: str, split: str, solver: str,
) -> list[float]:
    """Extract per-fold primary metric values from a results JSON."""
    runs = results.get("runs", [])
    task_type = None
    for r in runs:
        if r["target"] == target:
            task_type = r.get("task")
            break
    if not task_type:
        return []

    metric_name = PRIMARY_METRIC.get(task_type)
    if not metric_name:
        return []

    vals = []
    for r in runs:
        if (r["target"] == target and r["split"] == split
                and r["solver"] == solver):
            v = r["metrics"].get(metric_name)
            if v is not None:
                vals.append(float(v))
    return vals


def _mean_or_none(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(s3, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception:
        return None


def _load_diagnostics(s3, level: str) -> dict:
    """Load Phase 4 diagnostics and index by (scenario, target)."""
    data = _load_json(s3, f"{RESULTS_PREFIX}/diagnostics_{level}.json")
    if not data:
        return {}
    index = {}
    for cell in data.get("cells", []):
        key = (cell["scenario"], cell["target"])
        index[key] = cell
    return index


def _load_geometry_kappa(s3) -> dict:
    """Load Phase 0.5 geometry-only kappa, indexed by (scenario, target).

    Geometry kappa is computed pre-training from spatial graph, feature
    coverage, scale stability, and topology alignment.  It has zero
    dependency on RSN simplex, fold metrics, predictions, or residuals.
    """
    data = _load_json(s3, f"{RESULTS_PREFIX}/geometry_kappa.json")
    if not data:
        log.warning(
            "geometry_kappa.json not found -- run compute_geometry_kappa.py "
            "first.  kappa will default to 0.0 (unknown geometry)."
        )
        return {}
    index = {}
    for cell in data.get("cells", []):
        key = (cell["scenario"], cell["target"])
        index[key] = cell
    log.info("Loaded geometry kappa for %d cells", len(index))
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_certificates(s3, level: str, upload: bool = False) -> dict:
    """Build RSCT certificates for all cells at one representation level."""
    diag_index = _load_diagnostics(s3, level)
    log.info("Loaded %d diagnostic cells for %s", len(diag_index), level)

    geometry_index = _load_geometry_kappa(s3)

    all_certs = []

    for scenario in MODELABLE:
        results = _load_json(s3, f"{RESULTS_PREFIX}/{level_prefix(level)}_{scenario}.json")
        if not results:
            log.warning("No results for %s/%s", level, scenario)
            continue

        runs = results.get("runs", [])
        targets = sorted(set(r["target"] for r in runs))

        for target in targets:
            task_type = None
            for r in runs:
                if r["target"] == target:
                    task_type = r.get("task")
                    break
            if not task_type:
                continue

            spatial_folds = _extract_fold_metrics(
                results, target, "spatial_blocked", "histgbdt",
            )
            random_folds = _extract_fold_metrics(
                results, target, "random", "histgbdt",
            )

            spatial_metric = _mean_or_none(spatial_folds)
            random_metric = _mean_or_none(random_folds)

            diag_cell = diag_index.get((scenario, target), {})

            # Kappa is geometric compatibility (D*/D), NOT a model-quality
            # signal.  It must have zero computational dependency on RSN,
            # fold metrics, predictions, residuals, or model scores.
            # The 4 diagnostic proxies (diag_leakage, diag_transfer,
            # diag_solver, diag_residual_spatial) are model-derived and
            # remain as diagnostic fields -- they are NOT kappa inputs.
            geom_cell = geometry_index.get((scenario, target), {})
            kappa_geom = geom_cell.get("kappa_geom")

            # Delegate to rsct service layer
            cert = certify_experiment_cell(
                spatial_metric=spatial_metric,
                random_metric=random_metric,
                task_type=task_type,
                fold_metrics=spatial_folds,
                kappa_geom=kappa_geom,
            )
            cert_dict = cert.to_dict()
            cert_dict["scenario"] = scenario
            cert_dict["target"] = target
            cert_dict["level"] = level
            cert_dict["task_type"] = task_type
            cert_dict["spatial_metric"] = spatial_metric
            cert_dict["random_metric"] = random_metric
            cert_dict["n_folds"] = len(spatial_folds)

            all_certs.append(cert_dict)
            log.info(
                "  %s / %s: R=%.3f S=%.3f N=%.3f alpha=%.3f omega=%.3f kappa=%.3f tau=%.3f",
                scenario, target,
                cert.R, cert.S_sup, cert.N,
                cert.alpha, cert.omega, cert.kappa, cert.tau,
            )

    # Summary statistics + cell-level bootstrap CIs
    cert_summary = {}
    if all_certs:
        n_boot = 10_000
        boot_seed = 42
        rng = np.random.default_rng(boot_seed)
        n_cells = len(all_certs)

        for signal in ("R", "S_sup", "N", "alpha", "omega", "kappa", "tau", "sigma"):
            vals = np.array([c.get(signal, float("nan")) for c in all_certs], dtype=float)
            valid = vals[~np.isnan(vals)]
            if len(valid) < 2:
                continue
            obs_mean = float(np.mean(valid))
            boot_means = np.empty(n_boot)
            for i in range(n_boot):
                idx = rng.choice(len(valid), size=len(valid), replace=True)
                boot_means[i] = np.mean(valid[idx])
            cert_summary[signal] = {
                "mean": obs_mean,
                "ci_lower_95": float(np.percentile(boot_means, 2.5)),
                "ci_upper_95": float(np.percentile(boot_means, 97.5)),
                "n_cells": len(valid),
            }

        log.info(
            "Certificate summary: n=%d, R=[%.3f, %.3f], alpha=[%.3f, %.3f], omega=[%.3f, %.3f]",
            n_cells,
            min(c["R"] for c in all_certs), max(c["R"] for c in all_certs),
            min(c["alpha"] for c in all_certs), max(c["alpha"] for c in all_certs),
            min(c["omega"] for c in all_certs), max(c["omega"] for c in all_certs),
        )

    payload = {
        "experiment": "s035-model-ladder",
        "phase": f"certificates_{level}",
        "level": level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(all_certs),
        "certificates": all_certs,
        "summary_bootstrap_ci": {
            "description": (
                "Cell-level bootstrap 95% CIs over (scenario x target) cells. "
                "Uncertainty interval on reported aggregate certificate signals."
            ),
            "n_bootstrap": 10_000,
            "seed": 42,
            "signals": cert_summary,
        },
        "methodology": {
            "service": "rsct.experiment_cert.certify_experiment_cell",
            "R_derivation": "clip(spatial_blocked_metric, 0, 1) for regression; "
                            "2*(AUC-0.5) for classification",
            "S_sup_derivation": "(random_metric - spatial_metric) / |random_metric|, "
                                "clipped to [0, 1-R]",
            "N_derivation": "1 - R - S_sup (simplex closure)",
            "alpha": "R / (R + N) via yrsn (or fallback)",
            "omega": "1 - S_sup via yrsn (or fallback)",
            "kappa": "diag_leakage from Phase 4 diagnostics",
            "tau": "1 / (1 + CV) from per-fold variance",
            "sigma": "std(fold_metrics, ddof=1)",
            "diagnosis": "DegradationDiagnoser 3x3 grid (if yrsn available)",
            "reference_split": "spatial_blocked",
            "reference_solver": "histgbdt",
        },
    }

    output_json = json.dumps(payload, indent=2, default=str)

    if upload:
        json_key = f"{RESULTS_PREFIX}/certificates_{level}.json"
        upload_json_result(s3, BUCKET, json_key, payload)

        if all_certs:
            df = pd.DataFrame(all_certs)
            buf = io.BytesIO()
            df.to_parquet(buf, compression="zstd", index=False)
            pq_key = f"{RESULTS_PREFIX}/certificates_{level}.parquet"
            s3.put_object(Bucket=BUCKET, Key=pq_key, Body=buf.getvalue())
            log.info("Uploaded s3://%s/%s (%d rows)", BUCKET, pq_key, len(df))
    else:
        local = f"/tmp/certificates_{level}.json"
        Path(local).write_text(output_json)
        log.info("Wrote %s", local)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  S035 PHASE 4.5: RSCT CERTIFICATES -- {level.upper()}")
    print(f"{'='*60}\n")
    for c in all_certs:
        print(
            f"  {c['scenario']:25s} {c['target']:30s} "
            f"R={c['R']:.3f} S={c['S_sup']:.3f} N={c['N']:.3f} "
            f"a={c['alpha']:.2f} w={c['omega']:.2f} k={c['kappa']:.2f} "
            f"t={c['tau']:.2f} [{c.get('diagnosis_label', 'N/A')}]"
        )

    print(output_json)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4.5: RSCT certificates per experiment cell"
    )
    parser.add_argument("--level", required=True, choices=["r0", "r1", "r2"])
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()
    run_certificates(s3, args.level, args.upload)


if __name__ == "__main__":
    main()
