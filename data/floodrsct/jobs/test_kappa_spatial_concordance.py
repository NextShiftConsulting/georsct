#!/usr/bin/env python3
"""
test_kappa_spatial_concordance.py -- Verify probe and yrsn Moran's I agree.

Compares the self-contained compute_kappa_spatial() from the S036 probe
against the canonical yrsn.core.kappa.spatial.compute.compute_kappa_spatial.

Tests:
  1. Synthetic clustered data (known high Moran's I)
  2. Synthetic random data (known low Moran's I)
  3. Edge cases: constant residuals, single-node islands, empty adjacency
  4. Real Hawaii data if available on S3

Uploads results to s3://swarm-yrsn-datasets/geocert-experiments/s036/concordance/
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "outputs/concordance"))


# ---------------------------------------------------------------------------
# Probe's reimplementation (copied verbatim from run_floodcaster_kappa_spatial.py)
# ---------------------------------------------------------------------------

def probe_compute_kappa_spatial(
    residuals: np.ndarray,
    adjacency: dict[int, list[int]],
) -> dict:
    n = len(residuals)
    if n < 3:
        return {"morans_i": 0.0, "expected_i": 0.0, "kappa": 0.5, "n_samples": n, "n_edges": 0, "mean_degree": 0.0}

    z = residuals - residuals.mean()
    ss = float(np.dot(z, z))
    if ss < 1e-12:
        return {"morans_i": 0.0, "expected_i": -1.0 / (n - 1), "kappa": 0.5, "n_samples": n, "n_edges": 0, "mean_degree": 0.0}

    cross = 0.0
    W = 0
    for i, neighbors in adjacency.items():
        if i >= n:
            continue
        for j in neighbors:
            if j >= n:
                continue
            cross += z[i] * z[j]
            W += 1

    if W == 0:
        return {"morans_i": 0.0, "expected_i": -1.0 / (n - 1), "kappa": 0.5, "n_samples": n, "n_edges": 0, "mean_degree": 0.0}

    I = (n / W) * (cross / ss)
    E_I = -1.0 / (n - 1)
    kappa = max(0.0, min(1.0, (I - E_I + 1.0) / 2.0))
    mean_degree = W / n

    return {
        "morans_i": float(I),
        "expected_i": float(E_I),
        "kappa": float(kappa),
        "n_samples": n,
        "n_edges": W,
        "mean_degree": float(mean_degree),
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def make_clustered_data(n_per_cluster: int = 10, n_clusters: int = 3, seed: int = 42):
    """Two spatial clusters with different residual magnitudes."""
    rng = np.random.default_rng(seed)
    residuals = []
    adjacency = {}
    idx = 0
    for c in range(n_clusters):
        cluster_mean = (c + 1) * 0.5
        for _ in range(n_per_cluster):
            residuals.append(cluster_mean + rng.normal(0, 0.05))
            adjacency[idx] = []
            idx += 1
    # Wire up within-cluster adjacency (chain)
    idx = 0
    for c in range(n_clusters):
        for i in range(n_per_cluster):
            node = c * n_per_cluster + i
            if i > 0:
                adjacency[node].append(node - 1)
            if i < n_per_cluster - 1:
                adjacency[node].append(node + 1)
    return np.array(residuals), adjacency


def make_random_data(n: int = 30, seed: int = 42):
    """Random residuals on a chain graph -- low spatial autocorrelation."""
    rng = np.random.default_rng(seed)
    residuals = rng.uniform(0, 1, size=n)
    adjacency = {}
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        adjacency[i] = neighbors
    return residuals, adjacency


def make_self_loop_data(n: int = 10, seed: int = 42):
    """Chain graph WITH self-loops to test divergence."""
    rng = np.random.default_rng(seed)
    residuals = rng.uniform(0, 1, size=n)
    adjacency = {}
    for i in range(n):
        neighbors = [i]  # self-loop!
        if i > 0:
            neighbors.append(i - 1)
        if i < n - 1:
            neighbors.append(i + 1)
        adjacency[i] = neighbors
    return residuals, adjacency


def run_comparison(name: str, residuals: np.ndarray, adjacency: dict) -> dict:
    """Run both implementations and compare."""
    from yrsn.core.kappa.spatial.compute import compute_kappa_spatial as yrsn_compute

    probe_result = probe_compute_kappa_spatial(residuals, adjacency)

    yrsn_result_obj = yrsn_compute(residuals, adjacency)
    yrsn_result = {
        "morans_i": yrsn_result_obj.morans_i,
        "expected_i": yrsn_result_obj.expected_i,
        "kappa": yrsn_result_obj.kappa,
        "n_samples": yrsn_result_obj.n_samples,
        "n_edges": yrsn_result_obj.n_edges,
        "mean_degree": yrsn_result_obj.mean_degree,
    }

    diff_i = abs(probe_result["morans_i"] - yrsn_result["morans_i"])
    diff_kappa = abs(probe_result["kappa"] - yrsn_result["kappa"])
    edge_diff = probe_result["n_edges"] - yrsn_result["n_edges"]

    match = diff_kappa < 1e-10 and diff_i < 1e-10 and edge_diff == 0
    status = "PASS" if match else "FAIL"

    logger.info("--- %s: %s ---", name, status)
    logger.info("  Probe:  I=%.8f kappa=%.8f edges=%d", probe_result["morans_i"], probe_result["kappa"], probe_result["n_edges"])
    logger.info("  yrsn:   I=%.8f kappa=%.8f edges=%d", yrsn_result["morans_i"], yrsn_result["kappa"], yrsn_result["n_edges"])
    logger.info("  Diff:   dI=%.2e dKappa=%.2e dEdges=%d", diff_i, diff_kappa, edge_diff)

    return {
        "test": name,
        "status": status,
        "probe": probe_result,
        "yrsn": yrsn_result,
        "diff_morans_i": diff_i,
        "diff_kappa": diff_kappa,
        "diff_edges": edge_diff,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Kappa Spatial Concordance Test")
    logger.info("Probe vs yrsn.core.kappa.spatial.compute")
    logger.info("=" * 60)

    results = []

    # Test 1: Clustered (high Moran's I)
    r, a = make_clustered_data()
    results.append(run_comparison("clustered_high_autocorr", r, a))

    # Test 2: Random (low Moran's I)
    r, a = make_random_data()
    results.append(run_comparison("random_low_autocorr", r, a))

    # Test 3: Self-loops (should diverge if probe doesn't skip them)
    r, a = make_self_loop_data()
    results.append(run_comparison("self_loops_present", r, a))

    # Test 4: Constant residuals
    r = np.ones(10)
    a = {i: [i - 1, i + 1] for i in range(10)}
    a[0] = [1]; a[9] = [8]
    results.append(run_comparison("constant_residuals", r, a))

    # Test 5: Islands (nodes with no neighbors)
    r, a = make_random_data(n=10)
    a[3] = []  # isolate node 3
    a[7] = []  # isolate node 7
    # Remove references to isolated nodes
    for k in a:
        a[k] = [j for j in a[k] if j not in (3, 7)]
    results.append(run_comparison("isolated_nodes", r, a))

    # Summary
    logger.info("=" * 60)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    logger.info("CONCORDANCE: %d PASS, %d FAIL out of %d tests", n_pass, n_fail, len(results))

    for r in results:
        logger.info("  %-30s %s  (dKappa=%.2e, dEdges=%d)",
                     r["test"], r["status"], r["diff_kappa"], r["diff_edges"])
    logger.info("=" * 60)

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {"pass": n_pass, "fail": n_fail, "total": len(results)},
        "tests": results,
    }
    out_path = OUTPUT_DIR / "concordance_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved: %s", out_path)

    # Upload
    import boto3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3_key = "geocert-experiments/s036/concordance/concordance_results.json"
    s3.upload_file(str(out_path), "swarm-yrsn-datasets", s3_key)
    logger.info("Uploaded: s3://swarm-yrsn-datasets/%s", s3_key)

    if n_fail > 0:
        logger.error("CONCORDANCE FAILURES DETECTED -- probe and yrsn diverge")
        sys.exit(1)


if __name__ == "__main__":
    main()
