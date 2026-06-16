"""compute_permutation_null.py -- Permutation test for construct exchangeability.

Reads DOE-C1 per-scenario results from S3, permutes construct labels,
recomputes pairwise distances under the null, and reports p-values.

AC-1: >= 3 of all available pairs reject permutation null at p < 0.005.

Usage (SageMaker or local):
    python compute_permutation_null.py --scenario houston --upload
    python compute_permutation_null.py --all --upload
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from io import BytesIO

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCENARIOS = [
    "houston", "southwest_florida", "nyc",
    "riverside_coachella", "new_orleans",
]
DATA_BUCKET = "swarm-floodrsct-data"
RESULT_PREFIX = "results/s035/doe_c1"
N_PERMUTATIONS = 10000
ALPHA = 0.005  # Bonferroni-adjusted threshold per AC-1


def _get_s3():
    """Get S3 client via swarm_auth or bare boto3."""
    try:
        from swarm_auth import get_aws_credentials
        import boto3
        return boto3.client("s3", **get_aws_credentials())
    except ImportError:
        import boto3
        return boto3.client("s3")


def _load_scenario_result(s3, scenario):
    """Download and parse a DOE-C1 result JSON from S3."""
    key = f"{RESULT_PREFIX}/five_construct_{scenario}.json"
    resp = s3.get_object(Bucket=DATA_BUCKET, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


def _extract_certificate_vectors(result):
    """Extract certificate vectors for available constructs.

    Returns:
        names: list of construct names (available only)
        vectors: np.ndarray of shape (n_constructs, 3)
            columns: [forward_score, kappa_spatial, kappa_reconstruct]
    """
    names = []
    vecs = []
    for c in result["per_construct"]:
        if not c["target_available"]:
            continue
        fs = c["forward_score"] if c["forward_score"] is not None else 0.0
        ks = c["kappa_spatial"] if c["kappa_spatial"] is not None else 0.0
        kr = c["kappa_reconstruct"] if c["kappa_reconstruct"] is not None else 0.0
        names.append(c["construct"])
        vecs.append([fs, ks, kr])
    return names, np.array(vecs)


def _pairwise_distances(vectors):
    """Compute all pairwise Euclidean distances.

    Returns:
        pairs: list of (i, j) tuples
        distances: np.ndarray of distances
    """
    n = len(vectors)
    pairs = []
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(vectors[i] - vectors[j])
            pairs.append((i, j))
            dists.append(d)
    return pairs, np.array(dists)


def _mean_pairwise_distance(vectors):
    """Mean of all pairwise Euclidean distances."""
    _, dists = _pairwise_distances(vectors)
    return float(np.mean(dists)) if len(dists) > 0 else 0.0


def run_permutation_test(names, vectors, n_perm=N_PERMUTATIONS, seed=42):
    """Run permutation test for construct exchangeability.

    Null hypothesis: construct labels are exchangeable (shuffling labels
    does not change the distribution of pairwise distances).

    For each permutation:
    1. Shuffle the assignment of certificate vectors to construct labels
    2. Recompute all pairwise distances
    3. Record per-pair distance and mean distance

    Returns dict with per-pair p-values and overall summary.
    """
    rng = np.random.RandomState(seed)
    n = len(vectors)
    pairs, observed_dists = _pairwise_distances(vectors)
    observed_mean = float(np.mean(observed_dists))

    log.info("  %d constructs, %d pairs, observed mean distance: %.4f",
             n, len(pairs), observed_mean)

    # Null distribution: permute row indices (shuffle which vector
    # gets which construct label) and recompute distances.
    # Under the null, any assignment is equally likely.
    null_pair_dists = np.zeros((n_perm, len(pairs)))
    null_means = np.zeros(n_perm)

    for p in range(n_perm):
        perm_idx = rng.permutation(n)
        perm_vectors = vectors[perm_idx]
        _, perm_dists = _pairwise_distances(perm_vectors)
        null_pair_dists[p] = perm_dists
        null_means[p] = float(np.mean(perm_dists))

    # Per-pair p-values: fraction of null distances >= observed
    pair_results = []
    for k, (i, j) in enumerate(pairs):
        obs_d = observed_dists[k]
        null_d = null_pair_dists[:, k]
        p_value = float(np.mean(null_d >= obs_d))
        pair_results.append({
            "construct_a": names[i],
            "construct_b": names[j],
            "observed_distance": round(float(obs_d), 4),
            "null_mean": round(float(np.mean(null_d)), 4),
            "null_std": round(float(np.std(null_d)), 4),
            "p_value": round(p_value, 6),
            "reject_at_005": p_value < ALPHA,
        })

    # Overall: p-value for mean distance
    mean_p = float(np.mean(null_means >= observed_mean))

    n_reject = sum(1 for pr in pair_results if pr["reject_at_005"])
    ac1_pass = n_reject >= 3

    return {
        "n_constructs": n,
        "n_pairs": len(pairs),
        "n_permutations": n_perm,
        "observed_mean_distance": round(observed_mean, 4),
        "null_mean_distance": round(float(np.mean(null_means)), 4),
        "null_std_distance": round(float(np.std(null_means)), 4),
        "overall_p_value": round(mean_p, 6),
        "per_pair": pair_results,
        "n_reject_at_005": n_reject,
        "ac1_pass": ac1_pass,
        "ac1_criterion": ">= 3 pairs reject at p < 0.005",
    }


def process_scenario(s3, scenario, upload=False, n_perm=N_PERMUTATIONS):
    """Run permutation test for one scenario."""
    log.info("Loading DOE-C1 result for %s ...", scenario)
    result = _load_scenario_result(s3, scenario)

    names, vectors = _extract_certificate_vectors(result)
    log.info("  Available constructs: %s", names)

    if len(names) < 2:
        log.warning("  Fewer than 2 constructs available; skipping.")
        return None

    perm_result = run_permutation_test(names, vectors, n_perm=n_perm)
    perm_result["scenario"] = scenario
    perm_result["timestamp"] = datetime.now(timezone.utc).isoformat()
    perm_result["source_artifact"] = (
        f"s3://{DATA_BUCKET}/{RESULT_PREFIX}/five_construct_{scenario}.json"
    )

    # Report
    log.info("  === Permutation Test: %s ===", scenario.upper())
    log.info("  Observed mean distance: %.4f", perm_result["observed_mean_distance"])
    log.info("  Null mean distance:     %.4f +/- %.4f",
             perm_result["null_mean_distance"], perm_result["null_std_distance"])
    log.info("  Overall p-value:        %.6f", perm_result["overall_p_value"])
    log.info("  Per-pair results:")
    for pr in perm_result["per_pair"]:
        flag = " ***" if pr["reject_at_005"] else ""
        log.info("    %s vs %s: d=%.3f, p=%.4f%s",
                 pr["construct_a"][:20], pr["construct_b"][:20],
                 pr["observed_distance"], pr["p_value"], flag)
    log.info("  Pairs rejecting null (p < %.3f): %d / %d",
             ALPHA, perm_result["n_reject_at_005"], perm_result["n_pairs"])
    log.info("  AC-1 PASS: %s", perm_result["ac1_pass"])

    if upload:
        key = f"{RESULT_PREFIX}/permutation_null_{scenario}.json"
        body = json.dumps(perm_result, indent=2).encode("utf-8")
        s3.put_object(Bucket=DATA_BUCKET, Key=key, Body=body,
                      ContentType="application/json")
        log.info("  Uploaded: s3://%s/%s", DATA_BUCKET, key)

    return perm_result


def main():
    parser = argparse.ArgumentParser(
        description="Permutation null test for construct exchangeability (AC-1)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIOS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to S3")
    parser.add_argument("--n-perm", type=int, default=N_PERMUTATIONS,
                        help="Number of permutations (default: 10000)")
    args = parser.parse_args()

    n_perm = args.n_perm
    s3 = _get_s3()

    scenarios = SCENARIOS if args.all else [args.scenario]
    all_results = []

    for scenario in scenarios:
        result = process_scenario(s3, scenario, upload=args.upload,
                                  n_perm=n_perm)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        # Cross-scenario summary
        total_pairs = sum(r["n_pairs"] for r in all_results)
        total_reject = sum(r["n_reject_at_005"] for r in all_results)
        print()
        print("=" * 60)
        print("AC-1 CROSS-SCENARIO SUMMARY")
        print("=" * 60)
        for r in all_results:
            print("  %-25s  %d/%d reject  overall p=%.4f  AC-1=%s" % (
                r["scenario"], r["n_reject_at_005"], r["n_pairs"],
                r["overall_p_value"], r["ac1_pass"]))
        print("-" * 60)
        print("  TOTAL: %d/%d pairs reject at p < %.3f" % (
            total_reject, total_pairs, ALPHA))
        ac1_global = total_reject >= 3
        print("  AC-1 GLOBAL: %s (need >= 3)" % ("PASS" if ac1_global else "FAIL"))
        print("=" * 60)

        if args.upload:
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_scenarios": len(all_results),
                "total_pairs": total_pairs,
                "total_reject_at_005": total_reject,
                "ac1_global_pass": ac1_global,
                "per_scenario": [
                    {
                        "scenario": r["scenario"],
                        "n_pairs": r["n_pairs"],
                        "n_reject": r["n_reject_at_005"],
                        "overall_p": r["overall_p_value"],
                        "ac1_pass": r["ac1_pass"],
                    }
                    for r in all_results
                ],
            }
            key = f"{RESULT_PREFIX}/permutation_null_summary.json"
            body = json.dumps(summary, indent=2).encode("utf-8")
            s3.put_object(Bucket=DATA_BUCKET, Key=key, Body=body,
                          ContentType="application/json")
            log.info("Uploaded summary: s3://%s/%s", DATA_BUCKET, key)


if __name__ == "__main__":
    main()
