"""render_doe_c1_figures.py -- Generate paper figures from DOE-C1 results.

Produces:
  fig2_divergence_heatmap.pdf  — 5x5 divergence matrix (Houston primary)
  fig3_rsn_profiles.pdf        — Per-construct certificate bar chart
  fig5_cross_scenario.pdf      — Cross-scenario divergence comparison

Usage:
    python render_doe_c1_figures.py --upload
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_BUCKET = "swarm-floodrsct-data"
RESULT_PREFIX = "results/s035/doe_c1"
FIGURE_PREFIX = "results/s035/doe_c1/figures"

SCENARIOS = [
    "houston", "southwest_florida", "nyc",
    "riverside_coachella", "new_orleans",
]

# Short display names for constructs
CONSTRUCT_LABELS = {
    "jrc_observed_water": "JRC",
    "deltares_rp_depth": "Deltares",
    "fema_regulatory_zone": "FEMA",
    "fast_modeled_damage": "FAST",
    "nfip_administrative_loss": "NFIP",
}

CONSTRUCT_ORDER = [
    "jrc_observed_water",
    "deltares_rp_depth",
    "fema_regulatory_zone",
    "fast_modeled_damage",
    "nfip_administrative_loss",
]


def _get_s3():
    try:
        from swarm_auth import get_aws_credentials
        import boto3
        return boto3.client("s3", **get_aws_credentials())
    except ImportError:
        import boto3
        return boto3.client("s3")


def _load_result(s3, scenario):
    key = f"{RESULT_PREFIX}/five_construct_{scenario}.json"
    resp = s3.get_object(Bucket=DATA_BUCKET, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


def _available_constructs(result):
    """Return dict of construct_name -> certificate dict for available constructs."""
    out = {}
    for c in result["per_construct"]:
        if c["target_available"]:
            out[c["construct"]] = c
    return out


def render_fig2_divergence_heatmap(results, output_dir, scenario="houston"):
    """Fig 2: Divergence matrix heatmap for primary scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = results[scenario]
    avail = _available_constructs(result)
    avail_names = [c for c in CONSTRUCT_ORDER if c in avail]
    n = len(avail_names)

    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    pair_lookup = {}
    for p in result["pairwise"]:
        if p["both_available"] and p["euclidean_distance"] is not None:
            pair_lookup[(p["construct_a"], p["construct_b"])] = p["euclidean_distance"]
            pair_lookup[(p["construct_b"], p["construct_a"])] = p["euclidean_distance"]

    for i, ci in enumerate(avail_names):
        for j, cj in enumerate(avail_names):
            if i != j:
                dist_matrix[i, j] = pair_lookup.get((ci, cj), 0.0)

    labels = [CONSTRUCT_LABELS.get(c, c) for c in avail_names]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    im = ax.imshow(dist_matrix, cmap="YlOrRd", vmin=0, vmax=1.2)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "--"
                color = "gray"
            else:
                text = f"{dist_matrix[i, j]:.3f}"
                color = "white" if dist_matrix[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Certificate distance", fontsize=10)

    scenario_title = scenario.replace("_", " ").title()
    ax.set_title(f"Construct Divergence Matrix ({scenario_title})",
                 fontsize=11, pad=10)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_divergence_heatmap.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)
    return path


def render_fig3_rsn_profiles(results, output_dir, scenario="houston"):
    """Fig 3: Per-construct certificate profiles (forward_score, kappa_spatial)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = results[scenario]
    avail = _available_constructs(result)
    avail_names = [c for c in CONSTRUCT_ORDER if c in avail]
    labels = [CONSTRUCT_LABELS.get(c, c) for c in avail_names]

    forward_scores = []
    kappa_spatials = []
    for c in avail_names:
        cert = avail[c]
        forward_scores.append(cert["forward_score"] or 0.0)
        kappa_spatials.append(cert["kappa_spatial"] or 0.0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bars1 = ax.bar(x - width / 2, forward_scores, width, label="Forward score (R2)",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, kappa_spatials, width, label="Kappa spatial",
                   color="#FF9800", alpha=0.85)

    ax.set_ylabel("Certificate value", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.3, label="R gate (0.3)")

    scenario_title = scenario.replace("_", " ").title()
    ax.set_title(f"Per-Construct Certificate Profiles ({scenario_title})",
                 fontsize=11, pad=10)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_rsn_profiles.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)
    return path


def render_fig5_cross_scenario(results, output_dir):
    """Fig 5: Cross-scenario divergence comparison (grouped bar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_labels = []
    mean_dists = []
    max_dists = []

    for s in SCENARIOS:
        r = results[s]
        scenario_labels.append(s.replace("_", " ").title()[:12])
        mean_dists.append(r["mean_distance"])
        max_dists.append(r["max_distance"])

    x = np.arange(len(scenario_labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.bar(x - width / 2, mean_dists, width, label="Mean distance",
           color="#4CAF50", alpha=0.85)
    ax.bar(x + width / 2, max_dists, width, label="Max distance",
           color="#F44336", alpha=0.85)

    ax.set_ylabel("Certificate distance", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=9)
    ax.set_title("Construct Divergence Across Scenarios", fontsize=11, pad=10)

    # Annotate max pair above max bar
    for i, s in enumerate(SCENARIOS):
        r = results[s]
        pair = r["max_pair"]
        pair_short = "/".join([CONSTRUCT_LABELS.get(p, p[:4]) for p in pair])
        ax.text(x[i] + width / 2, max_dists[i] + 0.03, pair_short,
                ha="center", va="bottom", fontsize=7, rotation=20)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig5_cross_scenario.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Render DOE-C1 paper figures"
    )
    parser.add_argument("--upload", action="store_true",
                        help="Upload figures to S3")
    parser.add_argument("--output-dir", default="/tmp/doe_c1_figures",
                        help="Local output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    s3 = _get_s3()

    # Load all scenario results
    log.info("Loading DOE-C1 results for all scenarios...")
    results = {}
    for s in SCENARIOS:
        results[s] = _load_result(s3, s)
        log.info("  Loaded %s: %d constructs available",
                 s, sum(1 for c in results[s]["per_construct"] if c["target_available"]))

    # Render figures
    paths = []
    paths.append(render_fig2_divergence_heatmap(results, args.output_dir, "houston"))
    paths.append(render_fig3_rsn_profiles(results, args.output_dir, "houston"))
    paths.append(render_fig5_cross_scenario(results, args.output_dir))

    # Also render heatmaps for New Orleans (the anomaly)
    paths.append(render_fig2_divergence_heatmap(
        results, args.output_dir, "new_orleans"))
    # Rename the New Orleans one
    nola_src = os.path.join(args.output_dir, "fig2_divergence_heatmap.pdf")
    nola_dst = os.path.join(args.output_dir, "fig2b_divergence_heatmap_new_orleans.pdf")
    if os.path.exists(nola_src):
        os.rename(nola_src, nola_dst)
        paths[-1] = nola_dst

    # Re-render Houston (overwritten by NOLA)
    paths.append(render_fig2_divergence_heatmap(results, args.output_dir, "houston"))

    if args.upload:
        for path in paths:
            if path and os.path.exists(path):
                fname = os.path.basename(path)
                key = f"{FIGURE_PREFIX}/{fname}"
                s3.upload_file(path, DATA_BUCKET, key)
                log.info("Uploaded: s3://%s/%s", DATA_BUCKET, key)

    log.info("Done. %d figures generated.", len(paths))


if __name__ == "__main__":
    main()
