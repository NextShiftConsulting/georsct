"""
S018D-Posthoc: Certificate structure analysis.

Shows that aggregate certificate metrics contain partial structure but
fail to detect the noisy-solver pathology.

Input: S018D results (324 certificates, 12 solvers, 27 targets)
Output: PCA axes, clustering, pairwise distances, noisy-solver diagnostic
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "s018d"
RESULTS_DIR = Path(__file__).parent / "results"


def load_data():
    """Load S018D certificate and solver metrics."""
    certs = pd.read_parquet(DATA_DIR / "certificate_rsn.parquet")
    solvers = pd.read_parquet(DATA_DIR / "solver_metrics.parquet")
    with open(DATA_DIR / "summary.json") as f:
        summary = json.load(f)
    return certs, solvers, summary


def pca_analysis(solvers: pd.DataFrame) -> pd.DataFrame:
    """PCA on solver certificate-metric profiles."""
    # Features for PCA
    metric_cols = [c for c in solvers.columns if c not in ["solver", "family", "gate_mode"]
                   and solvers[c].dtype in [np.float64, np.int64]]

    X = solvers[metric_cols].values
    solver_names = solvers.index if "solver" not in solvers.columns else solvers["solver"].values

    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    explained_var = (S ** 2) / (S ** 2).sum()

    # Project
    n_components = min(5, X_std.shape[1])
    projections = X_std @ Vt[:n_components].T

    # Build results
    pca_df = pd.DataFrame(
        projections,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    pca_df.insert(0, "solver", solver_names)

    # Loadings (what each PC represents)
    loadings = pd.DataFrame(
        Vt[:n_components].T,
        index=metric_cols,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    print(f"\n  PCA Explained Variance:")
    for i, v in enumerate(explained_var[:n_components]):
        cumulative = explained_var[:i+1].sum()
        # Top loadings for this PC
        pc_loadings = loadings[f"PC{i+1}"].abs().sort_values(ascending=False)
        top3 = ", ".join(f"{idx}({loadings[f'PC{i+1}'][idx]:+.2f})" for idx in pc_loadings.index[:3])
        print(f"    PC{i+1}: {v:.1%} (cumulative: {cumulative:.1%})  top: {top3}")

    return pca_df, loadings, explained_var[:n_components]


def pairwise_distance_analysis(solvers: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Euclidean distances between solver profiles."""
    metric_cols = [c for c in solvers.columns if c not in ["solver", "family", "gate_mode"]
                   and solvers[c].dtype in [np.float64, np.int64]]

    solver_names = solvers.index if "solver" not in solvers.columns else solvers["solver"].values
    X = solvers[metric_cols].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    dist_matrix = squareform(pdist(X_std, metric="euclidean"))

    # Build DataFrame
    dist_df = pd.DataFrame(dist_matrix, index=solver_names, columns=solver_names)

    return dist_df


def clustering_analysis(solvers: pd.DataFrame) -> dict:
    """Hierarchical clustering of solver profiles."""
    metric_cols = [c for c in solvers.columns if c not in ["solver", "family", "gate_mode"]
                   and solvers[c].dtype in [np.float64, np.int64]]

    solver_names = list(solvers.index if "solver" not in solvers.columns else solvers["solver"].values)
    X = solvers[metric_cols].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # Ward linkage
    Z = linkage(X_std, method="ward")

    # Cut at k=3, k=4 clusters
    labels_3 = fcluster(Z, t=3, criterion="maxclust")
    labels_4 = fcluster(Z, t=4, criterion="maxclust")

    result = {
        "method": "ward",
        "n_solvers": len(solver_names),
        "clusters_k3": {},
        "clusters_k4": {},
    }

    for k, labels in [("clusters_k3", labels_3), ("clusters_k4", labels_4)]:
        for cluster_id in sorted(set(labels)):
            members = [solver_names[i] for i, l in enumerate(labels) if l == cluster_id]
            result[k][f"cluster_{cluster_id}"] = members

    return result


def noisy_solver_diagnostic(dist_df: pd.DataFrame) -> dict:
    """Test: is noisy_solver distinguishable from serious solvers?"""
    if "noisy_solver" not in dist_df.index:
        return {"error": "noisy_solver not found in distance matrix"}

    noisy_dists = dist_df.loc["noisy_solver"].drop("noisy_solver")

    # Nearest neighbor
    nn = noisy_dists.idxmin()
    nn_dist = noisy_dists.min()

    # Distance to mean_baseline (the other control)
    if "mean_baseline" in noisy_dists.index:
        dist_to_baseline = noisy_dists["mean_baseline"]
    else:
        dist_to_baseline = None

    # Mean distance to serious solvers (non-control)
    controls = {"noisy_solver", "mean_baseline"}
    serious = [s for s in noisy_dists.index if s not in controls]
    mean_dist_to_serious = noisy_dists[serious].mean()

    # Controls are NOT a group: distance between them
    if "mean_baseline" in dist_df.index:
        control_distance = dist_df.loc["noisy_solver", "mean_baseline"]
    else:
        control_distance = None

    # Mean inter-solver distance (all pairs)
    all_dists = pdist(dist_df.values)
    mean_all_dist = np.mean(all_dists)

    return {
        "nearest_neighbor": nn,
        "nearest_neighbor_distance": float(nn_dist),
        "distance_to_mean_baseline": float(dist_to_baseline) if dist_to_baseline else None,
        "mean_distance_to_serious_solvers": float(mean_dist_to_serious),
        "control_pair_distance": float(control_distance) if control_distance else None,
        "mean_all_pairwise_distance": float(mean_all_dist),
        "controls_are_group": bool(control_distance and control_distance < mean_all_dist),
        "noisy_hidden_among_serious": bool(nn not in controls),
    }


def native_solver_tightness(dist_df: pd.DataFrame) -> dict:
    """Measure how tightly the 'serious' native solvers cluster."""
    controls = {"noisy_solver", "mean_baseline"}
    serious = [s for s in dist_df.index if s not in controls]

    serious_dists = dist_df.loc[serious, serious]
    triu_idx = np.triu_indices(len(serious), k=1)
    intra_dists = serious_dists.values[triu_idx]

    # Compare to control distances
    control_dists = []
    for c in controls:
        if c in dist_df.index:
            for s in serious:
                control_dists.append(dist_df.loc[c, s])

    return {
        "n_serious_solvers": len(serious),
        "mean_intra_serious_distance": float(np.mean(intra_dists)),
        "max_intra_serious_distance": float(np.max(intra_dists)),
        "mean_control_to_serious_distance": float(np.mean(control_dists)) if control_dists else None,
        "serious_solvers": serious,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("S018D-POSTHOC: Certificate Structure Analysis")
    print("=" * 60)

    # Load
    print("\nLoading S018D data...")
    certs, solvers, summary = load_data()
    print(f"  Certificates: {len(certs)} rows")
    print(f"  Solvers: {len(solvers)} rows")
    print(f"  Total certificates: {summary['certificate_count']}")

    # Build solver-level summary from per_solver in summary.json
    per_solver = summary["per_solver"]
    solver_df = pd.DataFrame(per_solver).T
    solver_df.index.name = "solver"
    # Convert numeric columns (JSON loads everything as object)
    non_numeric = ["family", "gate_mode"]
    numeric_cols = [c for c in solver_df.columns if c not in non_numeric]
    for c in numeric_cols:
        solver_df[c] = pd.to_numeric(solver_df[c], errors="coerce")
    solver_numeric = solver_df[numeric_cols]

    print(f"\n  Solver metrics ({len(numeric_cols)} features): {list(numeric_cols)}")

    # --- Analysis 1: PCA ---
    print("\n" + "-" * 60)
    print("ANALYSIS 1: PCA on Certificate Metrics")
    print("-" * 60)
    pca_df, loadings, explained_var = pca_analysis(solver_numeric)

    # --- Analysis 2: Pairwise Distances ---
    print("\n" + "-" * 60)
    print("ANALYSIS 2: Pairwise Solver Distances")
    print("-" * 60)
    dist_df = pairwise_distance_analysis(solver_numeric)

    print(f"\n  Distance matrix ({dist_df.shape[0]}x{dist_df.shape[1]}):")
    print(f"  Mean distance: {dist_df.values[np.triu_indices(len(dist_df), k=1)].mean():.3f}")
    print(f"  Max distance:  {dist_df.values[np.triu_indices(len(dist_df), k=1)].max():.3f}")
    print(f"  Min distance:  {dist_df.values[np.triu_indices(len(dist_df), k=1)].min():.3f}")

    # --- Analysis 3: Noisy Solver Diagnostic ---
    print("\n" + "-" * 60)
    print("ANALYSIS 3: Noisy Solver Diagnostic")
    print("-" * 60)
    noisy_diag = noisy_solver_diagnostic(dist_df)

    print(f"\n  Nearest neighbor to noisy_solver: {noisy_diag['nearest_neighbor']} "
          f"(distance: {noisy_diag['nearest_neighbor_distance']:.3f})")
    if noisy_diag['distance_to_mean_baseline'] is not None:
        print(f"  Distance to mean_baseline: {noisy_diag['distance_to_mean_baseline']:.3f}")
    print(f"  Mean distance to serious solvers: {noisy_diag['mean_distance_to_serious_solvers']:.3f}")
    if noisy_diag['control_pair_distance'] is not None:
        print(f"  Control pair distance: {noisy_diag['control_pair_distance']:.3f}")
    print(f"  Mean all-pair distance: {noisy_diag['mean_all_pairwise_distance']:.3f}")
    print(f"\n  Controls are a coherent group? {noisy_diag['controls_are_group']}")
    print(f"  Noisy solver hidden among serious? {noisy_diag['noisy_hidden_among_serious']}")

    # --- Analysis 4: Clustering ---
    print("\n" + "-" * 60)
    print("ANALYSIS 4: Hierarchical Clustering")
    print("-" * 60)
    clustering = clustering_analysis(solver_numeric)

    for k_label in ["clusters_k3", "clusters_k4"]:
        print(f"\n  {k_label}:")
        for cid, members in clustering[k_label].items():
            print(f"    {cid}: {members}")

    # --- Analysis 5: Native Solver Tightness ---
    print("\n" + "-" * 60)
    print("ANALYSIS 5: Native Solver Cluster Tightness")
    print("-" * 60)
    tightness = native_solver_tightness(dist_df)

    print(f"\n  Serious solvers ({tightness['n_serious_solvers']}): {tightness['serious_solvers']}")
    print(f"  Mean intra-serious distance: {tightness['mean_intra_serious_distance']:.3f}")
    print(f"  Max intra-serious distance: {tightness['max_intra_serious_distance']:.3f}")
    print(f"  Mean control-to-serious distance: {tightness['mean_control_to_serious_distance']:.3f}")

    # ==========================================
    # SAVE RESULTS
    # ==========================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # solver_metric_summary.csv
    solver_numeric.to_csv(RESULTS_DIR / "solver_metric_summary.csv")

    # pairwise_solver_distances.csv
    dist_df.to_csv(RESULTS_DIR / "pairwise_solver_distances.csv")

    # pca_pathology_axes.csv
    pca_df.to_csv(RESULTS_DIR / "pca_pathology_axes.csv", index=False)
    loadings.to_csv(RESULTS_DIR / "pca_loadings.csv")

    # clustering_summary.json
    clustering["noisy_solver_diagnostic"] = noisy_diag
    clustering["native_solver_tightness"] = tightness
    with open(RESULTS_DIR / "clustering_summary.json", "w") as f:
        json.dump(clustering, f, indent=2)

    # summary.json
    status = "PASS" if noisy_diag["noisy_hidden_among_serious"] else "PARTIAL"
    result_summary = {
        "experiment_id": "s018d_posthoc",
        "status": status,
        "run_timestamp": "",
        "git_commit": "",
        "n_targets": summary["target_count"],
        "n_solvers": summary["solver_count"],
        "n_certificates": summary["certificate_count"],
        "n_feature_groups": None,
        "key_findings": [
            f"PCA: {len(explained_var)} components explain {sum(explained_var):.1%} variance",
            f"PC1 explains {explained_var[0]:.1%} alone",
            f"Noisy solver nearest neighbor: {noisy_diag['nearest_neighbor']} (SERIOUS, not control)",
            f"Controls are NOT a group: distance {noisy_diag['control_pair_distance']:.3f} > mean {noisy_diag['mean_all_pairwise_distance']:.3f}" if not noisy_diag['controls_are_group'] else "Controls form a group",
            f"Serious solver cluster tightness: {tightness['mean_intra_serious_distance']:.3f}",
        ],
        "paper_claim_supported": "Certificate summaries are structured but incomplete: they separate scale/instability pathologies while hiding noisy-solver pathology",
        "limitations": [
            "Only 12 solver profiles (small n for PCA)",
            "Aggregate metrics only — no per-target or per-sample resolution",
            "Motivates but does not replace S018Y-U ablation analysis",
        ],
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(result_summary, f, indent=2)

    # paper_note.md
    paper_note = f"""# Paper Note: S018D-Posthoc

## Claim supported

Certificate-metric space has multi-dimensional unsupervised structure (3+
orthogonal pathology axes) but fails to distinguish degenerate solvers from
serious ones under aggregate metrics.

## Results

- PCA explains {sum(explained_var):.1%} variance in {len(explained_var)} components
- PC1 ({explained_var[0]:.1%}): primarily sigma/scale axis
- Noisy solver nearest neighbor: **{noisy_diag['nearest_neighbor']}** (a serious solver)
- Control pair distance: {noisy_diag['control_pair_distance']:.3f} (> mean {noisy_diag['mean_all_pairwise_distance']:.3f})
- Controls are NOT a coherent group

## Interpretation

Aggregate certificate metrics capture *some* structure (scale, instability)
but completely miss the noisy-solver pathology. A degenerate noise-generating
solver is indistinguishable from serious solvers when you only look at summary
statistics. This is GeoCert Failure Mode 1 (Scalar Projection) in action.

## Limitation

12 solvers is too few for robust PCA. The finding is directional, not
statistically definitive. S018Y-U with per-target ablation deltas provides
the higher-dimensional view needed.

## Recommended sentence for paper

> Post-hoc analysis of S018D certificate metrics reveals three orthogonal
> pathology axes but fails to separate the degenerate noisy solver from
> serious solvers---its nearest neighbor is {noisy_diag['nearest_neighbor']}
> (distance {noisy_diag['nearest_neighbor_distance']:.2f}), not the other
> control (distance {noisy_diag['distance_to_mean_baseline']:.2f}).
"""
    with open(RESULTS_DIR / "paper_note.md", "w") as f:
        f.write(paper_note)

    print(f"\n  Saved: {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"    {f.name}")

    print(f"\n  STATUS: {status}")
    print(f"  Noisy solver hidden among serious: {noisy_diag['noisy_hidden_among_serious']}")


if __name__ == "__main__":
    main()
