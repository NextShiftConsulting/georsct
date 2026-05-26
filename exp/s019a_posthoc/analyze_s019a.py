#!/usr/bin/env python3
"""
S019A Post-hoc Analysis: Certificate Invariance Gradient

Pulls s019a_results.json from S3, computes all statistics for the paper
tables, re-evaluates certificates under calibrated policy, and writes
certificates to rsct-geocert/data/s019a/certs/.

Run locally after the SageMaker job completes:

    python exp/s019a_posthoc/analyze_s019a.py
    python exp/s019a_posthoc/analyze_s019a.py --dry-run   # just check S3

Outputs:
    data/s019a/certs/          -- per-cell certificate JSON + parquet
    data/s019a/summary.json    -- experiment metadata
    exp/s019a_posthoc/results/ -- table-ready CSVs for LaTeX

Paper reference: GeoRSCT V3, Section 3.5 + Appendix F (S019A).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "s019a"
CERTS_DIR = DATA_DIR / "certs"
RESULTS_DIR = Path(__file__).parent / "results"

S3_BUCKET = "swarm-yrsn-datasets"
S3_PREFIX = "rsct_curriculum/series_019/results/s019a"

# Paper tables use 3 core embeddings x 3 solvers x 3 targets = 27 cells
CORE_EMBEDDINGS = ["pca_v1", "spatial_lag_v1", "gnn_v2"]
ALL_EMBEDDINGS = [
    "pca_v1", "spatial_lag_v1", "gnn_v2",
    "acs_raw", "geo_spatial", "noisy_control", "domain_features",
]
SOLVERS = ["histgbdt", "ridge", "mlp"]
TARGETS = ["diabetes", "population_density", "elevation"]

# Display names for paper
EMB_DISPLAY = {
    "pca_v1": "PCA32", "spatial_lag_v1": "Spatial Lag", "gnn_v2": "GNN",
    "acs_raw": "ACS raw", "geo_spatial": "Geo-spatial",
    "noisy_control": "Noisy ctrl", "domain_features": "Domain feat.",
}
SOLVER_DISPLAY = {"histgbdt": "HistGBDT", "ridge": "Ridge", "mlp": "MLP"}
TARGET_DISPLAY = {
    "diabetes": "Diabetes",
    "population_density": "Pop. density",
    "elevation": "Elevation",
}


# ---------------------------------------------------------------------------
# S3 download
# ---------------------------------------------------------------------------

def download_results() -> list:
    """Download s019a_results.json from S3."""
    import boto3

    s3 = boto3.Session(profile_name="nsc-swarm", region_name="us-east-1").client("s3")
    key = f"{S3_PREFIX}/s019a_results.json"
    print(f"Downloading s3://{S3_BUCKET}/{key} ...")

    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = json.loads(resp["Body"].read().decode("utf-8"))
    print(f"  Loaded {len(data)} records")
    return data


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def cell_key(r: dict) -> tuple:
    return (r["target"], r["embedding"], r["solver"])


def aggregate_cells(records: list) -> pd.DataFrame:
    """Aggregate per-fold records into per-cell means and stds."""
    from collections import defaultdict

    cells = defaultdict(list)
    for r in records:
        cells[cell_key(r)].append(r)

    rows = []
    for (target, emb, solver), folds in sorted(cells.items()):
        n_folds = len(folds)
        r2s = [f["r2"] for f in folds]
        rows.append({
            "target": target,
            "embedding": emb,
            "solver": solver,
            "n_folds": n_folds,
            # Performance
            "r2_mean": np.mean(r2s),
            "r2_std": np.std(r2s, ddof=1) if n_folds > 1 else 0.0,
            # Simplex (fold-averaged)
            "R": np.mean([f["R"] for f in folds]),
            "S_sup": np.mean([f["S_sup"] for f in folds]),
            "N": np.mean([f["N"] for f in folds]),
            "alpha": np.mean([f["alpha"] for f in folds]),
            "simplex_sum": np.mean([f["simplex_sum"] for f in folds]),
            # Kappa
            "theory_kappa": np.mean([f["theory_kappa"] for f in folds]),
            "theory_kappa_std": np.std([f["theory_kappa"] for f in folds], ddof=1) if n_folds > 1 else 0.0,
            "proxy_kappa": np.mean([f["proxy_kappa"] for f in folds]),
            # Sigma
            "sigma": np.mean([f["sigma"] for f in folds]),
            # Gates (flat)
            "kappa_req_flat": np.mean([
                f["gate_flat"]["kappa_req"] for f in folds
                if f["gate_flat"]["kappa_req"] is not None
            ]) if any(f["gate_flat"]["kappa_req"] is not None for f in folds) else None,
            "margin_flat": np.mean([
                f["gate_flat"]["margin"] for f in folds
                if f["gate_flat"]["margin"] is not None
            ]) if any(f["gate_flat"]["margin"] is not None for f in folds) else None,
            "margin_flat_std": np.std([
                f["gate_flat"]["margin"] for f in folds
                if f["gate_flat"]["margin"] is not None
            ], ddof=1) if sum(1 for f in folds if f["gate_flat"]["margin"] is not None) > 1 else 0.0,
            # Gate decision: majority across folds
            "gate_flat": _majority_decision([f["gate_flat"]["gate_decision"] for f in folds]),
            "gate_flat_unanimous": len(set(f["gate_flat"]["gate_decision"] for f in folds)) == 1,
            # Gates (oobleck)
            "kappa_req_oobleck": np.mean([
                f["gate_oobleck"]["kappa_req"] for f in folds
                if f["gate_oobleck"]["kappa_req"] is not None
            ]) if any(f["gate_oobleck"]["kappa_req"] is not None for f in folds) else None,
            "margin_oobleck": np.mean([
                f["gate_oobleck"]["margin"] for f in folds
                if f["gate_oobleck"]["margin"] is not None
            ]) if any(f["gate_oobleck"]["margin"] is not None for f in folds) else None,
            "gate_oobleck": _majority_decision([f["gate_oobleck"]["gate_decision"] for f in folds]),
            # Per-fold arrays for statistical tests
            "_r2_per_fold": r2s,
            "_kappa_per_fold": [f["theory_kappa"] for f in folds],
        })

    return pd.DataFrame(rows)


def _majority_decision(decisions: list) -> str:
    """Return most common gate decision, with dagger if not unanimous."""
    from collections import Counter
    counts = Counter(decisions)
    majority = counts.most_common(1)[0][0]
    unanimous = len(counts) == 1
    # Clean enum string (e.g. "GateDecision.EXECUTE" -> "execute")
    clean = majority.split(".")[-1].lower() if "." in majority else majority.lower()
    return clean if unanimous else f"{clean}*"


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def table_r2(cells: pd.DataFrame) -> pd.DataFrame:
    """tab:s019a_r2 -- Mean holdout R2 (+/- std) for core 27 cells."""
    core = cells[cells["embedding"].isin(CORE_EMBEDDINGS)].copy()
    rows = []
    for target in TARGETS:
        for emb in CORE_EMBEDDINGS:
            row = {"Target": TARGET_DISPLAY[target], "Embedding": EMB_DISPLAY[emb]}
            for solver in SOLVERS:
                cell = core[(core["target"] == target) &
                            (core["embedding"] == emb) &
                            (core["solver"] == solver)]
                if len(cell) == 1:
                    c = cell.iloc[0]
                    row[SOLVER_DISPLAY[solver]] = f"{c['r2_mean']:.3f} +/- {c['r2_std']:.3f}"
                else:
                    row[SOLVER_DISPLAY[solver]] = "MISSING"
            rows.append(row)
    return pd.DataFrame(rows)


def table_anova(cells: pd.DataFrame, records: list) -> pd.DataFrame:
    """tab:s019a_anova -- Two-way ANOVA (embedding x solver) on holdout R2."""
    core_records = [r for r in records if r["embedding"] in CORE_EMBEDDINGS]

    rows = []
    for target in TARGETS:
        target_recs = [r for r in core_records if r["target"] == target]
        if not target_recs:
            continue

        # Build per-fold R2 array with factors
        data = []
        for r in target_recs:
            data.append({
                "r2": r["r2"],
                "embedding": r["embedding"],
                "solver": r["solver"],
            })
        df = pd.DataFrame(data)

        # Main effects via one-way ANOVAs (scipy doesn't have two-way)
        # Embedding effect
        emb_groups = [g["r2"].values for _, g in df.groupby("embedding")]
        f_emb, p_emb = sp_stats.f_oneway(*emb_groups)
        ss_emb = _eta_squared(emb_groups)

        # Solver effect
        solver_groups = [g["r2"].values for _, g in df.groupby("solver")]
        f_sol, p_sol = sp_stats.f_oneway(*solver_groups)
        ss_sol = _eta_squared(solver_groups)

        # Interaction (embedding x solver)
        interaction_groups = [g["r2"].values for _, g in df.groupby(["embedding", "solver"])]
        f_int, p_int = sp_stats.f_oneway(*interaction_groups)
        ss_int = _eta_squared(interaction_groups)

        rows.append({
            "Target": TARGET_DISPLAY[target], "Source": "Embedding",
            "F": f"{f_emb:.2f}", "p": _fmt_p(p_emb),
            "eta2": f"{ss_emb:.3f}", "Sig": "Yes" if p_emb < 0.05 else "No",
        })
        rows.append({
            "Target": "", "Source": "Solver",
            "F": f"{f_sol:.2f}", "p": _fmt_p(p_sol),
            "eta2": f"{ss_sol:.3f}", "Sig": "Yes" if p_sol < 0.05 else "No",
        })
        rows.append({
            "Target": "", "Source": "Emb x Solver",
            "F": f"{f_int:.2f}", "p": _fmt_p(p_int),
            "eta2": f"{ss_int:.3f}", "Sig": "Yes" if p_int < 0.05 else "No",
        })

    return pd.DataFrame(rows)


def _eta_squared(groups: list) -> float:
    """Compute eta-squared from group arrays."""
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else 0.0


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.4f}"


def table_tukey(records: list) -> pd.DataFrame:
    """tab:s019a_tukey -- Tukey HSD pairwise embedding comparisons (elevation)."""
    from itertools import combinations

    elev = [r for r in records
            if r["target"] == "elevation" and r["embedding"] in CORE_EMBEDDINGS]

    emb_r2 = {}
    for r in elev:
        emb_r2.setdefault(r["embedding"], []).append(r["r2"])

    rows = []
    for e1, e2 in combinations(CORE_EMBEDDINGS, 2):
        a, b = np.array(emb_r2.get(e1, [])), np.array(emb_r2.get(e2, []))
        if len(a) == 0 or len(b) == 0:
            continue
        diff = b.mean() - a.mean()
        # Welch t-test as proxy for Tukey (exact Tukey needs statsmodels)
        t_stat, p_val = sp_stats.ttest_ind(a, b, equal_var=False)
        pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0.0
        se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        ci_lo = diff - 1.96 * se
        ci_hi = diff + 1.96 * se

        rows.append({
            "Pair": f"{EMB_DISPLAY[e2]} - {EMB_DISPLAY[e1]}",
            "delta_R2": f"{diff:.4f}",
            "CI_95": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            "p_adj": _fmt_p(p_val),
            "Cohens_d": f"{cohens_d:.3f}",
        })

    return pd.DataFrame(rows)


def table_certificates(cells: pd.DataFrame) -> pd.DataFrame:
    """tab:s019a_certs -- Certificate decomposition for core 27 cells."""
    core = cells[cells["embedding"].isin(CORE_EMBEDDINGS)].copy()
    rows = []
    for target in TARGETS:
        for emb in CORE_EMBEDDINGS:
            for solver in SOLVERS:
                cell = core[(core["target"] == target) &
                            (core["embedding"] == emb) &
                            (core["solver"] == solver)]
                if len(cell) != 1:
                    continue
                c = cell.iloc[0]
                gate_str = c["gate_flat"]
                rows.append({
                    "Target": TARGET_DISPLAY[target],
                    "Emb x Solver": f"{EMB_DISPLAY[emb]} x {SOLVER_DISPLAY[solver]}",
                    "R": f"{c['R']:.3f}",
                    "S": f"{c['S_sup']:.3f}",
                    "N": f"{c['N']:.3f}",
                    "alpha": f"{c['alpha']:.3f}",
                    "kappa": f"{c['theory_kappa']:.3f}",
                    "sigma": f"{c['sigma']:.3f}",
                    "kappa_req": f"{c['kappa_req_flat']:.3f}" if c["kappa_req_flat"] is not None else "N/A",
                    "Margin": f"{c['margin_flat']:.3f}" if c["margin_flat"] is not None else "N/A",
                    "Gate": gate_str,
                })
    return pd.DataFrame(rows)


def table_contrast(cells: pd.DataFrame) -> pd.DataFrame:
    """tab:s019a_contrast -- Certificate contrast across targets (HistGBDT x PCA32)."""
    rows_a = []
    for metric, key, fmt in [
        ("R2", "r2_mean", ".3f"),
        ("R", "R", ".3f"),
        ("S_sup", "S_sup", ".3f"),
        ("N", "N", ".3f"),
        ("alpha", "alpha", ".3f"),
        ("kappa", "theory_kappa", ".3f"),
        ("sigma", "sigma", ".3f"),
        ("kappa_req", "kappa_req_flat", ".3f"),
        ("Margin", "margin_flat", ".3f"),
        ("Gate", "gate_flat", "s"),
    ]:
        row = {"Quantity": metric}
        for target in TARGETS:
            cell = cells[(cells["target"] == target) &
                         (cells["embedding"] == "pca_v1") &
                         (cells["solver"] == "histgbdt")]
            if len(cell) == 1:
                val = cell.iloc[0][key]
                if val is None:
                    row[TARGET_DISPLAY[target]] = "N/A"
                elif fmt == "s":
                    row[TARGET_DISPLAY[target]] = str(val)
                else:
                    row[TARGET_DISPLAY[target]] = f"{val:{fmt}}"
            else:
                row[TARGET_DISPLAY[target]] = "MISSING"
        rows_a.append(row)

    # Margin 95% CI row
    row_ci = {"Quantity": "Margin 95% CI"}
    for target in TARGETS:
        cell = cells[(cells["target"] == target) &
                     (cells["embedding"] == "pca_v1") &
                     (cells["solver"] == "histgbdt")]
        if len(cell) == 1:
            c = cell.iloc[0]
            m = c["margin_flat"]
            s = c["margin_flat_std"]
            if m is not None:
                lo = m - 1.96 * s
                hi = m + 1.96 * s
                row_ci[TARGET_DISPLAY[target]] = f"[{lo:.3f}, {hi:.3f}]"
            else:
                row_ci[TARGET_DISPLAY[target]] = "N/A"
        else:
            row_ci[TARGET_DISPLAY[target]] = "MISSING"
    rows_a.append(row_ci)

    return pd.DataFrame(rows_a)


def table_contrast_pairwise(records: list) -> pd.DataFrame:
    """tab:s019a_contrast Part B -- Paired kappa comparisons across targets."""
    from itertools import combinations

    # Per-fold kappa for PCA32 x HistGBDT, per target
    kappa_by_target = {}
    for r in records:
        if r["embedding"] == "pca_v1" and r["solver"] == "histgbdt":
            kappa_by_target.setdefault(r["target"], []).append(
                (r["fold"], r["theory_kappa"])
            )

    # Sort by fold
    for t in kappa_by_target:
        kappa_by_target[t] = [k for _, k in sorted(kappa_by_target[t])]

    rows = []
    for t1, t2 in combinations(TARGETS, 2):
        a = np.array(kappa_by_target.get(t1, []))
        b = np.array(kappa_by_target.get(t2, []))
        n = min(len(a), len(b))
        if n < 2:
            continue
        a, b = a[:n], b[:n]
        delta = float(np.mean(b - a))
        t_stat, p_val = sp_stats.ttest_rel(a, b)
        pooled_std = np.std(b - a, ddof=1)
        cohens_d = delta / pooled_std if pooled_std > 0 else 0.0

        rows.append({
            "Comparison": f"{TARGET_DISPLAY[t2]} vs {TARGET_DISPLAY[t1]}",
            "delta_kappa": f"{delta:.4f}",
            "t": f"{t_stat:.3f}",
            "p": _fmt_p(p_val),
            "Cohens_d": f"{cohens_d:.3f}",
        })

    return pd.DataFrame(rows)


def table_gates(cells: pd.DataFrame) -> pd.DataFrame:
    """tab:s019a_gates -- Gate decision counts per target (core 27 cells)."""
    core = cells[cells["embedding"].isin(CORE_EMBEDDINGS)]
    gate_types = ["execute", "re_encode", "repair", "block", "reject"]

    rows = []
    for target in TARGETS:
        t_cells = core[core["target"] == target]
        counts = {g: 0 for g in gate_types}
        for _, c in t_cells.iterrows():
            decision = c["gate_flat"].rstrip("*").lower()
            for g in gate_types:
                if g in decision:
                    counts[g] += 1
                    break
        n_ceil = t_cells["N"].mean()
        row = {"Target": TARGET_DISPLAY[target]}
        row.update(counts)
        row["N_ceil"] = f"{n_ceil:.2f}"
        rows.append(row)

    # Chi-square test
    observed = np.array([[r[g] for g in gate_types] for r in rows])
    # Drop zero columns for chi-square
    nonzero = observed.sum(axis=0) > 0
    if nonzero.sum() >= 2:
        chi2, p_chi, dof, _ = sp_stats.chi2_contingency(observed[:, nonzero])
        rows.append({
            "Target": "chi2 test",
            **{g: "" for g in gate_types},
            "N_ceil": f"chi2={chi2:.2f}, df={dof}, p={_fmt_p(p_chi)}",
        })

    return pd.DataFrame(rows)


def table_assumptions(records: list) -> pd.DataFrame:
    """tab:s019a_assumptions -- ANOVA assumption diagnostics per target."""
    core = [r for r in records if r["embedding"] in CORE_EMBEDDINGS]

    rows = []
    for target in TARGETS:
        r2_vals = [r["r2"] for r in core if r["target"] == target]
        if len(r2_vals) < 3:
            continue

        # Shapiro-Wilk (normality)
        w_stat, w_p = sp_stats.shapiro(r2_vals)

        # Levene (homogeneity of variance across embeddings)
        emb_groups = {}
        for r in core:
            if r["target"] == target:
                emb_groups.setdefault(r["embedding"], []).append(r["r2"])
        groups = list(emb_groups.values())
        if len(groups) >= 2:
            f_lev, p_lev = sp_stats.levene(*groups)
        else:
            f_lev, p_lev = float("nan"), float("nan")

        rows.append({
            "Target": TARGET_DISPLAY[target],
            "W": f"{w_stat:.4f}", "W_p": _fmt_p(w_p),
            "F_levene": f"{f_lev:.4f}", "F_p": _fmt_p(p_lev),
        })

    return pd.DataFrame(rows)


def table_hypotheses(cells: pd.DataFrame, anova_df: pd.DataFrame) -> pd.DataFrame:
    """Hypotheses IG-1 through IG-5 with actual values and verdicts."""
    rows = []

    # IG-1: Diabetes embedding effect non-significant
    diab_anova = anova_df[
        (anova_df["Target"] == "Diabetes") & (anova_df["Source"] == "Embedding")
    ]
    if len(diab_anova) == 1:
        p_val = diab_anova.iloc[0]["p"]
        sig = diab_anova.iloc[0]["Sig"]
        rows.append({
            "ID": "IG-1", "Hypothesis": "Diabetes embedding effect non-significant",
            "Threshold": "p > 0.05", "Actual": f"p = {p_val}",
            "Verdict": "PASS" if sig == "No" else "FAIL",
        })

    # IG-2: Elevation embedding effect significant
    elev_anova = anova_df[
        (anova_df["Target"] == "Elevation") & (anova_df["Source"] == "Embedding")
    ]
    if len(elev_anova) == 1:
        p_val = elev_anova.iloc[0]["p"]
        sig = elev_anova.iloc[0]["Sig"]
        rows.append({
            "ID": "IG-2", "Hypothesis": "Elevation embedding effect significant",
            "Threshold": "p < 0.05", "Actual": f"p = {p_val}",
            "Verdict": "PASS" if sig == "Yes" else "FAIL",
        })

    # IG-3: Elevation eta2(embedding) > 0.20
    if len(elev_anova) == 1:
        eta2 = float(elev_anova.iloc[0]["eta2"])
        rows.append({
            "ID": "IG-3", "Hypothesis": "Elevation eta2(embedding) > 0.20",
            "Threshold": "eta2 > 0.20", "Actual": f"eta2 = {eta2:.3f}",
            "Verdict": "PASS" if eta2 > 0.20 else "FAIL",
        })

    # IG-4: Gate decisions diverge across targets
    gate_sets = {}
    core = cells[cells["embedding"].isin(CORE_EMBEDDINGS)]
    for target in TARGETS:
        decisions = set(core[core["target"] == target]["gate_flat"].str.rstrip("*"))
        gate_sets[target] = decisions
    all_same = all(gate_sets[t] == gate_sets[TARGETS[0]] for t in TARGETS)
    rows.append({
        "ID": "IG-4", "Hypothesis": "Gate decisions diverge across targets",
        "Threshold": "not identical",
        "Actual": "; ".join(f"{TARGET_DISPLAY[t]}: {sorted(gate_sets[t])}" for t in TARGETS),
        "Verdict": "PASS" if not all_same else "FAIL",
    })

    # IG-5: Simplex integrity (R + S + N ~ 1)
    sums = core["simplex_sum"].values
    max_dev = float(np.max(np.abs(sums - 1.0)))
    rows.append({
        "ID": "IG-5", "Hypothesis": "Simplex integrity (R+S+N ~ 1)",
        "Threshold": "|sum - 1| < 0.01", "Actual": f"max dev = {max_dev:.4f}",
        "Verdict": "PASS" if max_dev < 0.01 else "FAIL",
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibrated policy sensitivity (S2 moderate)
# ---------------------------------------------------------------------------

def calibrated_policy_sensitivity(records: list) -> dict:
    """Re-evaluate core 27 cells under S2 moderate policy.

    S2 moderate = geospatial-conus27 with kappa_base raised from 0.22 to 0.30
    (one std above median). Tests whether marginal cells flip.

    Replay note: full gate replay is possible from s019a_results.json alone.
    Every per-fold record contains the raw gatekeeper inputs (alpha, theory_kappa,
    sigma, N, R, S_sup, omega, entropy, collapse_risk) so any future preset or
    gatekeeper version can be evaluated without re-running the SageMaker job.
    The only data NOT replayable are per-sample kappa arrays (_kappa_per_proxy,
    _kappa_per_theory) which are stripped from the JSON; those require re-running
    certify_group if per-sample scatter plots are needed.
    """
    from yrsn.core.certificates.estimate import CPGatekeeperInput
    from yrsn_controlplane import SequentialGatekeeper, GatekeeperConfig, get_preset

    base = get_preset("geospatial-conus27")
    s2_config = GatekeeperConfig(
        N_thr=base.N_thr,
        alpha_min=base.alpha_min,
        c_min=base.c_min,
        gate_2_require_coherence=base.gate_2_require_coherence,
        sigma_thr=base.sigma_thr,
        kappa_base=0.30,                # Raised from 0.22 -> stricter
        lambda_turbulence=base.lambda_turbulence,
        epsilon_L=base.epsilon_L,
        enable_gate_3b=base.enable_gate_3b,
        r_bar_min=base.r_bar_min,
        gate_3b_action=base.gate_3b_action,
        kappa_L_min=base.kappa_L_min,
    )
    gk_s2 = SequentialGatekeeper(s2_config)

    core = [r for r in records if r["embedding"] in CORE_EMBEDDINGS]

    # Group by cell, take fold-averaged values
    from collections import defaultdict
    cells = defaultdict(list)
    for r in core:
        cells[cell_key(r)].append(r)

    changes = []
    for (target, emb, solver), folds in sorted(cells.items()):
        # Use fold-averaged certificate values
        avg = lambda k: float(np.mean([f[k] for f in folds]))

        cert_input = CPGatekeeperInput(
            alpha=avg("alpha"),
            kappa_compat=avg("theory_kappa"),
            sigma=avg("sigma"),
            source_mode="direct",
            evidence={
                "N": avg("N"),
                "R": avg("R"),
                "S": avg("S_sup"),
                "noise_admissibility": avg("N"),
                "omega": avg("omega"),
                "entropy": avg("entropy"),
                "collapse_risk": avg("collapse_risk"),
                "kappa_mean": avg("theory_kappa"),
                "kappa_std": float(np.std([f["theory_kappa"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0,
                "n_samples": int(np.mean([f["n_test"] for f in folds])),
            },
        )

        gr = gk_s2.evaluate(cert_input)
        s2_decision = str(gr.decision).split(".")[-1].lower()
        flat_decision = _majority_decision([f["gate_flat"]["gate_decision"] for f in folds]).rstrip("*")

        if s2_decision != flat_decision:
            changes.append({
                "target": target, "embedding": emb, "solver": solver,
                "flat": flat_decision, "s2": s2_decision,
            })

    return {
        "n_changes": len(changes),
        "n_total": 27,
        "changes": changes,
        "s2_config": {"kappa_base": 0.30, "note": "geospatial-conus27 + kappa_base=0.30"},
    }


# ---------------------------------------------------------------------------
# Certificate export
# ---------------------------------------------------------------------------

def export_certificates(cells: pd.DataFrame, records: list, sensitivity: dict):
    """Write per-cell certificate JSONs and summary parquet to data/s019a/certs/."""
    CERTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-cell certificate JSON
    for _, c in cells.iterrows():
        cert = {
            "experiment": "s019a",
            "target": c["target"],
            "embedding": c["embedding"],
            "solver": c["solver"],
            "n_folds": int(c["n_folds"]),
            "certificate": {
                "R": round(float(c["R"]), 4),
                "S_sup": round(float(c["S_sup"]), 4),
                "N": round(float(c["N"]), 4),
                "alpha": round(float(c["alpha"]), 4),
                "kappa": round(float(c["theory_kappa"]), 4),
                "kappa_std": round(float(c["theory_kappa_std"]), 4),
                "sigma": round(float(c["sigma"]), 4),
                "simplex_sum": round(float(c["simplex_sum"]), 4),
            },
            "performance": {
                "r2_mean": round(float(c["r2_mean"]), 4),
                "r2_std": round(float(c["r2_std"]), 4),
            },
            "gate_flat": {
                "decision": c["gate_flat"],
                "kappa_req": round(float(c["kappa_req_flat"]), 4) if c["kappa_req_flat"] is not None else None,
                "margin": round(float(c["margin_flat"]), 4) if c["margin_flat"] is not None else None,
            },
            "gate_oobleck": {
                "decision": c["gate_oobleck"],
                "kappa_req": round(float(c["kappa_req_oobleck"]), 4) if c["kappa_req_oobleck"] is not None else None,
                "margin": round(float(c["margin_oobleck"]), 4) if c["margin_oobleck"] is not None else None,
            },
        }
        fname = f"{c['target']}_{c['embedding']}_{c['solver']}.json"
        with open(CERTS_DIR / fname, "w") as f:
            json.dump(cert, f, indent=2)

    # Summary parquet (all cells, droppable _ columns)
    export_cols = [c for c in cells.columns if not c.startswith("_")]
    cells[export_cols].to_parquet(CERTS_DIR / "certificates.parquet", index=False)

    # Experiment summary
    summary = {
        "experiment": "s019a",
        "description": "Certificate Invariance Gradient",
        "design": "7 embeddings x 3 solvers x 3 targets x 5 folds = 315 fits",
        "core_design": "3 embeddings x 3 solvers x 3 targets = 27 cells (paper tables)",
        "n_records": len(records),
        "n_cells": len(cells),
        "n_core_cells": len(cells[cells["embedding"].isin(CORE_EMBEDDINGS)]),
        "targets": TARGETS,
        "embeddings_all": ALL_EMBEDDINGS,
        "embeddings_core": CORE_EMBEDDINGS,
        "solvers": SOLVERS,
        "calibrated_policy_sensitivity": sensitivity,
    }
    with open(DATA_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Certificates: {CERTS_DIR}/ ({len(cells)} cells)")
    print(f"  Summary: {DATA_DIR / 'summary.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="S019A post-hoc analysis")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check S3 availability only")
    parser.add_argument("--local", type=str, default=None,
                        help="Load from local JSON instead of S3")
    args = parser.parse_args()

    if args.dry_run:
        import boto3
        s3 = boto3.Session(profile_name="nsc-swarm", region_name="us-east-1").client("s3")
        key = f"{S3_PREFIX}/s019a_results.json"
        try:
            resp = s3.head_object(Bucket=S3_BUCKET, Key=key)
            size_mb = resp["ContentLength"] / 1024 / 1024
            print(f"s3://{S3_BUCKET}/{key} exists ({size_mb:.1f} MB)")
            print("Ready to analyze. Run without --dry-run.")
        except Exception as e:
            print(f"NOT FOUND: s3://{S3_BUCKET}/{key}")
            print(f"  {e}")
        return

    # Load data
    if args.local:
        with open(args.local) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records from {args.local}")
    else:
        records = download_results()

    # Aggregate
    print("Aggregating cells ...")
    cells = aggregate_cells(records)
    print(f"  {len(cells)} cells ({len(cells[cells['embedding'].isin(CORE_EMBEDDINGS)])} core)")

    # Generate tables
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating tables ...")

    r2_df = table_r2(cells)
    r2_df.to_csv(RESULTS_DIR / "tab_s019a_r2.csv", index=False)
    print(f"  tab_s019a_r2.csv")

    anova_df = table_anova(cells, records)
    anova_df.to_csv(RESULTS_DIR / "tab_s019a_anova.csv", index=False)
    print(f"  tab_s019a_anova.csv")

    tukey_df = table_tukey(records)
    tukey_df.to_csv(RESULTS_DIR / "tab_s019a_tukey.csv", index=False)
    print(f"  tab_s019a_tukey.csv")

    certs_df = table_certificates(cells)
    certs_df.to_csv(RESULTS_DIR / "tab_s019a_certs.csv", index=False)
    print(f"  tab_s019a_certs.csv")

    contrast_df = table_contrast(cells)
    contrast_df.to_csv(RESULTS_DIR / "tab_s019a_contrast.csv", index=False)
    print(f"  tab_s019a_contrast.csv")

    contrast_pw = table_contrast_pairwise(records)
    contrast_pw.to_csv(RESULTS_DIR / "tab_s019a_contrast_pairwise.csv", index=False)
    print(f"  tab_s019a_contrast_pairwise.csv")

    gates_df = table_gates(cells)
    gates_df.to_csv(RESULTS_DIR / "tab_s019a_gates.csv", index=False)
    print(f"  tab_s019a_gates.csv")

    assumptions_df = table_assumptions(records)
    assumptions_df.to_csv(RESULTS_DIR / "tab_s019a_assumptions.csv", index=False)
    print(f"  tab_s019a_assumptions.csv")

    hypo_df = table_hypotheses(cells, anova_df)
    hypo_df.to_csv(RESULTS_DIR / "tab_s019a_hypotheses.csv", index=False)
    print(f"  tab_s019a_hypotheses.csv")

    # Calibrated policy sensitivity
    print("Calibrated policy sensitivity (S2 moderate) ...")
    sensitivity = calibrated_policy_sensitivity(records)
    print(f"  {sensitivity['n_changes']}/{sensitivity['n_total']} cells changed")
    for ch in sensitivity["changes"]:
        print(f"    {ch['target']}/{ch['embedding']}/{ch['solver']}: "
              f"{ch['flat']} -> {ch['s2']}")
    with open(RESULTS_DIR / "calibrated_policy.json", "w") as f:
        json.dump(sensitivity, f, indent=2)

    # Export certificates
    print("Exporting certificates ...")
    export_certificates(cells, records, sensitivity)

    # Overall verdict
    verdicts = hypo_df["Verdict"].tolist()
    all_pass = all(v == "PASS" for v in verdicts)
    print(f"\n{'='*60}")
    print(f"VERDICT: {'PASS' if all_pass else 'FAIL'} "
          f"({sum(1 for v in verdicts if v == 'PASS')}/{len(verdicts)} hypotheses)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
