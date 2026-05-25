"""
rsct_compat Analysis: Compatibility Surface across (Embedding × Task)

Mirrors the turbulence analysis in turbulence.txt, using theory_kappa_mean
as the compatibility measure. Answers:

  [1] Does embedding explain compatibility more than it explains R²?
      (η² decomposition: embedding vs task as variance sources)
  [2] Family-conditional compatibility: env vs health vs socioeconomic
  [3] Embedding × task interaction: which embeddings are substrate-selective?
  [4] rho(compat, r2) — is higher kappa associated with better R²?
  [5] Kappa vs turbulence (sigma): do compatible representations also settle?

Uses seed_42 s019d_results.json. Run from repo root:
    python exp/s019d_posthoc/analyze_compat.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, f_oneway
from scipy import stats

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = REPO_ROOT / "data" / "s019d" / "seed_42" / "s019d_results.json"

with open(DATA_FILE) as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Use theory_kappa_mean as rsct_compat: mean D*/D across the test set per fold
# (parallel to how rsct_turb was std of D*/D, measuring spread; this measures level)
df["rsct_compat"] = df["theory_kappa_mean"]

# Cell-level summary: mean over folds per (embedding, target)
cell = (
    df.groupby(["embedding", "target", "target_family"])
    .agg(
        rsct_compat=("rsct_compat", "mean"),
        r2=("r2", "mean"),
        theory_sigma=("theory_sigma", "mean"),   # canonical: std({kappa_i}), NOT sigma field (=N, degenerate proxy)
        theory_kappa=("theory_kappa", "mean"),
        proxy_kappa=("proxy_kappa", "mean"),
    )
    .reset_index()
)

EMB_ORDER = ["noisy_control", "pca_v1", "spatial_lag_v1", "gnn_v2", "geo_spatial", "domain_features"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def eta_squared_one_way(groups):
    """One-way eta² from list of arrays."""
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((v - grand_mean) ** 2) for g in groups for v in g)
    return ss_between / ss_total if ss_total > 0 else 0.0


def print_table(rows, headers, fmt=None):
    if fmt is None:
        fmt = ["<20"] + ["^10"] * (len(headers) - 1)
    header_line = "  ".join(f"{h:{f}}" for h, f in zip(headers, fmt))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(f"{v:{f}}" for v, f in zip(row, fmt)))


# ---------------------------------------------------------------------------
# [1] η² decomposition: how much of kappa variance is embedding vs task?
# ---------------------------------------------------------------------------

print("=" * 60)
print("[1] Variance decomposition: embedding vs task")
print("=" * 60)

for metric, label in [("rsct_compat", "rsct_compat"), ("r2", "r2")]:
    emb_groups = [cell[cell.embedding == e][metric].values for e in cell.embedding.unique()]
    task_groups = [cell[cell.target == t][metric].values for t in cell.target.unique()]

    eta_emb = eta_squared_one_way(emb_groups)
    eta_task = eta_squared_one_way(task_groups)

    print(f"\n  {label}:")
    print(f"    eta2(embedding) = {eta_emb:.3f}")
    print(f"    eta2(task)      = {eta_task:.3f}")
    print(f"    ratio task/emb  = {eta_task/eta_emb:.1f}x")

# ---------------------------------------------------------------------------
# [2] Embedding-level compatibility profile
# ---------------------------------------------------------------------------

print("\n")
print("=" * 60)
print("[2] Embedding compatibility profile")
print("=" * 60)

emb_summary = (
    cell.groupby("embedding")
    .agg(rsct_compat=("rsct_compat", "mean"), r2=("r2", "mean"), theory_sigma=("theory_sigma", "mean"))
    .reindex(EMB_ORDER)
)

print_table(
    [(e, f"{row.rsct_compat:.4f}", f"{row.r2:.3f}", f"{row.theory_sigma:.4f}")
     for e, row in emb_summary.iterrows()],
    ["embedding", "rsct_compat", "r2", "theory_sigma"],
)

print(f"\n  rho(embedding-mean compat, r2): "
      f"{spearmanr(emb_summary.rsct_compat, emb_summary.r2).statistic:.3f}")

# ---------------------------------------------------------------------------
# [3] Family-conditional compatibility
# ---------------------------------------------------------------------------

print("\n")
print("=" * 60)
print("[3] Family-conditional compatibility (env / socioeconomic / health)")
print("=" * 60)

for fam in ["environmental", "socioeconomic", "health"]:
    sub = cell[cell.target_family == fam]
    n = len(sub.target.unique())
    print(f"\n  {fam} (n={n} tasks):")
    emb_fam = sub.groupby("embedding").rsct_compat.mean().reindex(EMB_ORDER)
    for emb, val in emb_fam.items():
        print(f"    {emb:20s} {val:.4f}")
    print(f"    spread (max-min): {emb_fam.max() - emb_fam.min():.4f}")

# ---------------------------------------------------------------------------
# [4] rho(compat, r2) at cell level and by family
# ---------------------------------------------------------------------------

print("\n")
print("=" * 60)
print("[4] rho(rsct_compat, r2) — does kappa predict performance?")
print("=" * 60)

rho_all, p_all = spearmanr(cell.rsct_compat, cell.r2)
print(f"\n  All cells (n={len(cell)}): rho={rho_all:.3f}  p={p_all:.4f}")

for fam in ["environmental", "socioeconomic", "health"]:
    sub = cell[cell.target_family == fam]
    rho, p = spearmanr(sub.rsct_compat, sub.r2)
    print(f"  {fam:15s} (n={len(sub):3d}): rho={rho:.3f}  p={p:.4f}")

# ---------------------------------------------------------------------------
# [5] Kappa vs turbulence: do compatible representations also settle?
# ---------------------------------------------------------------------------

print("\n")
print("=" * 60)
print("[5] rho(rsct_compat, theory_sigma) — compatibility vs turbulence")
print("=" * 60)

rho_ks, p_ks = spearmanr(cell.rsct_compat, cell.theory_sigma)
print(f"\n  All cells: rho={rho_ks:.3f}  p={p_ks:.4f}")

# By embedding
print("\n  Per-embedding (mean compat vs mean theory_sigma):")
for emb, grp in cell.groupby("embedding"):
    print(f"    {emb:20s} compat={grp.rsct_compat.mean():.4f}  theory_sigma={grp.theory_sigma.mean():.4f}")

# ---------------------------------------------------------------------------
# [6] Top and bottom tasks by compatibility gap (max - min across embeddings)
# ---------------------------------------------------------------------------

print("\n")
print("=" * 60)
print("[6] Tasks with largest / smallest embedding compatibility spread")
print("=" * 60)

task_spread = (
    cell.groupby(["target", "target_family"])
    .rsct_compat.agg(["max", "min", "mean"])
    .assign(spread=lambda x: x["max"] - x["min"])
    .sort_values("spread", ascending=False)
    .reset_index()
)

print("\n  Top 5 (embedding-sensitive tasks):")
print_table(
    [(r["target"], r["target_family"], f"{r['mean']:.4f}", f"{r['spread']:.4f}")
     for _, r in task_spread.head(5).iterrows()],
    ["target", "family", "mean_kappa", "spread"],
    ["<25", "<15", "^12", "^10"],
)

print("\n  Bottom 5 (embedding-invariant tasks):")
print_table(
    [(r["target"], r["target_family"], f"{r['mean']:.4f}", f"{r['spread']:.4f}")
     for _, r in task_spread.tail(5).iterrows()],
    ["target", "family", "mean_kappa", "spread"],
    ["<25", "<15", "^12", "^10"],
)

# ---------------------------------------------------------------------------
# [7] Summary paragraph (mirrors turbulence.txt format)
# ---------------------------------------------------------------------------

print("\n")
print("=" * 60)
print("SUMMARY — rsct_compat surface (mirrors turbulence.txt)")
print("=" * 60)

eta_emb_compat = eta_squared_one_way(
    [cell[cell.embedding == e].rsct_compat.values for e in cell.embedding.unique()])
eta_task_compat = eta_squared_one_way(
    [cell[cell.target == t].rsct_compat.values for t in cell.target.unique()])
eta_emb_r2 = eta_squared_one_way(
    [cell[cell.embedding == e].r2.values for e in cell.embedding.unique()])
eta_task_r2 = eta_squared_one_way(
    [cell[cell.target == t].r2.values for t in cell.target.unique()])

rho_cr, _ = spearmanr(cell.rsct_compat, cell.r2)

best_emb = emb_summary.rsct_compat.idxmax()
worst_emb = emb_summary.rsct_compat.idxmin()
env_spread = (cell[cell.target_family=="environmental"]
              .groupby("embedding").rsct_compat.mean().max()
              - cell[cell.target_family=="environmental"]
              .groupby("embedding").rsct_compat.mean().min())
health_spread = (cell[cell.target_family=="health"]
                 .groupby("embedding").rsct_compat.mean().max()
                 - cell[cell.target_family=="health"]
                 .groupby("embedding").rsct_compat.mean().min())

print(f"""
  Key results from the (embedding x task) compatibility surface:

  [1] eta2 decomposition
      Source              eta2(r2)   eta2(rsct_compat)
      Embedding           {eta_emb_r2:.3f}       {eta_emb_compat:.3f}
      Task                {eta_task_r2:.3f}       {eta_task_compat:.3f}

  [2] Best compat embedding: {best_emb} ({emb_summary.loc[best_emb,'rsct_compat']:.4f})
      Worst:                  {worst_emb} ({emb_summary.loc[worst_emb,'rsct_compat']:.4f})

  [3] Family spread (max-min across embeddings):
      Environmental: {env_spread:.4f}
      Health:        {health_spread:.4f}

  [4] rho(rsct_compat, r2) = {rho_cr:.3f}

  [5] rho(rsct_compat, theory_sigma) = {rho_ks:.3f}
""")
