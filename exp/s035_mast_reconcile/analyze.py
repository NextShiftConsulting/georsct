"""
analyze.py — S035 MAST Reconcile analysis.

Loads 61 MAST annotated traces, extracts structural signals,
and tests whether MAST failure categories produce discriminative
signal profiles.

Usage:
    python analyze.py

Output:
    results/summary.json    — full results
    results/signals.csv     — per-trace signal matrix
    stdout                  — human-readable report
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from load_mast import load_all_traces, MAST_CATEGORIES, MODE_TO_CATEGORY, get_active_categories
from signals import extract_signals


RESULTS_DIR = Path(__file__).parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print("Loading MAST traces...")
    traces = load_all_traces()
    print(f"  {len(traces)} traces loaded ({sum(1 for t in traces if t['source']=='AG2')} AG2, "
          f"{sum(1 for t in traces if t['source']=='HyperAgent')} HyperAgent)")

    # ==========================================
    # 2. EXTRACT SIGNALS
    # ==========================================
    print("\nExtracting signals...")
    for trace in traces:
        trace["signals"] = extract_signals(trace)

    # ==========================================
    # 3. ANNOTATION STATISTICS
    # ==========================================
    print("\n" + "=" * 60)
    print("MAST ANNOTATION STATISTICS")
    print("=" * 60)

    mode_freq = Counter()
    for t in traces:
        for mode, active in t["annotations"].items():
            if active:
                mode_freq[mode] += 1

    n_failures_per_trace = [sum(t["annotations"].values()) for t in traces]
    print(f"\n  Traces with 0 failures: {n_failures_per_trace.count(0)}")
    print(f"  Traces with 1 failure:  {n_failures_per_trace.count(1)}")
    print(f"  Traces with 2+ failures: {sum(1 for n in n_failures_per_trace if n >= 2)}")
    print(f"  Mean failures/trace: {np.mean(n_failures_per_trace):.1f}")
    print(f"  Max failures/trace: {max(n_failures_per_trace)}")

    print(f"\n  Mode frequencies (n={len(traces)}):")
    for mode, count in mode_freq.most_common():
        cat = MODE_TO_CATEGORY.get(mode, "?")
        print(f"    {count:>3} ({count/len(traces)*100:4.0f}%) [{cat[:3]}] {mode}")

    # Category totals
    print(f"\n  Category totals (traces with at least one mode in category):")
    for cat_name in ["FC1_Specification", "FC2_Misalignment", "FC3_Verification"]:
        n = sum(1 for t in traces if cat_name in get_active_categories(t["annotations"]))
        print(f"    {cat_name}: {n} traces ({n/len(traces)*100:.0f}%)")

    # ==========================================
    # 4. SIGNAL PROFILES BY MAST CATEGORY
    # ==========================================
    print("\n" + "=" * 60)
    print("SIGNAL PROFILES BY MAST CATEGORY")
    print("=" * 60)

    # Group traces by MAST category
    cat_groups = defaultdict(list)
    for t in traces:
        cats = get_active_categories(t["annotations"])
        for c in cats:
            cat_groups[c].append(t)
    # Add no-failure baseline
    cat_groups["baseline"] = [t for t in traces if sum(t["annotations"].values()) == 0]

    signal_keys = [
        "repetition_ratio", "consecutive_dupe_rate", "dominance",
        "handoff_rate", "error_density", "verify_density",
        "code_density", "length_cv", "growth_ratio",
        "has_termination", "n_agents", "n_steps",
    ]

    # Print header
    short_keys = ["repet", "c_dup", "domin", "hndff", "error", "verif",
                  "code_", "ln_cv", "growt", "termi", "n_agt", "n_stp"]
    print(f"\n  {'Category':<20} {'n':>4}", end="")
    for sk in short_keys:
        print(f" {sk:>6}", end="")
    print()
    print("  " + "-" * (25 + 7 * len(short_keys)))

    cat_centroids = {}
    for cat_name in ["FC1_Specification", "FC2_Misalignment", "FC3_Verification", "baseline"]:
        group = cat_groups[cat_name]
        if not group:
            continue
        centroid = np.array([
            np.mean([t["signals"][k] for t in group]) for k in signal_keys
        ])
        cat_centroids[cat_name] = centroid

        label = cat_name.replace("_", " ")[:20]
        print(f"  {label:<20} {len(group):>4}", end="")
        for v in centroid:
            print(f" {v:>6.2f}", end="")
        print()

    # ==========================================
    # 5. DISCRIMINATION TEST
    # ==========================================
    print("\n" + "=" * 60)
    print("DISCRIMINATION: PAIRWISE CENTROID DISTANCES")
    print("=" * 60)

    # Normalize signals for distance computation
    all_signals = np.array([[t["signals"][k] for k in signal_keys] for t in traces])
    means = all_signals.mean(axis=0)
    stds = all_signals.std(axis=0)
    stds[stds == 0] = 1.0  # avoid div by zero

    cat_centroids_norm = {}
    for cat_name, group in cat_groups.items():
        if len(group) < 3:
            continue
        raw = np.array([[t["signals"][k] for k in signal_keys] for t in group])
        cat_centroids_norm[cat_name] = ((raw - means) / stds).mean(axis=0)

    print(f"\n  (Distances in z-score space, {len(signal_keys)} features)")
    cats_to_compare = [c for c in ["FC1_Specification", "FC2_Misalignment", "FC3_Verification", "baseline"]
                       if c in cat_centroids_norm]

    for i in range(len(cats_to_compare)):
        for j in range(i + 1, len(cats_to_compare)):
            c1, c2 = cats_to_compare[i], cats_to_compare[j]
            d = np.linalg.norm(cat_centroids_norm[c1] - cat_centroids_norm[c2])
            print(f"  {c1:<20} vs {c2:<20}: {d:.3f}")

    # ==========================================
    # 6. PER-MODE PROFILES (top modes)
    # ==========================================
    print("\n" + "=" * 60)
    print("PER-MODE SIGNAL PROFILES (modes with n >= 5)")
    print("=" * 60)

    mode_groups = defaultdict(list)
    for t in traces:
        for mode, active in t["annotations"].items():
            if active:
                mode_groups[mode].append(t)

    # Filter to modes with enough samples
    top_modes = [(m, c) for m, c in mode_freq.most_common() if c >= 5]

    print(f"\n  {'Mode':<42} {'n':>3}", end="")
    for sk in short_keys[:8]:
        print(f" {sk:>6}", end="")
    print()
    print("  " + "-" * (46 + 7 * 8))

    mode_centroids_norm = {}
    for mode, count in top_modes:
        group = mode_groups[mode]
        raw = np.array([[t["signals"][k] for k in signal_keys] for t in group])
        centroid_norm = ((raw - means) / stds).mean(axis=0)
        mode_centroids_norm[mode] = centroid_norm

        centroid_raw = raw.mean(axis=0)
        print(f"  {mode[:42]:<42} {count:>3}", end="")
        for v in centroid_raw[:8]:
            print(f" {v:>6.3f}", end="")
        print()

    # Mode-mode distances
    if len(mode_centroids_norm) >= 2:
        print(f"\n  Pairwise mode distances (z-score, top 5 most separated):")
        distances = []
        mode_names = list(mode_centroids_norm.keys())
        for i in range(len(mode_names)):
            for j in range(i + 1, len(mode_names)):
                d = np.linalg.norm(mode_centroids_norm[mode_names[i]] - mode_centroids_norm[mode_names[j]])
                distances.append((mode_names[i], mode_names[j], d))
        distances.sort(key=lambda x: -x[2])

        for m1, m2, d in distances[:5]:
            print(f"    {d:.3f}  {m1[:30]} vs {m2[:30]}")
        print(f"\n  Mean: {np.mean([d[2] for d in distances]):.3f}  "
              f"Min: {np.min([d[2] for d in distances]):.3f}  "
              f"Max: {np.max([d[2] for d in distances]):.3f}")

    # ==========================================
    # 7. SOURCE COMPARISON (AG2 vs HyperAgent)
    # ==========================================
    print("\n" + "=" * 60)
    print("SOURCE COMPARISON: AG2 vs HyperAgent")
    print("=" * 60)

    for source in ["AG2", "HyperAgent"]:
        group = [t for t in traces if t["source"] == source]
        sigs = np.array([[t["signals"][k] for k in signal_keys] for t in group])
        n_fail = [sum(t["annotations"].values()) for t in group]

        print(f"\n  {source} (n={len(group)}):")
        print(f"    Mean failures/trace: {np.mean(n_fail):.1f}")
        print(f"    Signals (mean):", end="")
        for k, v in zip(signal_keys[:6], sigs.mean(axis=0)[:6]):
            print(f"  {k[:5]}={v:.3f}", end="")
        print()

    # ==========================================
    # 8. CO-OCCURRENCE MATRIX
    # ==========================================
    print("\n" + "=" * 60)
    print("CO-OCCURRENCE (top 8 modes)")
    print("=" * 60)

    top8 = [m for m, _ in mode_freq.most_common(8)]
    cooc = np.zeros((8, 8), dtype=int)
    for t in traces:
        active = [m for m in top8 if t["annotations"].get(m, False)]
        for i, m1 in enumerate(top8):
            for j, m2 in enumerate(top8):
                if m1 in active and m2 in active:
                    cooc[i, j] += 1

    print(f"\n  {'':>4}", end="")
    for i in range(8):
        print(f" {i:>4}", end="")
    print()
    for i, mode in enumerate(top8):
        print(f"  {i}.", end="")
        for j in range(8):
            print(f" {cooc[i,j]:>4}", end="")
        print(f"  {mode[:40]}")

    # ==========================================
    # SAVE RESULTS
    # ==========================================
    # Signal matrix CSV
    csv_path = RESULTS_DIR / "signals.csv"
    with open(csv_path, "w") as f:
        header = ["instance_id", "source", "n_failures"] + signal_keys + list(traces[0]["annotations"].keys())
        f.write(",".join(header) + "\n")
        for t in traces:
            row = [
                t["instance_id"],
                t["source"],
                str(sum(t["annotations"].values())),
            ]
            row += [f"{t['signals'][k]:.6f}" for k in signal_keys]
            row += [str(int(t["annotations"].get(m, False))) for m in traces[0]["annotations"].keys()]
            f.write(",".join(row) + "\n")

    # Summary JSON
    summary = {
        "n_traces": len(traces),
        "sources": {"AG2": sum(1 for t in traces if t["source"] == "AG2"),
                    "HyperAgent": sum(1 for t in traces if t["source"] == "HyperAgent")},
        "annotation_stats": {
            "n_modes_observed": len(mode_freq),
            "mean_failures_per_trace": float(np.mean(n_failures_per_trace)),
            "mode_frequencies": dict(mode_freq.most_common()),
        },
        "category_profiles": {
            cat: {
                "n": len(cat_groups[cat]),
                "signals": {k: float(v) for k, v in zip(signal_keys, cat_centroids.get(cat, np.zeros(len(signal_keys))))}
            }
            for cat in ["FC1_Specification", "FC2_Misalignment", "FC3_Verification", "baseline"]
            if cat in cat_centroids
        },
        "category_distances_zscore": {},
        "mode_profiles": {
            mode: {
                "n": count,
                "category": MODE_TO_CATEGORY.get(mode, "unknown"),
            }
            for mode, count in top_modes
        },
    }

    # Add distances
    for i in range(len(cats_to_compare)):
        for j in range(i + 1, len(cats_to_compare)):
            c1, c2 = cats_to_compare[i], cats_to_compare[j]
            d = np.linalg.norm(cat_centroids_norm[c1] - cat_centroids_norm[c2])
            summary["category_distances_zscore"][f"{c1}_vs_{c2}"] = float(d)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
