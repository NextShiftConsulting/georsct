"""
E6: Periodic Storm Season Seam
===============================

Validates kappa_periodic on synthetic cyclic data calibrated from CONUS-27 statistics.
Tests whether FFT spectral concentration detects periodic structure in residuals.

Flood question: "Does encoding handle storm season boundaries (Dec-Jan seam)?"

Pass criteria:
  - SCR > 0.3 for signals with known periodicity
  - kappa_periodic differentiates periodic from aperiodic residuals
  - SCR near 0 for white noise (control)

Usage:
  python run_E6_periodic_seam.py [--output-dir outputs/E6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YRSN_SRC = Path(__file__).resolve().parents[3] / "yrsn" / "src"
if str(YRSN_SRC) not in sys.path:
    sys.path.insert(0, str(YRSN_SRC))

from yrsn.core.kappa.periodic import compute_kappa_periodic


def generate_test_signals(n: int = 360, seed: int = 42) -> dict:
    """Generate test signals with known periodic properties.

    Simulates residuals from models with and without cyclic encoding,
    calibrated to CONUS-27 flood statistics.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi * 3, n)  # 3 full cycles

    signals = {}

    # 1. Strong 12-month seasonal pattern (hurricane season: June-Nov)
    seasonal = 0.6 * np.sin(t) + 0.3 * np.sin(2 * t) + rng.normal(0, 0.2, n)
    signals["flood_seasonal_strong"] = {
        "residuals": seasonal,
        "expected_periodic": True,
        "description": "Strong seasonal pattern (hurricane season)",
    }

    # 2. Weak seasonal pattern (winter flooding)
    weak_seasonal = 0.2 * np.sin(t) + rng.normal(0, 0.5, n)
    signals["flood_seasonal_weak"] = {
        "residuals": weak_seasonal,
        "expected_periodic": True,
        "description": "Weak seasonal pattern (winter floods)",
    }

    # 3. White noise (no periodicity - control)
    noise = rng.normal(0, 1, n)
    signals["white_noise_control"] = {
        "residuals": noise,
        "expected_periodic": False,
        "description": "White noise control (no periodicity)",
    }

    # 4. Linear trend (non-periodic structure)
    trend = np.linspace(0, 2, n) + rng.normal(0, 0.3, n)
    signals["linear_trend"] = {
        "residuals": trend,
        "expected_periodic": False,
        "description": "Linear trend (non-periodic)",
    }

    # 5. Mixed: seasonal + county-level jumps
    county_jumps = np.zeros(n)
    for i in range(0, n, n // 5):
        county_jumps[i:i + n // 10] = rng.normal(0, 0.5)
    mixed = 0.4 * np.sin(t) + county_jumps + rng.normal(0, 0.2, n)
    signals["flood_mixed_seasonal_county"] = {
        "residuals": mixed,
        "expected_periodic": True,
        "description": "Seasonal + county-level jumps",
    }

    # 6. Cyclic encoding residual (smooth wraparound)
    cyclic_resid = 0.3 * np.sin(t) + 0.1 * np.sin(3 * t) + rng.normal(0, 0.15, n)
    signals["cyclic_encoding_residual"] = {
        "residuals": cyclic_resid,
        "expected_periodic": True,
        "description": "Residuals from cyclic-encoded model",
    }

    # 7. Linear encoding residual (seam discontinuity)
    linear_resid = rng.normal(0, 0.3, n)
    # Add seam spikes at wraparound points
    period = n // 3
    for p in range(3):
        seam_idx = p * period
        width = max(1, period // 20)
        linear_resid[seam_idx:seam_idx + width] += 1.5
    signals["linear_encoding_residual"] = {
        "residuals": linear_resid,
        "expected_periodic": True,
        "description": "Residuals from linear-encoded model (seam spikes)",
    }

    return signals


def compute_all_periodic_kappas(signals: dict) -> dict:
    """Compute kappa_periodic for all test signals."""
    results = {}
    for name, sig in sorted(signals.items()):
        result = compute_kappa_periodic(
            residuals=sig["residuals"],
            min_period=3,
        )
        results[name] = {
            "kappa": result.kappa,
            "spectral_ratio": result.spectral_ratio,
            "dominant_period": result.dominant_period,
            "n_samples": result.n_samples,
            "expected_periodic": sig["expected_periodic"],
            "description": sig["description"],
        }
    return results


def render_figure(signals: dict, results: dict, output_path: Path):
    """Render the publishable periodic seam figure."""
    names = sorted(results.keys(), key=lambda n: results[n]["kappa"], reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel A: kappa_periodic bar chart
    kappas = [results[n]["kappa"] for n in names]
    expected = [results[n]["expected_periodic"] for n in names]
    bar_colors = ["#4C72B0" if e else "#DD8452" for e in expected]
    axes[0, 0].barh(range(len(names)), kappas, color=bar_colors, edgecolor="gray", linewidth=0.5)
    axes[0, 0].set_yticks(range(len(names)))
    axes[0, 0].set_yticklabels([n.replace("_", " ")[:30] for n in names], fontsize=8)
    axes[0, 0].set_xlabel("kappa_periodic (spectral concentration ratio)")
    axes[0, 0].set_title("Panel A: kappa_periodic by Signal Type", fontweight="bold")
    axes[0, 0].set_xlim(0, 1)
    # Legend
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[
        Patch(facecolor="#4C72B0", label="Expected periodic"),
        Patch(facecolor="#DD8452", label="Expected aperiodic"),
    ], fontsize=8)

    # Panel B: Time series of 3 key signals
    key_signals = ["flood_seasonal_strong", "white_noise_control", "linear_encoding_residual"]
    colors_ts = ["#4C72B0", "#DD8452", "#55A868"]
    for i, name in enumerate(key_signals):
        if name in signals:
            resid = signals[name]["residuals"]
            axes[0, 1].plot(resid[:120], color=colors_ts[i], alpha=0.7,
                           label=f"{name.replace('_', ' ')[:25]} (k={results[name]['kappa']:.2f})")
    axes[0, 1].set_xlabel("Sample index")
    axes[0, 1].set_ylabel("Residual")
    axes[0, 1].set_title("Panel B: Residual Time Series (first 120 samples)", fontweight="bold")
    axes[0, 1].legend(fontsize=7, loc="upper right")

    # Panel C: FFT power spectrum for strong seasonal
    if "flood_seasonal_strong" in signals:
        resid = signals["flood_seasonal_strong"]["residuals"]
        fft_vals = np.fft.rfft(resid)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(resid))
        axes[1, 0].semilogy(freqs[1:], power[1:], color="#4C72B0", alpha=0.8)
        axes[1, 0].set_xlabel("Frequency")
        axes[1, 0].set_ylabel("Power (log scale)")
        axes[1, 0].set_title("Panel C: FFT Power Spectrum (strong seasonal)", fontweight="bold")
        # Mark dominant frequency
        dom_idx = np.argmax(power[1:]) + 1
        axes[1, 0].axvline(x=freqs[dom_idx], color="red", linestyle="--",
                          label=f"Dominant period={len(resid)/dom_idx:.0f}")
        axes[1, 0].legend(fontsize=8)

    # Panel D: Cyclic vs Linear encoding comparison
    if "cyclic_encoding_residual" in results and "linear_encoding_residual" in results:
        scenarios = ["cyclic_encoding_residual", "linear_encoding_residual"]
        x = np.arange(len(scenarios))
        kvals = [results[s]["kappa"] for s in scenarios]
        svals = [results[s]["spectral_ratio"] for s in scenarios]
        width = 0.35
        axes[1, 1].bar(x - width/2, kvals, width, label="kappa_periodic", color="#4C72B0")
        axes[1, 1].bar(x + width/2, svals, width, label="SCR", color="#DD8452")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(["Cyclic\nEncoding", "Linear\nEncoding"], fontsize=9)
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].set_title("Panel D: Cyclic vs Linear Encoding", fontweight="bold")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_ylim(0, 1)

    fig.suptitle("E6: Periodic Storm Season -- Does encoding handle Dec-Jan seam?",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E6: Periodic Storm Season Seam")
    parser.add_argument("--output-dir", type=str, default="outputs/E6")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating test signals...")
    signals = generate_test_signals()
    print(f"  {len(signals)} signals generated")

    print("Computing kappa_periodic for all signals...")
    results = compute_all_periodic_kappas(signals)

    # Summary table
    print(f"\n{'Signal':<35} {'kappa':>8} {'SCR':>8} {'Period':>8} {'Expected':>10}")
    print("-" * 72)
    for name in sorted(results.keys(), key=lambda n: results[n]["kappa"], reverse=True):
        r = results[name]
        print(f"{name:<35} {r['kappa']:>8.4f} {r['spectral_ratio']:>8.4f} "
              f"{r['dominant_period']:>8.1f} {'periodic' if r['expected_periodic'] else 'aperiodic':>10}")

    # Check: periodic signals should have higher kappa than aperiodic
    periodic_kappas = [r["kappa"] for r in results.values() if r["expected_periodic"]]
    aperiodic_kappas = [r["kappa"] for r in results.values() if not r["expected_periodic"]]
    print(f"\nPeriodic kappa mean: {np.mean(periodic_kappas):.4f}")
    print(f"Aperiodic kappa mean: {np.mean(aperiodic_kappas):.4f}")
    separation = np.mean(periodic_kappas) > np.mean(aperiodic_kappas)
    print(f"Separation: {'YES' if separation else 'NO'}")

    # Render
    print("\nRendering figure...")
    render_figure(signals, results, output_dir / "fig_E6_periodic_seam_comparison.png")

    # Evidence
    evidence = {
        "experiment": "E6_periodic_seam",
        "n_signals": len(results),
        "periodic_kappa_mean": float(np.mean(periodic_kappas)),
        "aperiodic_kappa_mean": float(np.mean(aperiodic_kappas)),
        "per_signal": {n: {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in r.items()} for n, r in results.items()},
        "pass_criteria": {
            "periodic_gt_aperiodic": separation,
            "strong_seasonal_scr_gt_0.3": results.get("flood_seasonal_strong", {}).get("kappa", 0) > 0.3,
            "noise_control_scr_lt_0.2": results.get("white_noise_control", {}).get("kappa", 1) < 0.2,
        },
    }
    with open(output_dir / "evidence_E6.json", "w") as f:
        json.dump(evidence, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.__class__) and isinstance(o, (np.bool_,)) else float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))

    all_pass = all(evidence["pass_criteria"].values())
    print(f"\n{'='*50}")
    print(f"E6 VERDICT: {'PASS' if all_pass else 'FAIL'}")
    for k, v in evidence["pass_criteria"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"{'='*50}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
