#!/usr/bin/env python3
"""
task_residual_floor_estimator.py — Estimate per-task TRF from ceiling model ensemble.

Task Residual Floor (TRF, formerly N_ceiling) is the irreducible residual
variance per task, estimated as:

    TRF = 1 − max_k(E[R²_k])

where k indexes architecturally diverse ceiling models. This is the paper
quantity — invariant across model versions, characterises the task, not the model.

Confidence intervals via state-level block bootstrap (1000 resamples) to
respect spatial autocorrelation in ZCTA-level residuals.

Diagnostics:
  - Pairwise residual correlation matrix across models, per task
  - Tasks flagged when mean pairwise |ρ| > RESIDUAL_CORRELATION_FLAG_THRESHOLD
  - Rank correlation: per-task best R² vs mean residual correlation
    (high correlation → architectural diversity is insufficient)

Usage:
    from apps.geo_cert.models.ceiling.task_residual_floor_estimator import TRFEstimator

    estimator = TRFEstimator.from_oof_dir("C:/tmp/oof_predictions/")
    results = estimator.estimate()
    estimator.print_report(results)

CLI:
    python task_residual_floor_estimator.py --oof-dir C:/tmp/oof_predictions/
    python task_residual_floor_estimator.py --oof-dir C:/tmp/oof_predictions/ --json
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from apps.geo_cert.models.ceiling.ceiling_schema import (
    RESIDUAL_CORRELATION_FLAG_THRESHOLD,
    REQUIRED_COLUMNS,
    validate,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TaskCeilingResult:
    """Task residual floor estimate for a single task."""
    task: str
    n_models: int
    best_r2: float
    best_model: str
    task_residual_floor: float
    ci_lower: float
    ci_upper: float
    per_model_r2: dict                          # {model_version: R²}
    mean_residual_correlation: float
    flagged: bool                                # mean |ρ| > threshold
    residual_correlation_matrix: dict            # {(m_i, m_j): ρ}


@dataclass
class CeilingEstimateReport:
    """Full task residual floor estimation report."""
    tasks: list                                  # list[TaskCeilingResult]
    rank_correlation_rho: float                  # Spearman ρ(best_R², mean_|ρ|)
    rank_correlation_p: float
    n_flagged: int
    n_tasks: int
    n_models: int
    model_versions: list
    bootstrap_resamples: int = N_BOOTSTRAP
    ci_level: float = CI_LEVEL
    correlation_flag_threshold: float = RESIDUAL_CORRELATION_FLAG_THRESHOLD


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

class TRFEstimator:
    """Estimate per-task task residual floor from an ensemble of OOF predictions."""

    def __init__(self, oof_df: pd.DataFrame):
        """Initialize from a validated OOF DataFrame.

        Args:
            oof_df: Combined OOF predictions from all ceiling models.
                    Must conform to ceiling_schema.
        """
        errors = validate(oof_df, strict=False)
        if errors:
            raise ValueError(
                f"OOF data fails schema validation: {errors}"
            )
        self.df = oof_df
        self.tasks = sorted(oof_df["task"].unique())
        self.model_versions = sorted(oof_df["model_version"].unique())
        log.info(
            f"TRFEstimator: {len(self.tasks)} tasks, "
            f"{len(self.model_versions)} models, "
            f"{len(oof_df)} rows"
        )

    @classmethod
    def from_oof_dir(cls, oof_dir: str) -> "TRFEstimator":
        """Load all OOF parquet files from a directory.

        Each file should be a parquet with columns matching ceiling_schema.
        Files are concatenated; model_version column distinguishes them.
        """
        p = Path(oof_dir)
        files = sorted(p.glob("oof_*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No oof_*.parquet files found in {oof_dir}"
            )

        frames = []
        for f in files:
            df = pd.read_parquet(f)
            # Enforce string dtypes for key columns
            for col in ("zcta", "task", "fold", "model_version", "state_fips"):
                if col in df.columns:
                    df[col] = df[col].astype(str)
            frames.append(df)
            log.info(f"  Loaded {f.name}: {len(df)} rows")

        combined = pd.concat(frames, ignore_index=True)
        log.info(f"Combined: {len(combined)} rows")
        return cls(combined)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TRFEstimator":
        """Create from a pre-built DataFrame."""
        return cls(df)

    # ------------------------------------------------------------------
    # Block bootstrap (state-level)
    # ------------------------------------------------------------------

    @staticmethod
    def _block_bootstrap_r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        state_labels: np.ndarray,
        n_boot: int = N_BOOTSTRAP,
        seed: int = BOOTSTRAP_SEED,
    ) -> np.ndarray:
        """Compute R² bootstrap distribution using state-level blocks.

        Resamples states (not individual ZCTAs) to respect spatial
        autocorrelation within states.

        Args:
            y_true: (n,) true values
            y_pred: (n,) predicted values
            state_labels: (n,) state identifiers (FIPS or abbreviation)
            n_boot: number of bootstrap resamples
            seed: random seed

        Returns:
            (n_boot,) array of bootstrap R² values
        """
        rng = np.random.RandomState(seed)
        unique_states = np.unique(state_labels)
        n_states = len(unique_states)

        # Pre-build index arrays per state for speed
        state_indices = {
            s: np.where(state_labels == s)[0] for s in unique_states
        }

        r2_boot = np.empty(n_boot)
        for b in range(n_boot):
            # Resample states with replacement
            boot_states = rng.choice(unique_states, size=n_states, replace=True)
            # Gather all ZCTAs from resampled states
            idx = np.concatenate([state_indices[s] for s in boot_states])
            if len(idx) < 10:
                r2_boot[b] = np.nan
                continue
            yt = y_true[idx]
            yp = y_pred[idx]
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2_boot[b] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return r2_boot

    # ------------------------------------------------------------------
    # Per-task estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _base_model(version: str) -> str:
        """Strip seed suffix to get base model name.

        "gnn_v1_seed0" → "gnn_v1", "pca_v1" → "pca_v1"
        """
        import re
        return re.sub(r"_seed\d+$", "", version)

    @staticmethod
    def _collapse_seed_variants(
        per_version_r2: dict,
        per_version_boot: dict,
        residuals_by_version: dict,
    ) -> tuple:
        """Collapse seed variants into seed-averaged base models.

        Stochastic models (GNN, MLP) emit multiple seeds. The ceiling
        estimate uses E[R²] across seeds, not the best seed (which
        would inflate the bound in the wrong direction).

        Returns:
            (collapsed_r2, collapsed_boot, collapsed_residuals, collapsed_models)
        """
        import re

        # Group versions by base model
        base_groups = {}
        for v in per_version_r2:
            base = re.sub(r"_seed\d+$", "", v)
            base_groups.setdefault(base, []).append(v)

        collapsed_r2 = {}
        collapsed_boot = {}
        collapsed_residuals = {}

        for base, variants in base_groups.items():
            if len(variants) == 1:
                # Deterministic model — pass through
                v = variants[0]
                collapsed_r2[base] = per_version_r2[v]
                collapsed_boot[base] = per_version_boot[v]
                collapsed_residuals[base] = residuals_by_version[v]
            else:
                # Seed-averaged: E[R²] across seeds
                r2s = [per_version_r2[v] for v in variants]
                collapsed_r2[base] = round(float(np.mean(r2s)), 6)

                # Average bootstrap distributions element-wise
                boots = np.column_stack([per_version_boot[v] for v in variants])
                collapsed_boot[base] = np.nanmean(boots, axis=1)

                # Average residuals across seeds per ZCTA
                all_zctas = set()
                for v in variants:
                    all_zctas.update(residuals_by_version[v].keys())
                avg_resid = {}
                for z in all_zctas:
                    vals = [
                        residuals_by_version[v][z]
                        for v in variants
                        if z in residuals_by_version[v]
                    ]
                    avg_resid[z] = float(np.mean(vals))
                collapsed_residuals[base] = avg_resid

        collapsed_models = sorted(collapsed_r2.keys())
        return collapsed_r2, collapsed_boot, collapsed_residuals, collapsed_models

    def _estimate_task(self, task: str) -> TaskCeilingResult:
        """Estimate task residual floor for a single task."""
        task_df = self.df[self.df["task"] == task]
        raw_versions = sorted(task_df["model_version"].unique())

        # Phase 1: compute per-version R², bootstrap, residuals
        per_version_r2 = {}
        per_version_boot = {}
        residuals_by_version = {}

        for mv in raw_versions:
            mv_df = task_df[task_df["model_version"] == mv]
            y_true = mv_df["y_true"].values
            y_pred = mv_df["y_pred"].values
            states = mv_df["state_fips"].values

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            per_version_r2[mv] = round(r2, 6)

            boot = self._block_bootstrap_r2(y_true, y_pred, states)
            per_version_boot[mv] = boot

            residuals_by_version[mv] = dict(
                zip(mv_df["zcta"].values, mv_df["residual"].values)
            )

        # Phase 2: collapse seed variants → base models with E[R²]
        per_model_r2, per_model_boot, residuals_by_model, models = (
            self._collapse_seed_variants(
                per_version_r2, per_version_boot, residuals_by_version
            )
        )

        # Best model and TRF point estimate
        best_model = max(per_model_r2, key=per_model_r2.get)
        best_r2 = per_model_r2[best_model]
        trf = 1.0 - best_r2

        # Bootstrap CI for TRF = 1 - max_k(E[R²_k])
        boot_matrix = np.column_stack(
            [per_model_boot[mv] for mv in models]
        )
        boot_max_r2 = np.nanmax(boot_matrix, axis=1)
        boot_trf = 1.0 - boot_max_r2

        alpha = 1.0 - CI_LEVEL
        ci_lower = float(np.nanpercentile(boot_trf, 100 * alpha / 2))
        ci_upper = float(np.nanpercentile(boot_trf, 100 * (1 - alpha / 2)))

        # Residual correlation matrix (on collapsed/base models)
        corr_matrix = {}
        mean_abs_corr = 0.0
        n_pairs = 0

        if len(models) >= 2:
            common_zctas = set.intersection(
                *[set(residuals_by_model[mv].keys()) for mv in models]
            )
            common_zctas = sorted(common_zctas)

            if len(common_zctas) > 0:
                resid_matrix = np.column_stack([
                    np.array([residuals_by_model[mv][z] for z in common_zctas])
                    for mv in models
                ])

                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        rho, _ = sp_stats.pearsonr(
                            resid_matrix[:, i], resid_matrix[:, j]
                        )
                        key = f"{models[i]}|{models[j]}"
                        corr_matrix[key] = round(rho, 4)
                        mean_abs_corr += abs(rho)
                        n_pairs += 1

            if n_pairs > 0:
                mean_abs_corr /= n_pairs

        flagged = mean_abs_corr > RESIDUAL_CORRELATION_FLAG_THRESHOLD

        return TaskCeilingResult(
            task=task,
            n_models=len(models),
            best_r2=round(best_r2, 6),
            best_model=best_model,
            task_residual_floor=round(trf, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            per_model_r2=per_model_r2,
            mean_residual_correlation=round(mean_abs_corr, 4),
            flagged=flagged,
            residual_correlation_matrix=corr_matrix,
        )

    # ------------------------------------------------------------------
    # Full estimation
    # ------------------------------------------------------------------

    def estimate(self) -> CeilingEstimateReport:
        """Run task residual floor estimation for all tasks.

        Returns:
            CeilingEstimateReport with per-task results and diagnostics.
        """
        results = []
        for task in self.tasks:
            log.info(f"Estimating TRF for {task}...")
            result = self._estimate_task(task)
            results.append(result)
            flag_str = " *** FLAGGED" if result.flagged else ""
            log.info(
                f"  {task}: TRF={result.task_residual_floor:.4f} "
                f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}] "
                f"best={result.best_model} R2={result.best_r2:.4f} "
                f"mean_|rho|={result.mean_residual_correlation:.3f}{flag_str}"
            )

        # Rank correlation diagnostic: does best R² track mean |ρ|?
        best_r2s = [r.best_r2 for r in results]
        mean_corrs = [r.mean_residual_correlation for r in results]
        if len(results) >= 3:
            rank_rho, rank_p = sp_stats.spearmanr(best_r2s, mean_corrs)
        else:
            rank_rho, rank_p = float("nan"), float("nan")

        n_flagged = sum(1 for r in results if r.flagged)

        report = CeilingEstimateReport(
            tasks=results,
            rank_correlation_rho=round(rank_rho, 4)
            if not np.isnan(rank_rho) else None,
            rank_correlation_p=round(rank_p, 4)
            if not np.isnan(rank_p) else None,
            n_flagged=n_flagged,
            n_tasks=len(results),
            n_models=len(self.model_versions),
            model_versions=self.model_versions,
        )

        log.info(f"\nRank correlation (best R² vs mean |ρ|): "
                 f"ρ={report.rank_correlation_rho}, p={report.rank_correlation_p}")
        if n_flagged > 0:
            log.warning(
                f"{n_flagged}/{len(results)} tasks flagged for high "
                f"residual correlation (threshold={RESIDUAL_CORRELATION_FLAG_THRESHOLD})"
            )

        return report

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: CeilingEstimateReport) -> None:
        """Print formatted task residual floor report to stdout."""
        print("\n" + "=" * 90)
        print("TASK RESIDUAL FLOOR (TRF) ESTIMATION REPORT")
        print("=" * 90)
        print(f"Models: {report.n_models} ({', '.join(report.model_versions)})")
        print(f"Tasks:  {report.n_tasks}")
        print(f"Bootstrap: {report.bootstrap_resamples} resamples, "
              f"{report.ci_level*100:.0f}% CI (state-level blocks)")
        print(f"Correlation flag threshold: {report.correlation_flag_threshold}")
        print()

        hdr = (f"{'Task':<30} {'Best R2':>8} {'Best Model':<16} "
               f"{'TRF':>8} {'CI_lo':>8} {'CI_hi':>8} "
               f"{'mean|rho|':>9} {'Flag':>5}")
        print(hdr)
        print("-" * len(hdr))

        for r in sorted(report.tasks, key=lambda x: x.task_residual_floor):
            flag = "***" if r.flagged else ""
            print(
                f"{r.task:<30} {r.best_r2:>8.4f} {r.best_model:<16} "
                f"{r.task_residual_floor:>8.4f} {r.ci_lower:>8.4f} {r.ci_upper:>8.4f} "
                f"{r.mean_residual_correlation:>9.3f} {flag:>5}"
            )

        print("-" * len(hdr))

        # Summary stats
        trf_vals = [r.task_residual_floor for r in report.tasks]
        print(f"\nTRF summary: "
              f"mean={np.mean(trf_vals):.4f}, "
              f"median={np.median(trf_vals):.4f}, "
              f"min={np.min(trf_vals):.4f}, "
              f"max={np.max(trf_vals):.4f}")

        print(f"\nRank correlation diagnostic (best R² vs mean |ρ|):")
        print(f"  Spearman ρ = {report.rank_correlation_rho}, "
              f"p = {report.rank_correlation_p}")
        if report.rank_correlation_rho is not None and report.rank_correlation_rho > 0.5:
            print("  WARNING: High rank correlation suggests models with "
                  "higher R² also have more correlated residuals.")
            print("  TRF estimates for those tasks may be optimistic. "
                  "Consider adding architecturally diverse models.")

        if report.n_flagged > 0:
            print(f"\n  {report.n_flagged} task(s) flagged for high mean "
                  f"residual correlation (>{report.correlation_flag_threshold}):")
            for r in report.tasks:
                if r.flagged:
                    print(f"    - {r.task}: mean|ρ|={r.mean_residual_correlation:.3f}")

        print("=" * 90)

    @staticmethod
    def to_json(report: CeilingEstimateReport) -> str:
        """Serialize report to JSON."""
        d = {
            "n_tasks": report.n_tasks,
            "n_models": report.n_models,
            "model_versions": report.model_versions,
            "bootstrap_resamples": report.bootstrap_resamples,
            "ci_level": report.ci_level,
            "correlation_flag_threshold": report.correlation_flag_threshold,
            "rank_correlation_rho": report.rank_correlation_rho,
            "rank_correlation_p": report.rank_correlation_p,
            "n_flagged": report.n_flagged,
            "tasks": [asdict(t) for t in report.tasks],
        }
        return json.dumps(d, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-task task residual floor (TRF) from ceiling model OOF predictions."
    )
    parser.add_argument(
        "--oof-dir", required=True,
        help="Directory containing oof_*.parquet files"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of formatted report"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write report to file (default: stdout)"
    )
    args = parser.parse_args()

    estimator = TRFEstimator.from_oof_dir(args.oof_dir)
    report = estimator.estimate()

    if args.json:
        output = TRFEstimator.to_json(report)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            log.info(f"JSON report written to {args.output}")
        else:
            print(output)
    else:
        TRFEstimator.print_report(report)
        if args.output:
            Path(args.output).write_text(
                TRFEstimator.to_json(report), encoding="utf-8"
            )
            log.info(f"JSON report also written to {args.output}")


if __name__ == "__main__":
    main()
