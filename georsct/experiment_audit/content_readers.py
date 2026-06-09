"""Content readers that parse result JSONs to extract per-cell metric status."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from .models import CellKey


class MetricStatus(str, Enum):
    """Status of a metric value within a cell."""

    VALID = "VALID"
    NULL = "NULL"
    DEGENERATE = "DEGENERATE"


@dataclass
class CellMetric:
    """A single cell's metric extraction result."""

    cell: CellKey
    best_metric: float | None
    status: MetricStatus
    solver: str | None = None
    n_folds: int = 0
    n_null_folds: int = 0


def _pick_metric(run: dict) -> float | None:
    """Extract the primary metric from a run's metrics dict.

    Uses r2 for regression tasks and roc_auc for classification/binary tasks.
    """
    metrics = run.get("metrics", {})
    task = run.get("task", "")
    if task == "regression":
        return metrics.get("r2")
    return metrics.get("roc_auc")


def extract_cells_from_runs(data: dict) -> list[CellMetric]:
    """Extract per-cell metrics from a per-scenario run results JSON.

    Args:
        data: Parsed JSON with a ``runs`` array. Each run has scenario,
              target, task, solver, fold, and metrics fields.

    Returns:
        One CellMetric per unique (scenario, target) group.
    """
    groups: dict[CellKey, list[dict]] = defaultdict(list)
    for run in data.get("runs", []):
        key = CellKey(scenario=run["scenario"], target=run["target"])
        groups[key].append(run)

    results: list[CellMetric] = []
    for cell_key, runs in groups.items():
        best_metric: float | None = None
        best_solver: str | None = None
        n_folds = len(runs)
        n_null_folds = 0

        for run in runs:
            val = _pick_metric(run)
            if val is None:
                n_null_folds += 1
                continue
            if best_metric is None or val > best_metric:
                best_metric = val
                best_solver = run.get("solver")

        status = MetricStatus.VALID if best_metric is not None else MetricStatus.NULL
        results.append(
            CellMetric(
                cell=cell_key,
                best_metric=best_metric,
                status=status,
                solver=best_solver,
                n_folds=n_folds,
                n_null_folds=n_null_folds,
            )
        )
    return results


def extract_money_table_cells(data: dict) -> list[CellMetric]:
    """Extract per-cell metrics from a money table JSON.

    Checks r2_metric, r1_metric, r0_metric in descending order and takes
    the highest-level non-null value.

    Args:
        data: Parsed JSON with a ``cells`` array.

    Returns:
        One CellMetric per cell entry.
    """
    results: list[CellMetric] = []
    for cell in data.get("cells", []):
        cell_key = CellKey(scenario=cell["scenario"], target=cell["target"])
        best: float | None = None
        for field in ("r2_metric", "r1_metric", "r0_metric"):
            val = cell.get(field)
            if val is not None:
                best = val
                break

        status = MetricStatus.VALID if best is not None else MetricStatus.NULL
        results.append(
            CellMetric(cell=cell_key, best_metric=best, status=status)
        )
    return results


def extract_aggregate_cells(data: dict, value_field: str) -> list[CellMetric]:
    """Extract per-cell metrics from an aggregate cell array JSON.

    Args:
        data: Parsed JSON with a ``cells`` array.
        value_field: Name of the field to extract (e.g. ``kappa_geom``).

    Returns:
        One CellMetric per cell entry.
    """
    results: list[CellMetric] = []
    for cell in data.get("cells", []):
        cell_key = CellKey(scenario=cell["scenario"], target=cell["target"])
        val = cell.get(value_field)
        status = MetricStatus.VALID if val is not None else MetricStatus.NULL
        results.append(
            CellMetric(cell=cell_key, best_metric=val, status=status)
        )
    return results
