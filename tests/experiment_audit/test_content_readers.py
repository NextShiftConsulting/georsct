"""Tests for content_readers: run results, money table, and aggregate cells."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from georsct.experiment_audit.content_readers import (
    CellMetric,
    MetricStatus,
    extract_aggregate_cells,
    extract_cells_from_runs,
    extract_money_table_cells,
)
from georsct.experiment_audit.models import CellKey

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ---------- extract_cells_from_runs ----------


class TestExtractCellsFromRuns:
    """Tests for per-scenario run result parsing."""

    @pytest.fixture()
    def cells(self) -> list[CellMetric]:
        data = _load("sample_r0.json")
        return extract_cells_from_runs(data)

    def test_finds_three_cells(self, cells: list[CellMetric]) -> None:
        assert len(cells) == 3

    def test_nfip_valid_best_metric(self, cells: list[CellMetric]) -> None:
        nfip = [c for c in cells if c.cell.target == "obs_nfip_event_claims"][0]
        assert nfip.status == MetricStatus.VALID
        assert nfip.best_metric == pytest.approx(0.42)
        assert nfip.solver == "histgbdt"

    def test_nfip_fold_counts(self, cells: list[CellMetric]) -> None:
        nfip = [c for c in cells if c.cell.target == "obs_nfip_event_claims"][0]
        assert nfip.n_folds == 2
        assert nfip.n_null_folds == 0

    def test_311_valid(self, cells: list[CellMetric]) -> None:
        c311 = [c for c in cells if c.cell.target == "obs_has_311"][0]
        assert c311.status == MetricStatus.VALID
        assert c311.best_metric == pytest.approx(0.78)

    def test_hwm_null(self, cells: list[CellMetric]) -> None:
        hwm = [c for c in cells if c.cell.target == "obs_has_hwm"][0]
        assert hwm.status == MetricStatus.NULL
        assert hwm.best_metric is None
        assert hwm.n_null_folds == 1


# ---------- extract_money_table_cells ----------


class TestExtractMoneyTableCells:
    """Tests for money table parsing."""

    @pytest.fixture()
    def cells(self) -> list[CellMetric]:
        data = _load("sample_money_table.json")
        return extract_money_table_cells(data)

    def test_finds_three_cells(self, cells: list[CellMetric]) -> None:
        assert len(cells) == 3

    def test_houston_nfip_takes_r2(self, cells: list[CellMetric]) -> None:
        nfip = [c for c in cells if c.cell.scenario == "houston" and c.cell.target == "obs_nfip_event_claims"][0]
        assert nfip.status == MetricStatus.VALID
        assert nfip.best_metric == pytest.approx(0.55)

    def test_houston_311_takes_r2(self, cells: list[CellMetric]) -> None:
        c311 = [c for c in cells if c.cell.target == "obs_has_311"][0]
        assert c311.best_metric == pytest.approx(0.82)

    def test_nyc_nfip_null(self, cells: list[CellMetric]) -> None:
        nyc = [c for c in cells if c.cell.scenario == "nyc"][0]
        assert nyc.status == MetricStatus.NULL
        assert nyc.best_metric is None


# ---------- extract_aggregate_cells ----------


class TestExtractAggregateCells:
    """Tests for aggregate cell array parsing."""

    @pytest.fixture()
    def cells(self) -> list[CellMetric]:
        data = _load("sample_geometry_kappa.json")
        return extract_aggregate_cells(data, value_field="kappa_geom")

    def test_finds_three_cells(self, cells: list[CellMetric]) -> None:
        assert len(cells) == 3

    def test_houston_nfip_kappa(self, cells: list[CellMetric]) -> None:
        nfip = [c for c in cells if c.cell.scenario == "houston" and c.cell.target == "obs_nfip_event_claims"][0]
        assert nfip.status == MetricStatus.VALID
        assert nfip.best_metric == pytest.approx(0.65)

    def test_houston_311_kappa(self, cells: list[CellMetric]) -> None:
        c311 = [c for c in cells if c.cell.target == "obs_has_311"][0]
        assert c311.best_metric == pytest.approx(0.58)

    def test_nyc_null_kappa(self, cells: list[CellMetric]) -> None:
        nyc = [c for c in cells if c.cell.scenario == "nyc"][0]
        assert nyc.status == MetricStatus.NULL
        assert nyc.best_metric is None
