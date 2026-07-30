"""Adapter: parquet-backed ConstructDataSource.

Implements the ``ConstructDataSource`` port (georsct.ports.construct_data_source)
by loading per-construct target columns from one parquet table per scenario.

Each scenario table is keyed by an id column (default ``zcta_id``); one row
per geographic unit.  A construct's target lives in a single column of that
table (see ``CONSTRUCT_TARGET_COLUMNS`` in the domain).  A construct is
*available* when its target column exists and carries at least ``min_finite``
finite values; otherwise ``ConstructData.available`` is False with a reason,
per the ADR-020 D8 discipline (missing = explicit reason, never a silent zero).

Alignment contract
------------------
The application use case (``compute_five_construct_divergence``) supplies a
single shared feature matrix and reads only ``ConstructData.target_values``
back for each construct.  Those target vectors MUST be row-aligned with the
shared features.  This adapter sorts each scenario deterministically by the id
column and exposes that ordering via :meth:`canonical_order`; the composition
job builds its features/folds/coords in the same order, so every construct
target and the shared features share one row index.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from rsct_spatial.geometry import CentroidSource
from rsct_spatial.quality import recover_coordinates, require_finite

from georsct.domain.construct_certificate import (
    CONSTRUCT_TARGET_COLUMNS,
    ConstructLabel,
)
from georsct.ports.construct_data_source import ConstructData, ConstructDataSource

log = logging.getLogger(__name__)


class ParquetConstructDataSource(ConstructDataSource):
    """ConstructDataSource backed by one parquet table per scenario.

    Args:
        scenario_paths: Map of scenario_id -> parquet path holding that
            scenario's per-ZCTA construct targets.
        id_column: Column that uniquely identifies each geographic unit.
            Used to establish a deterministic canonical row order.
        min_finite: Minimum finite target values for a construct to be
            reported available. Matches the use case's n_finite floor (30).
        target_columns: Optional override of the construct -> column mapping.
            Defaults to the canonical CONSTRUCT_TARGET_COLUMNS.
        require_finite: Columns that must hold a finite value for a row to
            be retained (e.g. coordinate columns). Rows still failing after
            recovery are dropped once, before the canonical order is fixed,
            so targets/features/coordinates stay aligned. Drops are recorded
            in :meth:`cleanup_report` (no silent truncation).
        centroid_source: Optional authoritative CentroidSource. When set,
            rows with missing coordinates are recovered from it before the
            require_finite drop -- a missing centroid for a known unit is a
            broken join, not missing data.
        lat_column, lon_column: Coordinate columns for recovery.
        event_column, event_id: If a scenario table is replicated per event
            (multiple rows per region), set both to slice it to one event so
            the analysis is one row per region. If duplicate region ids
            survive load, the adapter raises rather than silently collapsing
            them -- picking an event/aggregation is the caller's decision.

    The recovery / require_finite / coverage steps are the central
    ``rsct_spatial.quality`` process -- this adapter orchestrates them, it
    does not reimplement them.
    """

    def __init__(
        self,
        scenario_paths: Mapping[str, str | Path],
        id_column: str = "zcta_id",
        min_finite: int = 30,
        target_columns: Optional[Mapping[ConstructLabel, str]] = None,
        require_finite: Optional[Sequence[str]] = None,
        centroid_source: Optional[CentroidSource] = None,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
        event_column: Optional[str] = None,
        event_id: Optional[str] = None,
    ):
        self._scenario_paths = {k: Path(v) for k, v in scenario_paths.items()}
        self._id_column = id_column
        self._min_finite = int(min_finite)
        self._target_columns = dict(target_columns or CONSTRUCT_TARGET_COLUMNS)
        self._require_finite = tuple(require_finite or ())
        self._centroid_source = centroid_source
        self._lat_column = lat_column
        self._lon_column = lon_column
        self._event_column = event_column
        self._event_id = event_id
        self._cache: dict[str, pd.DataFrame] = {}
        self._cleanup: dict[str, dict] = {}

    # -- internal ---------------------------------------------------------

    def _load_scenario(self, scenario_id: str) -> pd.DataFrame:
        """Load and cache a scenario table, sorted by the id column."""
        if scenario_id in self._cache:
            return self._cache[scenario_id]

        path = self._scenario_paths.get(scenario_id)
        if path is None:
            raise KeyError(
                f"No parquet registered for scenario '{scenario_id}'. "
                f"Known scenarios: {sorted(self._scenario_paths)}"
            )
        if not path.exists():
            raise FileNotFoundError(f"Scenario table not found: {path}")

        df = pd.read_parquet(path)
        if self._id_column not in df.columns:
            raise ValueError(
                f"Scenario '{scenario_id}' table missing id column "
                f"'{self._id_column}' (columns: {list(df.columns)[:12]}...)"
            )

        report: dict = {}

        # 1. Slice to one event if the table is replicated per event, so the
        #    analysis is one row per region (never a silent dedup).
        if self._event_column and self._event_column in df.columns and self._event_id is not None:
            n_before = len(df)
            df = df[df[self._event_column].astype(str) == str(self._event_id)]
            report["event_filter"] = {
                "column": self._event_column,
                "event_id": self._event_id,
                "n_before": n_before,
                "n_after": len(df),
            }

        # 2. Recover missing coordinates from an authoritative source before
        #    dropping -- a missing centroid for a known unit is a broken join.
        if self._centroid_source is not None and {self._lat_column, self._lon_column} <= set(df.columns):
            rec = recover_coordinates(
                df, self._centroid_source,
                id_column=self._id_column,
                lat_column=self._lat_column, lon_column=self._lon_column,
            )
            df = rec.frame
            report["coordinate_recovery"] = {
                "n_missing_before": rec.n_missing_before,
                "n_recovered": rec.n_recovered,
                "n_unrecovered": rec.n_unrecovered,
                "recovered_ids": list(rec.recovered_ids),
                "unrecovered_ids": list(rec.unrecovered_ids),
            }

        # 3. Drop rows still lacking a required finite value (e.g. an
        #    unrecoverable centroid) so the retained set is usable and aligned.
        if self._require_finite:
            missing = [c for c in self._require_finite if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Scenario '{scenario_id}' missing require_finite column(s): {missing}"
                )
            rf = require_finite(df, list(self._require_finite), id_column=self._id_column)
            df = rf.frame
            report["require_finite"] = {
                "columns": list(self._require_finite),
                "n_dropped": rf.n_dropped,
                "dropped_ids": list(rf.dropped_ids),
            }
            if rf.n_dropped:
                log.warning(
                    "scenario '%s': dropped %d rows lacking finite %s (ids=%s)",
                    scenario_id, rf.n_dropped, list(self._require_finite), list(rf.dropped_ids),
                )

        # Fail loud on duplicate regions: the use case indexes targets by
        # region id, so duplicates would silently collapse. Force the caller
        # to pick an event (event_id) or pre-aggregate rather than guess.
        n_dup = int(df[self._id_column].duplicated().sum())
        if n_dup:
            dup_ids = sorted(df.loc[df[self._id_column].duplicated(keep=False), self._id_column].astype(str).unique())
            raise ValueError(
                f"scenario '{scenario_id}': {n_dup} duplicate '{self._id_column}' rows "
                f"({len(dup_ids)} ids, e.g. {dup_ids[:5]}). The table is likely "
                f"replicated per event -- pass event_column/event_id to slice one "
                f"event, or pre-aggregate to one row per region."
            )

        # Deterministic canonical order for alignment with shared features.
        df = df.sort_values(self._id_column).reset_index(drop=True)
        self._cache[scenario_id] = df
        self._cleanup[scenario_id] = report
        return df

    # -- port surface -----------------------------------------------------

    def load_construct_target(
        self,
        construct: ConstructLabel,
        scenario_id: str,
        event_id: Optional[str] = None,
    ) -> ConstructData:
        """Load target values for one construct in one scenario.

        ``event_id`` is accepted for port conformance; the static
        construct tables in this adapter are event-independent, so it is
        recorded in the unavailability reason only when relevant.
        """
        df = self._load_scenario(scenario_id)

        target_col = self._target_columns.get(construct)
        if target_col is None:
            return ConstructData(
                construct=construct,
                target_values=None,
                region_ids=None,
                available=False,
                reason=f"no target column mapped for construct {construct.value}",
            )

        if target_col not in df.columns:
            return ConstructData(
                construct=construct,
                target_values=None,
                region_ids=None,
                available=False,
                reason=f"column '{target_col}' absent in scenario '{scenario_id}'",
            )

        values = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        region_ids = df[self._id_column].astype(str).to_numpy()

        n_finite = int(np.isfinite(values).sum())
        if n_finite < self._min_finite:
            return ConstructData(
                construct=construct,
                target_values=None,
                region_ids=None,
                available=False,
                reason=(
                    f"column '{target_col}' has {n_finite} finite values "
                    f"(need {self._min_finite})"
                ),
            )

        return ConstructData(
            construct=construct,
            target_values=values,
            region_ids=region_ids,
            available=True,
            reason="ok",
        )

    def available_constructs(self, scenario_id: str) -> list[ConstructLabel]:
        """List constructs whose targets are loadable for a scenario."""
        return [
            c
            for c in self._target_columns
            if self.load_construct_target(c, scenario_id).available
        ]

    # -- composition helper (not part of the port ABC) -------------------

    def canonical_order(self, scenario_id: str) -> np.ndarray:
        """Return the canonical id ordering for a scenario.

        The composition root builds its shared feature matrix, folds, and
        coordinates in this exact order so that every construct target
        returned by :meth:`load_construct_target` is row-aligned with them.
        """
        df = self._load_scenario(scenario_id)
        return df[self._id_column].astype(str).to_numpy()

    def scenario_frame(self, scenario_id: str) -> pd.DataFrame:
        """Return the (canonically ordered) scenario DataFrame.

        Exposed so the composition root reads features from the same
        table and ordering the targets are drawn from -- one load, one
        sort, guaranteed alignment.
        """
        return self._load_scenario(scenario_id)

    def cleanup_report(self, scenario_id: str) -> dict:
        """Return the dedup / recovery / drop accounting for a scenario.

        The composition root writes this into the output's data-quality
        block so every exclusion is auditable.
        """
        self._load_scenario(scenario_id)
        return self._cleanup.get(scenario_id, {})
