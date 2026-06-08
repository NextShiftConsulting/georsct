"""Deployment-domain definitions for deployment-aligned validation.

Why this module exists
----------------------
Deployment-aligned validation (TWCV; see ``deployment_alignment``) is only
meaningful if the *deployment task distribution* is defined independently of the
*observed sample*. The observed sample is preferentially sampled toward certain
locations. The deployment domain is the set of units for which we will actually
issue guidance, defined by physical eligibility rather than by where labels
happen to exist. The gap between the two is exactly the covariate shift the
certificate must measure. Setting the deployment target equal to the sample
assumes that shift away and yields misleadingly reassuring certificates.

Each regime therefore declares an eligibility rule over the full candidate
universe. Rules are declarative and fail loudly via ``MissingHazardLayer``
when a required layer is absent -- there is no silent fallback to "all rows",
which would reintroduce the circularity.

Domain-specific registries (e.g. flood regimes) belong in application packages
(e.g. ``georsct-flood``), not here. This module provides only the framework
classes: ``RegimeDomain``, ``EligibilityRule``, and the exception types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


class MissingHazardLayer(RuntimeError):
    """Raised when a regime's deployment domain cannot be materialized because a
    required layer column is absent.

    Intentionally a hard error: a regime whose layer is not yet ingested must
    not silently fall back to the full universe or to the observed sample;
    either would corrupt the deployment-alignment certificate.
    """


class EmptyDeploymentDomain(RuntimeError):
    """Raised when every required layer is present but no row qualifies.

    Distinct from ``MissingHazardLayer``: the data are there, but the rules
    (or their thresholds) selected nothing. Usually a threshold-calibration or
    universe-scoping problem, not a missing-ingest problem.
    """


_VALID_COMBINE = {"and", "or"}
_NUMERIC_OPS = {"ge", "le", "gt", "lt"}
_TRUE_TOKENS = {"y", "yes", "true", "t", "1"}
_FALSE_TOKENS = {"n", "no", "false", "f", "0", "", "nan", "none"}


def _as_bool(s: pd.Series) -> pd.Series:
    """Coerce a column to boolean without the ``astype(bool)`` string footgun.

    bool dtype -> NaN treated False. Numeric -> nonzero is True. Object/string ->
    a recognized true/false token set; an unrecognized token raises rather than
    silently counting as True (fail closed, consistent with the module).
    """

    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0) != 0
    norm = s.astype(str).str.strip().str.lower()
    mapped = norm.map(
        lambda v: True if v in _TRUE_TOKENS else (False if v in _FALSE_TOKENS else None)
    )
    if mapped.isna().any():
        bad = sorted(set(norm[mapped.isna()]))[:5]
        raise ValueError(
            f"'truthy' column contains unrecognized boolean tokens {bad}; "
            f"store the layer as bool / 0-1 / standard y-n-true-false strings"
        )
    return mapped.astype(bool)


_OPS: dict[str, Callable[[pd.Series, float | None], pd.Series]] = {
    "ge": lambda s, t: s >= t,
    "le": lambda s, t: s <= t,
    "gt": lambda s, t: s > t,
    "lt": lambda s, t: s < t,
    "eq": lambda s, t: s == t,
    "notnull": lambda s, t: s.notna(),
    "truthy": lambda s, t: _as_bool(s),
}


@dataclass(frozen=True)
class EligibilityRule:
    """A single column-level eligibility predicate."""

    column: str
    op: str
    threshold: float | None = None
    layer: str = ""   # human-readable name of the data layer this rule needs
    units: str = ""   # units the threshold is expressed in (e.g. 'm')

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"unknown op '{self.op}'; valid: {sorted(_OPS)}")
        if self.op in _NUMERIC_OPS and self.threshold is None:
            raise ValueError(f"op '{self.op}' on column '{self.column}' requires a threshold")

    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            raise MissingHazardLayer(
                f"column '{self.column}' (layer: {self.layer or 'unspecified'}) "
                f"is required to resolve this deployment domain but is absent"
            )
        if self.op in _NUMERIC_OPS:
            series = pd.to_numeric(df[self.column], errors="coerce")
        else:
            series = df[self.column]
        return _OPS[self.op](series, self.threshold).reindex(df.index).fillna(False).astype(bool)


@dataclass(frozen=True)
class RegimeDomain:
    """Declarative deployment-domain spec for one regime.

    Application packages define concrete instances (e.g. flood regimes).
    The framework provides the resolution and audit machinery.
    """

    regime_id: str
    name: str
    hazard_mechanism: str
    label_source: str
    rules: tuple[EligibilityRule, ...] = ()
    combine: str = "and"            # "and" | "or"
    full_universe: bool = False     # True => every row in the universe is eligible
    note: str = ""

    def __post_init__(self) -> None:
        if self.combine not in _VALID_COMBINE:
            raise ValueError(f"combine must be one of {_VALID_COMBINE}, got '{self.combine}'")
        if self.full_universe and self.rules:
            raise ValueError(
                f"regime '{self.regime_id}': full_universe=True must not also carry "
                f"rules (they would be silently ignored)"
            )
        if not self.full_universe and not self.rules:
            raise ValueError(
                f"regime '{self.regime_id}': must set full_universe=True or supply rules"
            )

    @property
    def required_layers(self) -> list[str]:
        return sorted({r.layer for r in self.rules if r.layer})

    def resolve_mask(self, universe: pd.DataFrame) -> pd.Series:
        """Boolean Series over ``universe`` selecting eligible rows."""

        if self.full_universe:
            return pd.Series(True, index=universe.index)
        masks = [r.mask(universe) for r in self.rules]
        combined = masks[0]
        for m in masks[1:]:
            combined = (combined & m) if self.combine == "and" else (combined | m)
        return combined.fillna(False)

    def mask_report(self, universe: pd.DataFrame) -> dict:
        """Per-rule eligibility / NaN-drop counts for logging and audit trails."""

        report: dict = {"regime_id": self.regime_id, "n_universe": int(len(universe))}
        if self.full_universe:
            report["n_eligible"] = int(len(universe))
            report["rules"] = []
            return report
        per_rule = []
        for r in self.rules:
            if r.column not in universe.columns:
                per_rule.append({"column": r.column, "status": "MISSING_LAYER", "layer": r.layer})
                continue
            n_nan = int(pd.to_numeric(universe[r.column], errors="coerce").isna().sum()) \
                if r.op in _NUMERIC_OPS else int(universe[r.column].isna().sum())
            per_rule.append({
                "column": r.column, "op": r.op, "threshold": r.threshold, "units": r.units,
                "n_eligible": int(r.mask(universe).sum()), "n_nan_dropped": n_nan,
            })
        report["rules"] = per_rule
        report["n_eligible"] = int(self.resolve_mask(universe).sum())
        return report

    def resolve(self, universe: pd.DataFrame) -> pd.DataFrame:
        """Return the eligible deployment subset of ``universe``."""

        mask = self.resolve_mask(universe)
        eligible = universe.loc[mask].copy()
        if eligible.empty:
            raise EmptyDeploymentDomain(
                f"regime '{self.regime_id}': all required layers present but no "
                f"row qualified; check thresholds and universe scope"
            )
        return eligible
