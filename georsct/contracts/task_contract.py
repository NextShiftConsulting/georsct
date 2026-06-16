"""Public task contract for GeoRSCT-X benchmark execution.

The task contract is what the harness sees at inference time.
Ground-truth values live in TaskGold (separate module) and are
only visible to the scorer, never to the execution layer.

Design: TerraBench-style structured output with tolerance schema,
adapted for geospatial certificate-governed workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NumericField:
    """Output field specification with tolerance contract.

    TerraBench tolerance model:
        hit iff |y_hat - y| <= max(abs_tol, rel_tol * |y|, floor_scale)

    The gold value is NOT stored here -- it lives in TaskGold.
    """

    key: str
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    floor_scale: float = 0.0


@dataclass(frozen=True)
class TaskContract:
    """Public benchmark task contract.

    Visible to the harness and experts at execution time.
    Does not contain ground-truth values.

    Attributes:
        task_id: Unique identifier (e.g., "FloodRSCT-Harvey-Houston-001").
        geometry: Task geometry type
            (prediction | ranking | clustering | transfer | relational | allocation).
        reasoning_level: Pearl's causal ladder
            (0=observational, 1=interventional, 2=counterfactual, 3=structural).
        question: Natural-language task description.
        scenario: Event, region, dates, substrate, and other context.
        output_fields: Schema of expected numeric outputs with tolerances.
    """

    task_id: str
    geometry: str
    reasoning_level: int
    question: str
    scenario: dict[str, Any] = field(default_factory=dict)
    output_fields: tuple[NumericField, ...] = ()

    def __post_init__(self):
        valid_geom = {"prediction", "ranking", "clustering",
                      "transfer", "relational", "allocation"}
        if self.geometry not in valid_geom:
            raise ValueError(
                f"geometry must be one of {valid_geom}, got {self.geometry!r}"
            )
        if not 0 <= self.reasoning_level <= 3:
            raise ValueError(
                f"reasoning_level must be 0-3, got {self.reasoning_level}"
            )
