"""Private ground-truth for GeoRSCT-X task scoring.

This module is ONLY imported by the evaluation layer (scoring.py).
The execution layer (harness, experts, gearbox) must NEVER import this.

Separation ensures the gold answer cannot leak into the inference path.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoldField:
    """Ground-truth value for one output field."""

    key: str
    value: float
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    floor_scale: float = 0.0


@dataclass(frozen=True)
class TaskGold:
    """Private ground-truth for a benchmark task.

    Only the scorer may access this. The harness never sees it.
    """

    task_id: str
    fields: tuple[GoldField, ...] = ()

    def as_dict(self) -> dict[str, GoldField]:
        """Map field key -> GoldField for scorer lookup."""
        return {f.key: f for f in self.fields}
