"""Trace store port — ABC for persisting execution traces.

Tables implied by this interface:
  - workflow_trace (task_id, timestamp, verdict, ...)
  - workflow_step (step_index, tool_name, admission_reason, ...)
  - artifact (artifact_id, uri, checksum, ...)
  - score_process / score_outcome / score_geocert
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from georsct.provenance.trace import Trace


class TraceStore(ABC):
    """Port: persistence for execution traces."""

    @abstractmethod
    def save_trace(self, trace: Trace) -> str:
        """Persist a trace. Returns a storage key."""

    @abstractmethod
    def load_trace(self, task_id: str) -> Optional[Trace]:
        """Load a trace by task_id."""

    @abstractmethod
    def list_traces(self, prefix: str = "") -> list[str]:
        """List stored trace task_ids."""
