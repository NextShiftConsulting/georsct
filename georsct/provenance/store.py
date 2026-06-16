"""GeoCertDB provenance store — Layer 4.

Persists traces, certificates, artifacts, and scores.
Extends the existing CertificateRepository pattern from ports/.

Import rule: depends only on provenance.trace and contracts.
S3 access uses the port pattern — the ABC is here, the S3
adapter would live in flood/adapters/outbound/s3/.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from georsct.provenance.trace import Trace


class TraceStore(ABC):
    """Port: persistence for execution traces.

    Tables implied by this interface:
      - workflow_trace (task_id, timestamp, verdict, ...)
      - workflow_step (step_index, tool_name, admission_reason, ...)
      - artifact (artifact_id, uri, checksum, ...)
      - score_process / score_outcome / score_geocert
    """

    @abstractmethod
    def save_trace(self, trace: Trace) -> str:
        """Persist a trace. Returns a storage key."""

    @abstractmethod
    def load_trace(self, task_id: str) -> Optional[Trace]:
        """Load a trace by task_id."""

    @abstractmethod
    def list_traces(self, prefix: str = "") -> list[str]:
        """List stored trace task_ids."""


class LocalTraceStore(TraceStore):
    """File-system trace store for local development and testing."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_trace(self, trace: Trace) -> str:
        path = self.base_dir / f"{trace.task_id}.json"
        path.write_text(trace.to_json(), encoding="utf-8")
        return str(path)

    def load_trace(self, task_id: str) -> Optional[Trace]:
        path = self.base_dir / f"{task_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        # Reconstruct minimal Trace (full deserialization deferred)
        trace = Trace(task_id=data["task_id"])
        trace.final_json = data.get("final_json", {})
        return trace

    def list_traces(self, prefix: str = "") -> list[str]:
        return [
            p.stem for p in self.base_dir.glob(f"{prefix}*.json")
        ]
