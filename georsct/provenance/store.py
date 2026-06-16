"""Local trace store — file-system implementation.

Generic provenance infrastructure for development and testing.
The TraceStore ABC lives in ports/trace_store.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from georsct.ports.trace_store import TraceStore
from georsct.provenance.trace import Trace


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
