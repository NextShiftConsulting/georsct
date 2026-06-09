"""Completion manifest writer for experiment audit."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import ArtifactRecord, CellKey


def build_manifest(
    experiment: str,
    bucket: str,
    artifacts: dict[str, ArtifactRecord],
    contracted_cells: frozenset[CellKey],
) -> dict[str, Any]:
    """Build a JSON-serializable completion manifest for an experiment audit.

    Args:
        experiment: Experiment identifier string.
        bucket: S3 bucket name.
        artifacts: Mapping of artifact keys to their ArtifactRecord metadata.
        contracted_cells: Set of CellKeys declared in the experiment contract.

    Returns:
        A JSON-serializable dict summarising the audit state.
    """
    return {
        "experiment": experiment,
        "bucket": bucket,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contracted_cells": sorted(str(c) for c in contracted_cells),
        "artifact_count": len(artifacts),
        "artifacts": [
            {
                "s3_key": record.s3_key,
                "exists": record.exists,
                "size_bytes": record.size_bytes,
                "internal_timestamp": record.internal_timestamp,
                "s3_last_modified": record.last_modified,
            }
            for record in sorted(artifacts.values(), key=lambda r: r.s3_key)
        ],
    }
