"""Tests for experiment_audit.manifest module."""
from __future__ import annotations

import json

from georsct.experiment_audit.manifest import build_manifest
from georsct.experiment_audit.models import ArtifactRecord, CellKey


def _sample_artifacts() -> dict[str, ArtifactRecord]:
    return {
        "b_key": ArtifactRecord(
            s3_key="s3://bucket/b.json",
            exists=True,
            size_bytes=256,
            last_modified="2026-01-02T00:00:00+00:00",
            internal_timestamp="2026-01-01T12:00:00+00:00",
        ),
        "a_key": ArtifactRecord(
            s3_key="s3://bucket/a.json",
            exists=False,
            size_bytes=None,
            last_modified=None,
            internal_timestamp=None,
        ),
    }


def _sample_cells() -> frozenset[CellKey]:
    return frozenset({
        CellKey(scenario="houston", target="depth"),
        CellKey(scenario="nola", target="claims"),
    })


class TestBuildManifest:
    """Tests for build_manifest."""

    def test_structure_has_correct_experiment_and_bucket(self) -> None:
        result = build_manifest(
            experiment="s035",
            bucket="swarm-yrsn-checkpoints",
            artifacts=_sample_artifacts(),
            contracted_cells=_sample_cells(),
        )
        assert result["experiment"] == "s035"
        assert result["bucket"] == "swarm-yrsn-checkpoints"

    def test_contracted_cells_are_sorted_strings(self) -> None:
        result = build_manifest(
            experiment="s035",
            bucket="bucket",
            artifacts={},
            contracted_cells=_sample_cells(),
        )
        cells = result["contracted_cells"]
        assert cells == sorted(cells)
        assert all(isinstance(c, str) for c in cells)
        assert "houston/depth" in cells
        assert "nola/claims" in cells

    def test_json_serializable(self) -> None:
        result = build_manifest(
            experiment="s035",
            bucket="bucket",
            artifacts=_sample_artifacts(),
            contracted_cells=_sample_cells(),
        )
        # Must not raise
        serialized = json.dumps(result)
        roundtrip = json.loads(serialized)
        assert roundtrip["experiment"] == "s035"

    def test_artifact_count_matches(self) -> None:
        arts = _sample_artifacts()
        result = build_manifest(
            experiment="s035",
            bucket="bucket",
            artifacts=arts,
            contracted_cells=frozenset(),
        )
        assert result["artifact_count"] == len(arts)

    def test_artifacts_sorted_by_s3_key(self) -> None:
        result = build_manifest(
            experiment="s035",
            bucket="bucket",
            artifacts=_sample_artifacts(),
            contracted_cells=frozenset(),
        )
        keys = [a["s3_key"] for a in result["artifacts"]]
        assert keys == sorted(keys)

    def test_provenance_internal_timestamp_present(self) -> None:
        result = build_manifest(
            experiment="s035",
            bucket="bucket",
            artifacts=_sample_artifacts(),
            contracted_cells=frozenset(),
        )
        for entry in result["artifacts"]:
            assert "internal_timestamp" in entry

        # The artifact with a timestamp should have its value
        ts_values = [a["internal_timestamp"] for a in result["artifacts"] if a["internal_timestamp"] is not None]
        assert len(ts_values) == 1
        assert ts_values[0] == "2026-01-01T12:00:00+00:00"

    def test_generated_at_is_iso_string(self) -> None:
        result = build_manifest(
            experiment="s035",
            bucket="bucket",
            artifacts={},
            contracted_cells=frozenset(),
        )
        assert isinstance(result["generated_at"], str)
        assert "T" in result["generated_at"]

    def test_s3_last_modified_mapped(self) -> None:
        result = build_manifest(
            experiment="s035",
            bucket="bucket",
            artifacts=_sample_artifacts(),
            contracted_cells=frozenset(),
        )
        existing = [a for a in result["artifacts"] if a["exists"]]
        assert len(existing) == 1
        assert existing[0]["s3_last_modified"] == "2026-01-02T00:00:00+00:00"
