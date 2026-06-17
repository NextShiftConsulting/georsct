"""Tests for kappa independence: geometry kappa must not read model artifacts."""

import pytest
import sys
from pathlib import Path

# Add jobs directory to path for import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wsp" / "floodrsct" / "jobs"))

from compute_geometry_kappa import _guard_s3_read


class TestGuardRejectsModelOutputs:
    """_guard_s3_read must raise on any model-derived S3 key."""

    BAD_PATHS = [
        "results/s035/r0_houston.json",
        "results/s035/r1_houston.json",
        "results/s035/r2_houston.json",
        "results/s035/r0_houston_predictions.parquet",
        "results/s035/r1_nyc_predictions.parquet",
        "results/s035/r2_southwest_florida.json",
        "results/s035/diagnostics_r0.json",
        "results/s035/diagnostics_r1.json",
        "results/s035/diagnostics_r2.json",
        "results/s035/certificates_r0.json",
        "results/s035/certificates_r1.json",
        "results/s035/certificates_r2.json",
        "results/s035/uplift_table.json",
        "results/s035/money_table.json",
        "results/s035/predictions/r0_houston.parquet",
        "results/s035/residuals/r0_houston.parquet",
        "folds/houston_folds.parquet",
        "folds/nyc_folds.parquet",
    ]

    @pytest.mark.parametrize("key", BAD_PATHS)
    def test_rejects_model_artifact(self, key):
        with pytest.raises(RuntimeError, match="KAPPA INDEPENDENCE VIOLATION"):
            _guard_s3_read(key)


class TestGuardAllowsGeometryInputs:
    """_guard_s3_read must allow geometry-only S3 keys."""

    GOOD_PATHS = [
        "raw/geocertdb2026/zcta_adjacency.parquet",
        "raw/geocert/zcta_adjacency.parquet",
        "raw/geocertdb2026/zcta_county_crosswalk.parquet",
        "processed/houston/houston_assembled.parquet",
        "processed/nyc/nyc_assembled.parquet",
        "raw/geocertdb2026/zcta_features_labels.parquet",
        "results/s035/geometry_kappa.json",
    ]

    @pytest.mark.parametrize("key", GOOD_PATHS)
    def test_allows_geometry_input(self, key):
        _guard_s3_read(key)  # Should not raise
