"""Tests for launcher instance type allowlist (M5)."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Module under test
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wsp" / "floodrsct" / "scripts"))
from _launcher_base import ALLOWED_INSTANCES, launch_processing_job


# Patch upload_code and boto3 so tests never touch AWS or the filesystem.
# We only care about the allowlist guard, which fires before these calls.
_PATCHES = [
    "_launcher_base.upload_code",
    "_launcher_base._upload_bootstrap",
    "_launcher_base.boto3",
    "_launcher_base._get_git_info",
    "_launcher_base._run_preflight",
]


def _launch_with_mocks(**kwargs):
    """Call launch_processing_job with all AWS/filesystem ops mocked."""
    defaults = {
        "job_name": "test-job",
        "job_script": "dummy.py",
        "job_args": [],
        "dry_run": True,
    }
    defaults.update(kwargs)
    with patch(_PATCHES[0], return_value="code/prefix/"), \
         patch(_PATCHES[1]), \
         patch(_PATCHES[2]) as mock_boto3, \
         patch(_PATCHES[3], return_value={"git_hash": "abc", "git_dirty": "false"}), \
         patch(_PATCHES[4], return_value=True):
        mock_boto3.client.return_value = MagicMock()
        return launch_processing_job(**defaults)


class TestInstanceAllowlist:
    """Verify instance type guard on launch_processing_job."""

    def test_allowed_instance_not_rejected(self):
        """Every instance in ALLOWED_INSTANCES passes the guard."""
        for inst in sorted(ALLOWED_INSTANCES):
            result = _launch_with_mocks(instance_type=inst)
            assert result == "test-job"

    def test_denied_instance_raises(self):
        """An instance NOT in the allowlist raises ValueError."""
        with pytest.raises(ValueError, match="not in allowlist"):
            _launch_with_mocks(instance_type="ml.p3.2xlarge")

    def test_missing_instance_type_raises(self):
        """Bogus instance type string is rejected."""
        with pytest.raises(ValueError, match="not in allowlist"):
            _launch_with_mocks(instance_type="not-a-real-instance")

    def test_empty_string_rejected(self):
        """Empty string instance type is rejected."""
        with pytest.raises(ValueError, match="not in allowlist"):
            _launch_with_mocks(instance_type="")

    def test_override_bypasses_allowlist(self):
        """allow_instance_override=True permits any instance type."""
        result = _launch_with_mocks(
            job_name="test-gpu-job",
            instance_type="ml.p3.2xlarge",
            allow_instance_override=True,
        )
        assert result == "test-gpu-job"

    def test_allowlist_is_frozen(self):
        """ALLOWED_INSTANCES is a frozenset -- cannot be mutated at runtime."""
        assert isinstance(ALLOWED_INSTANCES, frozenset)

    def test_allowlist_contains_expected_types(self):
        """Sanity: the default allowlist includes the common m5 family."""
        assert "ml.m5.large" in ALLOWED_INSTANCES
        assert "ml.m5.xlarge" in ALLOWED_INSTANCES
        assert "ml.m5.2xlarge" in ALLOWED_INSTANCES
