"""Tests for _s3_result.upload_json_result archive-before-overwrite logic."""

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

# Module under test lives in jobs/; inject into sys.path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "floodrsct" / "jobs"))
from _s3_result import upload_json_result, _s3_safe_ts, _archive_existing


# ---------------------------------------------------------------------------
# _s3_safe_ts
# ---------------------------------------------------------------------------

class TestS3SafeTimestamp:
    def test_iso_with_tz(self):
        assert _s3_safe_ts("2026-06-02T14:30:00+00:00") == "20260602T143000Z"

    def test_iso_no_tz(self):
        result = _s3_safe_ts("2026-06-02T14:30:00")
        assert ":" not in result
        assert "/" not in result

    def test_unknown_passthrough(self):
        result = _s3_safe_ts("unknown")
        assert result == "unknown"

    def test_truncation(self):
        long_ts = "2026-06-02T14:30:00.123456+00:00"
        result = _s3_safe_ts(long_ts)
        assert len(result) <= 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_s3_mock(existing_body: bytes | None = None, error_code: str | None = None):
    """Build a mock S3 client.

    Args:
        existing_body: If not None, get_object returns this body.
        error_code: If set, get_object raises ClientError with this code.
    """
    s3 = MagicMock()
    if error_code:
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": error_code, "Message": "test"}}, "GetObject"
        )
    elif existing_body is not None:
        body_mock = MagicMock()
        body_mock.read.return_value = existing_body
        s3.get_object.return_value = {"Body": body_mock}
    else:
        # NoSuchKey
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject"
        )
    return s3


# ---------------------------------------------------------------------------
# upload_json_result
# ---------------------------------------------------------------------------

class TestUploadJsonResult:
    """Core contract tests for the archive-safe upload helper."""

    def test_first_write_no_archive(self):
        """First write (no existing key) uploads without archive."""
        s3 = _make_s3_mock(error_code="NoSuchKey")
        payload = {"experiment": "test", "timestamp": "2026-06-02T00:00:00+00:00"}

        upload_json_result(s3, "bucket", "results/r0.json", payload, git_hash="abc123")

        # get_object attempted (to check for existing)
        s3.get_object.assert_called_once_with(Bucket="bucket", Key="results/r0.json")
        # put_object called exactly once (the canonical write, no archive)
        assert s3.put_object.call_count == 1
        call_kwargs = s3.put_object.call_args.kwargs
        assert call_kwargs["Key"] == "results/r0.json"
        assert call_kwargs["ContentType"] == "application/json"

    def test_overwrite_archives_old(self):
        """When a result already exists, it is archived before overwrite."""
        old = json.dumps({"timestamp": "2026-06-01T12:00:00+00:00", "data": "old"}).encode()
        s3 = _make_s3_mock(existing_body=old)

        payload = {"experiment": "test", "timestamp": "2026-06-02T00:00:00+00:00"}
        upload_json_result(s3, "bucket", "results/r0.json", payload, git_hash="deadbeef99")

        # Two put_object calls: archive + canonical
        assert s3.put_object.call_count == 2

        archive_call = s3.put_object.call_args_list[0]
        archive_key = archive_call.kwargs["Key"]
        assert "/archive/" in archive_key
        assert "deadbeef" in archive_key  # git hash truncated to 8
        assert "20260601" in archive_key  # old timestamp in key
        assert archive_call.kwargs["Body"] == old

        canonical_call = s3.put_object.call_args_list[1]
        assert canonical_call.kwargs["Key"] == "results/r0.json"

    def test_nosuchkey_allowed(self):
        """NoSuchKey on read (404) is silently handled -- first run."""
        s3 = _make_s3_mock(error_code="404")
        payload = {"experiment": "test"}
        # Should not raise
        upload_json_result(s3, "bucket", "results/r0.json", payload)
        assert s3.put_object.call_count == 1

    def test_access_denied_raises(self):
        """AccessDenied during archive read must propagate -- not swallowed."""
        s3 = _make_s3_mock(error_code="AccessDenied")
        payload = {"experiment": "test"}

        with pytest.raises(ClientError) as exc_info:
            upload_json_result(s3, "bucket", "results/r0.json", payload)
        assert exc_info.value.response["Error"]["Code"] == "AccessDenied"

    def test_malformed_json_still_archives(self):
        """If old object is not valid JSON, archive it anyway with fallback ts."""
        s3 = _make_s3_mock(existing_body=b"NOT-JSON{{{")
        payload = {"experiment": "test"}
        upload_json_result(s3, "bucket", "results/r0.json", payload, git_hash="abc12345")

        assert s3.put_object.call_count == 2
        archive_call = s3.put_object.call_args_list[0]
        assert "unparseable" in archive_call.kwargs["Key"]
        assert archive_call.kwargs["Body"] == b"NOT-JSON{{{"

    def test_archive_key_is_s3_safe(self):
        """Archive key must not contain colons or other S3-unsafe chars."""
        old = json.dumps({"timestamp": "2026-06-01T12:30:45+00:00"}).encode()
        s3 = _make_s3_mock(existing_body=old)
        payload = {"experiment": "test"}

        upload_json_result(s3, "bucket", "results/s035/r0_houston.json", payload, git_hash="ff00ff00")

        archive_key = s3.put_object.call_args_list[0].kwargs["Key"]
        assert ":" not in archive_key
        assert "+" not in archive_key
        assert archive_key.startswith("results/s035/archive/")

    def test_git_hash_from_env(self):
        """Falls back to S035_GIT_HASH env var when git_hash not provided."""
        old = json.dumps({"timestamp": "2026-01-01T00:00:00+00:00"}).encode()
        s3 = _make_s3_mock(existing_body=old)

        with patch.dict("os.environ", {"S035_GIT_HASH": "envhash123"}):
            upload_json_result(s3, "bucket", "results/r0.json", {"test": 1})

        archive_key = s3.put_object.call_args_list[0].kwargs["Key"]
        assert "envhash1" in archive_key  # truncated to 8

    def test_numpy_scalars_serialize(self):
        """ResultJSONEncoder handles numpy int/float/array."""
        import numpy as np
        s3 = _make_s3_mock(error_code="NoSuchKey")
        payload = {
            "value": np.float64(0.95),
            "count": np.int64(42),
            "array": np.array([1.0, 2.0, 3.0]),
        }
        upload_json_result(s3, "bucket", "results/r0.json", payload)
        assert s3.put_object.call_count == 1
        written = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert written["value"] == 0.95
        assert written["count"] == 42
        assert written["array"] == [1.0, 2.0, 3.0]

    def test_datetime_serializes(self):
        """ResultJSONEncoder handles datetime objects."""
        from datetime import datetime, timezone
        s3 = _make_s3_mock(error_code="NoSuchKey")
        payload = {"ts": datetime(2026, 6, 2, tzinfo=timezone.utc)}
        upload_json_result(s3, "bucket", "results/r0.json", payload)
        written = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert written["ts"] == "2026-06-02T00:00:00+00:00"

    def test_unknown_type_raises_before_s3_write(self):
        """Unrecognized types raise TypeError before any S3 write."""
        s3 = _make_s3_mock(error_code="NoSuchKey")

        class BadObj:
            pass

        with pytest.raises(TypeError):
            upload_json_result(s3, "bucket", "results/r0.json", {"bad": BadObj()})
        s3.put_object.assert_not_called()
