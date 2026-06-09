"""Tests for S3Inventory using a stub client (no real AWS calls)."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pytest
from botocore.exceptions import ClientError

from georsct.experiment_audit.s3_inventory import S3Inventory
from georsct.experiment_audit.models import ArtifactRecord


def _make_client_error(code: str = "404") -> ClientError:
    """Build a botocore ClientError for a missing key."""
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "Not Found"}},
        operation_name="HeadObject",
    )


class StubS3Client:
    """In-memory S3 client stub for testing."""

    def __init__(self) -> None:
        self._objects: dict[str, dict] = {}

    def put(self, key: str, body: bytes, last_modified: str | None = None) -> None:
        """Store an object for later retrieval."""
        self._objects[key] = {
            "Body": body,
            "ContentLength": len(body),
            "LastModified": last_modified or "2026-01-15T10:00:00+00:00",
        }

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self._objects:
            raise _make_client_error("404")
        obj = self._objects[Key]
        return {
            "ContentLength": obj["ContentLength"],
            "LastModified": obj["LastModified"],
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self._objects:
            raise _make_client_error("NoSuchKey")
        obj = self._objects[Key]
        return {
            "Body": io.BytesIO(obj["Body"]),
            "ContentLength": obj["ContentLength"],
            "LastModified": obj["LastModified"],
        }

    def list_objects_v2(self, *, Bucket: str, Prefix: str, **kwargs) -> dict:
        matching = sorted(k for k in self._objects if k.startswith(Prefix))
        return {
            "Contents": [{"Key": k} for k in matching],
            "IsTruncated": False,
        }


@pytest.fixture
def stub() -> StubS3Client:
    return StubS3Client()


@pytest.fixture
def inventory(stub: StubS3Client) -> S3Inventory:
    return S3Inventory(client=stub, bucket="test-bucket")


class TestCheckKey:
    def test_existing_key(self, stub: StubS3Client, inventory: S3Inventory) -> None:
        stub.put("data/file.json", b"hello")

        record = inventory.check_key("data/file.json")

        assert record.exists is True
        assert record.size_bytes == 5
        assert record.last_modified is not None
        assert record.s3_key == "data/file.json"

    def test_missing_key(self, inventory: S3Inventory) -> None:
        record = inventory.check_key("no/such/key.json")

        assert record.exists is False
        assert record.size_bytes is None
        assert record.last_modified is None


class TestDownloadJson:
    def test_downloads_and_parses(
        self, stub: StubS3Client, inventory: S3Inventory
    ) -> None:
        payload = {"timestamp": "2026-01-10T08:00:00Z", "value": 42}
        stub.put("results/out.json", json.dumps(payload).encode())

        record = inventory.download_json("results/out.json")

        assert record.exists is True
        assert record.content == payload
        assert record.internal_timestamp == "2026-01-10T08:00:00Z"

    def test_missing_key(self, inventory: S3Inventory) -> None:
        record = inventory.download_json("missing/file.json")

        assert record.exists is False
        assert record.content is None

    def test_extracts_created_at(
        self, stub: StubS3Client, inventory: S3Inventory
    ) -> None:
        payload = {"created_at": "2026-02-01T12:00:00Z", "data": [1, 2]}
        stub.put("results/v2.json", json.dumps(payload).encode())

        record = inventory.download_json("results/v2.json")

        assert record.internal_timestamp == "2026-02-01T12:00:00Z"

    def test_timestamp_priority(
        self, stub: StubS3Client, inventory: S3Inventory
    ) -> None:
        """'timestamp' field takes priority over 'created_at'."""
        payload = {
            "created_at": "2026-02-01T12:00:00Z",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        stub.put("results/priority.json", json.dumps(payload).encode())

        record = inventory.download_json("results/priority.json")

        assert record.internal_timestamp == "2026-01-01T00:00:00Z"

    def test_no_internal_timestamp(
        self, stub: StubS3Client, inventory: S3Inventory
    ) -> None:
        payload = {"value": 99}
        stub.put("results/no_ts.json", json.dumps(payload).encode())

        record = inventory.download_json("results/no_ts.json")

        assert record.internal_timestamp is None


class TestListPrefix:
    def test_returns_matching_keys(
        self, stub: StubS3Client, inventory: S3Inventory
    ) -> None:
        stub.put("exp/a/1.json", b"{}")
        stub.put("exp/a/2.json", b"{}")
        stub.put("exp/b/1.json", b"{}")

        keys = inventory.list_prefix("exp/a/")

        assert keys == ["exp/a/1.json", "exp/a/2.json"]

    def test_empty_prefix(self, inventory: S3Inventory) -> None:
        keys = inventory.list_prefix("nonexistent/")

        assert keys == []


class TestBestTimestamp:
    def test_prefers_internal_over_s3(self) -> None:
        record = ArtifactRecord(
            s3_key="k",
            exists=True,
            last_modified="2026-01-15T10:00:00Z",
            internal_timestamp="2026-01-10T08:00:00Z",
        )
        assert record.best_timestamp() == "2026-01-10T08:00:00Z"

    def test_falls_back_to_s3(self) -> None:
        record = ArtifactRecord(
            s3_key="k",
            exists=True,
            last_modified="2026-01-15T10:00:00Z",
        )
        assert record.best_timestamp() == "2026-01-15T10:00:00Z"

    def test_returns_none_when_neither(self) -> None:
        record = ArtifactRecord(s3_key="k", exists=False)
        assert record.best_timestamp() is None
