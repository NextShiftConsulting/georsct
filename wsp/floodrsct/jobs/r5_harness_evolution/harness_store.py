"""Harness version storage for R5.

Reads/writes harness versions, patches, failure reports, and certificate
trajectories to local filesystem or S3. All artifacts are JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class HarnessStore:
    """Local filesystem store for harness evolution artifacts."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / "harness").mkdir(exist_ok=True)
        (self.base / "patches").mkdir(exist_ok=True)
        (self.base / "failures").mkdir(exist_ok=True)
        (self.base / "scores").mkdir(exist_ok=True)

    def save_harness(self, harness_id: str, data: dict) -> Path:
        path = self.base / "harness" / f"{harness_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        log.info("Saved harness %s -> %s", harness_id, path)
        return path

    def load_harness(self, harness_id: str) -> dict:
        path = self.base / "harness" / f"{harness_id}.json"
        return json.loads(path.read_text())

    def save_patch(self, patch_id: str, data: dict) -> Path:
        path = self.base / "patches" / f"{patch_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        log.info("Saved patch %s -> %s", patch_id, path)
        return path

    def save_failure_report(self, step: int, data: dict) -> Path:
        path = self.base / "failures" / f"failure_report_t{step:03d}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def save_scores(self, step: int, split: str, data: dict) -> Path:
        path = self.base / "scores" / f"scores_t{step:03d}_{split}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def save_trajectory(self, data: dict) -> Path:
        path = self.base / "certificate_trajectory.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        log.info("Saved certificate trajectory -> %s", path)
        return path

    def list_harness_versions(self) -> list[str]:
        return sorted(
            p.stem for p in (self.base / "harness").glob("harness_v*.json")
        )


class S3HarnessStore:
    """S3-backed store. Delegates to local store then uploads."""

    def __init__(self, local_dir: str | Path, bucket: str, prefix: str):
        self.local = HarnessStore(local_dir)
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._s3 = None

    def _get_s3(self):
        if self._s3 is None:
            import boto3
            from swarm_auth import get_aws_credentials
            self._s3 = boto3.client(
                "s3", region_name="us-east-1", **get_aws_credentials()
            )
        return self._s3

    def _upload(self, local_path: Path) -> str:
        rel = local_path.relative_to(self.local.base)
        key = f"{self.prefix}/{rel.as_posix()}"
        self._get_s3().upload_file(str(local_path), self.bucket, key)
        log.info("Uploaded s3://%s/%s", self.bucket, key)
        return key

    def save_harness(self, harness_id: str, data: dict) -> str:
        local_path = self.local.save_harness(harness_id, data)
        return self._upload(local_path)

    def save_patch(self, patch_id: str, data: dict) -> str:
        local_path = self.local.save_patch(patch_id, data)
        return self._upload(local_path)

    def save_failure_report(self, step: int, data: dict) -> str:
        local_path = self.local.save_failure_report(step, data)
        return self._upload(local_path)

    def save_trajectory(self, data: dict) -> str:
        local_path = self.local.save_trajectory(data)
        return self._upload(local_path)
