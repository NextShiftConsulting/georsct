"""Tests for experiment_audit.cli -- parser only, no S3 calls."""
from __future__ import annotations

import pytest

from georsct.experiment_audit.cli import build_parser


class TestBuildParser:
    """Tests for build_parser argument parsing."""

    def test_required_args_parse(self) -> None:
        """Required args (--contract, --s3-prefix, --out) parse correctly."""
        parser = build_parser()
        args = parser.parse_args([
            "--contract", "path/to/contract.yaml",
            "--s3-prefix", "results/s035/",
            "--out", "/tmp/audit_out",
        ])
        assert args.contract == "path/to/contract.yaml"
        assert args.s3_prefix == "results/s035/"
        assert args.out == "/tmp/audit_out"

    def test_default_bucket(self) -> None:
        """Default bucket is swarm-floodrsct-data."""
        parser = build_parser()
        args = parser.parse_args([
            "--contract", "c.yaml",
            "--s3-prefix", "results/s035/",
            "--out", "/tmp/out",
        ])
        assert args.bucket == "swarm-floodrsct-data"

    def test_custom_bucket(self) -> None:
        """--bucket overrides the default."""
        parser = build_parser()
        args = parser.parse_args([
            "--contract", "c.yaml",
            "--s3-prefix", "results/s035/",
            "--out", "/tmp/out",
            "--bucket", "my-custom-bucket",
        ])
        assert args.bucket == "my-custom-bucket"

    def test_default_json_output_false(self) -> None:
        """json_output defaults to False."""
        parser = build_parser()
        args = parser.parse_args([
            "--contract", "c.yaml",
            "--s3-prefix", "results/s035/",
            "--out", "/tmp/out",
        ])
        assert args.json_output is False

    def test_json_flag_sets_true(self) -> None:
        """--json sets json_output to True."""
        parser = build_parser()
        args = parser.parse_args([
            "--contract", "c.yaml",
            "--s3-prefix", "results/s035/",
            "--out", "/tmp/out",
            "--json",
        ])
        assert args.json_output is True

    def test_missing_required_args_exits(self) -> None:
        """Missing required args cause SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_missing_contract_exits(self) -> None:
        """Missing --contract causes SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--s3-prefix", "x", "--out", "y"])

    def test_missing_s3_prefix_exits(self) -> None:
        """Missing --s3-prefix causes SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--contract", "x", "--out", "y"])

    def test_missing_out_exits(self) -> None:
        """Missing --out causes SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--contract", "x", "--s3-prefix", "y"])

    def test_prog_name(self) -> None:
        """Parser prog is experiment-audit."""
        parser = build_parser()
        assert parser.prog == "experiment-audit"

    def test_description(self) -> None:
        """Parser description matches spec."""
        parser = build_parser()
        assert parser.description == "Post-flight experiment contract verification"
