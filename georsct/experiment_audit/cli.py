"""CLI entry point for experiment audit post-flight verification."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for experiment-audit.

    Returns:
        Configured ArgumentParser with required and optional arguments.
    """
    parser = argparse.ArgumentParser(
        prog="experiment-audit",
        description="Post-flight experiment contract verification",
    )
    parser.add_argument(
        "--contract",
        required=True,
        help="Path to EXPERIMENT_CONTRACT.yaml",
    )
    parser.add_argument(
        "--s3-prefix",
        required=True,
        help="S3 prefix for results (e.g. results/s035/)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for audit report and manifest",
    )
    parser.add_argument(
        "--bucket",
        default="swarm-floodrsct-data",
        help="S3 bucket (default: swarm-floodrsct-data)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="JSON only, no Markdown",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Orchestrate the full experiment audit.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        0 if all checks pass or warn, 1 if any check fails.
    """
    from .contract import parse_contract
    from .content_readers import extract_cells_from_runs
    from .manifest import build_manifest
    from .models import AuditResult, CellKey
    from .report import render_json, render_markdown
    from .s3_inventory import S3Inventory
    from .validators import (
        gate_1_contract_parse,
        gate_2_declared_outputs,
        gate_3_convention_supplement,
        gate_4_cell_matrix,
        gate_5_aggregate_content,
        gate_6_ordering,
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Parse contract
    contract = parse_contract(args.contract)

    # 2. Build S3 inventory
    inventory = S3Inventory(bucket=args.bucket)
    s3_keys = inventory.list_prefix(args.s3_prefix)

    # Download JSONs and check declared output keys
    artifacts: dict[str, object] = {}
    for key in s3_keys:
        if key.endswith(".json"):
            artifacts[key] = inventory.download_json(key)
        else:
            artifacts[key] = inventory.check_key(key)

    # Also check declared output keys that may not be in the listing
    for phase in contract.phases:
        for resolved_key in phase.resolve_output_keys(contract.scenarios):
            if resolved_key not in artifacts:
                artifacts[resolved_key] = inventory.check_key(resolved_key)

    # 3. Run all 6 gates in order
    checks = []
    checks.extend(gate_1_contract_parse(contract))
    checks.extend(gate_2_declared_outputs(contract, artifacts))
    checks.extend(gate_3_convention_supplement(contract, s3_keys, args.s3_prefix))

    # Gate 4: extract actual cells from per-scenario JSONs
    actual_cells: set[CellKey] = set()
    for key, record in artifacts.items():
        if hasattr(record, "content") and record.content is not None:
            cell_metrics = extract_cells_from_runs(record.content)
            for cm in cell_metrics:
                actual_cells.add(cm.cell)
    checks.extend(gate_4_cell_matrix(contract, actual_cells))

    checks.extend(gate_5_aggregate_content(contract, artifacts))
    checks.extend(gate_6_ordering(artifacts, contract.scenarios))

    # 4. Build AuditResult
    result = AuditResult(
        checks=checks,
        metadata={
            "experiment_name": contract.experiment,
            "bucket": args.bucket,
            "s3_prefix": args.s3_prefix,
            "contract_path": str(args.contract),
            "counts": {
                "s3_keys": len(s3_keys),
                "artifacts_checked": len(artifacts),
                "contracted_cells": len(contract.cell_matrix),
                "actual_cells": len(actual_cells),
            },
        },
    )

    # 5-7. Write outputs
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "experiment_audit.json"
    json_path.write_text(render_json(result), encoding="utf-8")

    if not args.json_output:
        md_path = out_dir / "experiment_audit.md"
        md_path.write_text(render_markdown(result), encoding="utf-8")

    manifest = build_manifest(
        experiment=contract.experiment,
        bucket=args.bucket,
        artifacts=artifacts,
        contracted_cells=contract.cell_matrix,
    )
    manifest_path = out_dir / "completion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # 8. Log summary
    counts = result.summary_counts()
    status = result.overall_status()
    logger.info(
        "Audit complete: %s (PASS=%d, WARN=%d, FAIL=%d)",
        status,
        counts["PASS"],
        counts["WARN"],
        counts["FAIL"],
    )

    # 9. Return exit code
    return 1 if status == "FAIL" else 0
