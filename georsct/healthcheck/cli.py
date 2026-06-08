"""CLI entry point for the failure taxonomy diagnostic tool."""

from __future__ import annotations

import argparse
import sys

from .health_card import build_health_cards
from .loader import load_folder
from .report import render_json, render_text
from .thresholds import resolve_preset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rsct-healthcheck",
        description="RSCT failure taxonomy diagnostic tool",
    )
    parser.add_argument(
        "folder",
        help="Path to folder of JSON result files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Override threshold preset (default: read from certificate)",
    )
    parser.add_argument(
        "--cell",
        action="append",
        dest="cells",
        metavar="SCENARIO:TARGET",
        help="Filter to specific cell(s), repeatable",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Filter to specific level (r0, r1, r2)",
    )
    parser.add_argument(
        "--warnings-only",
        action="store_true",
        help="Only show cells with warnings or failures",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail hard on malformed or conflicting inputs",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show gate math and thresholds used per cell",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print aggregate counts and top issues",
    )

    args = parser.parse_args(argv)

    try:
        snapshot = load_folder(args.folder, strict=args.strict)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if snapshot.loader_warnings:
        for w in snapshot.loader_warnings:
            print(f"[loader] {w}", file=sys.stderr)

    # Resolve preset
    preset_is_override = args.preset is not None
    if args.preset:
        preset = resolve_preset(args.preset)
        snapshot.metadata["preset"] = args.preset
        snapshot.metadata["preset_source"] = "override"
    else:
        cert_preset = snapshot.metadata.get("preset")
        preset = resolve_preset(cert_preset)
        snapshot.metadata["preset_source"] = "certificate"

    cards = build_health_cards(snapshot, preset, preset_is_override=preset_is_override)

    # Filter by cell
    if args.cells:
        cell_filters = set()
        for spec in args.cells:
            parts = spec.split(":", 1)
            if len(parts) == 2:
                cell_filters.add((parts[0], parts[1]))
        if cell_filters:
            cards = [
                c for c in cards
                if (c.cell.scenario, c.cell.target) in cell_filters
            ]

    # Filter by level
    if args.level:
        level = args.level.lower()
        cards = [c for c in cards if c.level == level]

    if not cards:
        print("No matching cells found.", file=sys.stderr)
        return 1

    if args.json_output:
        print(render_json(cards, snapshot.metadata))
    else:
        print(render_text(
            cards,
            snapshot.metadata,
            explain=args.explain,
            warnings_only=args.warnings_only,
            summary_only=args.summary_only,
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
