#!/usr/bin/env python3
"""Thin wrapper: render model-ladder figure from per_target_h2_breakdown.json."""

from __future__ import annotations

import json
from pathlib import Path

from georsct.visualization.model_ladder import render_model_ladder

DATA = Path(__file__).resolve().parents[1] / (
    "wsp/floodrsct/exp/s035-model-ladder"
)
OUT = DATA / "mmar-input" / "figures"


def main():
    with open(DATA / "results" / "per_target_h2_breakdown.json") as f:
        cells = json.load(f)["per_cell_table"]

    pdf = render_model_ladder(cells, OUT / "fig_model_ladder.pdf")
    print(f"Saved: {pdf}")
    print(f"Saved: {pdf.with_suffix('.png')}")


if __name__ == "__main__":
    main()
