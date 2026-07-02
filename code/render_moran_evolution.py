#!/usr/bin/env python3
"""Thin wrapper: render Moran's I evolution figure from rollup CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from georsct.visualization.moran_evolution import render_moran_evolution

DATA = Path(__file__).resolve().parents[1] / (
    "wsp/floodrsct/exp/s035-model-ladder/mmar-input"
)


def main():
    lisa = pd.read_csv(DATA / "lisa_rollup.csv")
    geary = pd.read_csv(DATA / "geary_rollup.csv")

    pdf = render_moran_evolution(lisa, geary, DATA / "figures" / "fig_moran_evolution.pdf")
    print(f"Saved: {pdf}")
    print(f"Saved: {pdf.with_suffix('.png')}")


if __name__ == "__main__":
    main()
