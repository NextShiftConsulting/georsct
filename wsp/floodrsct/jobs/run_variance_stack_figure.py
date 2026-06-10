#!/usr/bin/env python3
"""
run_variance_stack_figure.py -- Generate variance stack figure from S3 results.

Reads variance_stack_results.json from S3, reconstructs per-cell ladder
DataFrames, and renders the multi-panel figure via render_ladder_panel.

Usage:
    python run_variance_stack_figure.py --upload
    python run_variance_stack_figure.py
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client

from georsct.analysis.render_ladder import render_ladder_panel

SIDECAR_PREFIX = "results/s035/sidecar/variance_stack"
RESULTS_KEY = f"{SIDECAR_PREFIX}/variance_stack_results.json"
FIGURE_KEY = f"{SIDECAR_PREFIX}/fig_variance_stack.pdf"
FIGURE_PNG_KEY = f"{SIDECAR_PREFIX}/fig_variance_stack.png"


def load_results(s3) -> dict:
    """Load variance_stack_results.json from S3."""
    log.info("Loading s3://%s/%s", BUCKET, RESULTS_KEY)
    resp = s3.get_object(Bucket=BUCKET, Key=RESULTS_KEY)
    return json.loads(resp["Body"].read().decode())


def results_to_ladders(payload: dict) -> dict[str, pd.DataFrame]:
    """Convert JSON cells to {label: DataFrame} for render_ladder_panel."""
    ladders = {}
    for cell in payload["cells"]:
        scenario = cell["scenario"].replace("_", " ").title()
        label = f"{scenario}\n{cell['target_label']}"
        df = pd.DataFrame(cell["ladder"])
        ladders[label] = df
    return ladders


def main():
    parser = argparse.ArgumentParser(
        description="Generate variance stack figure from S3 results"
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()
    payload = load_results(s3)
    log.info("Loaded %d cells", len(payload["cells"]))

    ladders = results_to_ladders(payload)
    log.info("Cells: %s", list(ladders.keys()))

    out_path = Path("/tmp/fig_variance_stack.pdf")
    render_ladder_panel(
        ladders, out_path,
        suptitle="Variance-Control Stack: Deployment-Aligned Decomposition (S035)",
        also_png=True,
        ncols=3,
    )
    log.info("Rendered figure: %s", out_path)

    if args.upload:
        # Upload PDF
        with open(out_path, "rb") as f:
            s3.put_object(
                Bucket=BUCKET, Key=FIGURE_KEY,
                Body=f.read(),
                ContentType="application/pdf",
            )
        log.info("Uploaded s3://%s/%s", BUCKET, FIGURE_KEY)

        # Upload PNG
        png_path = out_path.with_suffix(".png")
        if png_path.exists():
            with open(png_path, "rb") as f:
                s3.put_object(
                    Bucket=BUCKET, Key=FIGURE_PNG_KEY,
                    Body=f.read(),
                    ContentType="image/png",
                )
            log.info("Uploaded s3://%s/%s", BUCKET, FIGURE_PNG_KEY)
    else:
        log.info("Local output: %s", out_path)
        log.info("Local PNG: %s", out_path.with_suffix(".png"))

    print(f"\nDone. {len(ladders)} cells rendered.")


if __name__ == "__main__":
    main()
