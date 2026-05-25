"""
DOE-6: Geospatial Transfer
Test geometry features on PDFM geospatial embeddings.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")


def construct_pairs_from_quintiles(df: pd.DataFrame, target_col: str):
    """Construct R/S/N pairs from quintile-based labeling."""
    df["quintile"] = pd.qcut(df[target_col], 5, labels=[0, 1, 2, 3, 4])
    pairs = []
    labels = []

    # Sample pairs
    for i in range(len(df)):
        for j in range(i+1, min(i+10, len(df))):
            q_diff = abs(df.iloc[i]["quintile"] - df.iloc[j]["quintile"])
            if q_diff == 0:
                label = 0  # R (same quintile)
            elif q_diff == 1:
                label = 1  # S_sup (adjacent)
            else:
                label = 2  # N (distant)
            pairs.append((i, j))
            labels.append(label)

    return pairs, labels


def main():
    parser = argparse.ArgumentParser(description="DOE-6: Geospatial Transfer")
    parser.add_argument("--target", default="median_income", help="Target variable for quintile labeling")
    args = parser.parse_args()

    print(f"DOE-6: Geospatial Transfer")
    print(f"Target: {args.target}")

    # Load PDFM conus27
    conus_path = INPUT_DIR / "pdfm_conus27.csv"
    if conus_path.exists():
        df = pd.read_csv(conus_path)
        print(f"Loaded {len(df)} zip codes")
    else:
        print(f"PDFM data not found at {conus_path}")
        df = None

    results = {
        "experiment": "DOE-6",
        "timestamp": datetime.now().isoformat(),
        "target": args.target,
        "n_samples": len(df) if df is not None else 0,
        "status": "NOT_IMPLEMENTED"
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "doe6_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
