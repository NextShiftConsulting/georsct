"""Validate that the FloodRSCT HuggingFace dataset loads correctly after upload.

Usage:
    pip install datasets
    python validate_hf.py                        # validate from Hub
    python validate_hf.py --local ./staging      # validate local staging dir
"""

import argparse
import sys

from georsct.validation.floodrsct import SCENARIOS, validate_scenario

REPO_ID = "rudymartin/floodrsct"


def main():
    parser = argparse.ArgumentParser(description="Validate FloodRSCT HF dataset")
    parser.add_argument("--local", type=str, help="Path to local staging dir (skip Hub)")
    args = parser.parse_args()

    from datasets import load_dataset

    all_issues = []

    for scenario in SCENARIOS:
        try:
            if args.local:
                ds = load_dataset(
                    "parquet",
                    data_files=f"{args.local}/data/{scenario}/*.parquet",
                    split="train",
                )
            else:
                ds = load_dataset(REPO_ID, scenario, split="train")
            print(f"  {scenario}: {ds.num_rows} rows, {len(ds.column_names)} columns")
            issues = validate_scenario(ds, scenario)
            all_issues.extend(issues)
        except Exception as e:
            all_issues.append(f"{scenario}: failed to load -- {e}")
            print(f"  {scenario}: LOAD FAILED -- {e}")

    if all_issues:
        print(f"\n{len(all_issues)} issue(s) found:")
        for issue in all_issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("\nAll scenarios validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
