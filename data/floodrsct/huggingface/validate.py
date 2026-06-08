"""Validate that the HuggingFace dataset loads correctly after upload.

Usage:
    pip install datasets
    python validate.py                        # validate from Hub
    python validate.py --local ./staging      # validate local staging dir
"""

import argparse
import sys

REPO_ID = "rudymartin/floodrsct"
SCENARIOS = ["houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"]

# Columns that must exist in every scenario
REQUIRED_COLUMNS = [
    "acs_total_pop",
    "svi_overall",
    "flood_pct_zone_a",
    "nfip_historical_frequency",
]

# Columns specific to certain scenarios
SCENARIO_COLUMNS = {
    "houston": ["rainfall_total_mm", "peak_stage_ft"],
    "new_orleans": ["levee_condition_rating"],
    "nyc": ["subway_station_count", "flood_311_count"],
    "riverside_coachella": ["burn_scar_overlap"],
    "southwest_florida": ["slosh_max_surge_m"],
}


def validate_scenario(ds, scenario: str) -> list[str]:
    """Validate a single scenario dataset. Returns list of issues."""
    issues = []

    if ds.num_rows == 0:
        issues.append(f"{scenario}: empty dataset")
        return issues

    columns = ds.column_names

    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in columns:
            issues.append(f"{scenario}: missing required column '{col}'")

    # Check scenario-specific columns
    for col in SCENARIO_COLUMNS.get(scenario, []):
        if col not in columns:
            issues.append(f"{scenario}: missing scenario column '{col}'")

    print(f"  {scenario}: {ds.num_rows} rows, {len(columns)} columns")
    return issues


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
