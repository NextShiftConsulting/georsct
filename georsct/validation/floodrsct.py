"""FloodRSCT dataset schema validation.

Defines required columns and per-scenario column contracts for the
FloodRSCT HuggingFace dataset.
"""

from __future__ import annotations

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

    for col in REQUIRED_COLUMNS:
        if col not in columns:
            issues.append(f"{scenario}: missing required column '{col}'")

    for col in SCENARIO_COLUMNS.get(scenario, []):
        if col not in columns:
            issues.append(f"{scenario}: missing scenario column '{col}'")

    return issues
