"""Shared constants for experiment audit checks.

TODO(post-submission): move to georsct.experiment_audit.constants as canonical
location. Other consumers (wsp/floodrsct/jobs/_coverage_common.py,
wsp/floodrsct/jobs/validate_experiment_readiness.py) should import from here.
Currently mirrors wsp/floodrsct/jobs/_coverage_common.py lines 30-46.
"""
from __future__ import annotations

from pathlib import Path

from .models import CellKey

BUCKET = "swarm-floodrsct-data"

# scenario -> assembled parquet S3 key
OUTPUT_KEYS: dict[str, str] = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}

# scenario -> short key used in supplement filenames
SCENARIO_KEYS: dict[str, str] = {
    sc: Path(path).stem.replace("_event_features", "")
    for sc, path in OUTPUT_KEYS.items()
}

# logical level -> S3 prefix used in result filenames
LEVEL_PREFIXES: dict[str, str] = {
    "r0": "r0",
    "r1": "r1_hydrology",
    "r2": "r2",
    "r3": "r3",
}

# Known VLM providers used in R4 phases.
# The contract uses {vlm} template but does not enumerate providers.
# This is the canonical list until the contract YAML is fixed.
VLM_PROVIDERS: list[str] = [
    "claude_sonnet",
    "gemini_flash",
    "gpt4o",
    "gemini_pro",
    "qwen_vl",
    "claude_haiku",
]

# The contracted cell matrix: (scenario, target) pairs.
# Derived from EXPERIMENT_CONTRACT.yaml lines 44-66.
CONTRACTED_CELLS: frozenset[CellKey] = frozenset([
    # obs_nfip_event_claims: all 5 scenarios
    CellKey("houston", "obs_nfip_event_claims"),
    CellKey("southwest_florida", "obs_nfip_event_claims"),
    CellKey("nyc", "obs_nfip_event_claims"),
    CellKey("riverside_coachella", "obs_nfip_event_claims"),
    CellKey("new_orleans", "obs_nfip_event_claims"),
    # obs_has_311: houston, nyc
    CellKey("houston", "obs_has_311"),
    CellKey("nyc", "obs_has_311"),
    # obs_has_hwm: houston, new_orleans
    CellKey("houston", "obs_has_hwm"),
    CellKey("new_orleans", "obs_has_hwm"),
])

# Target metadata from the contract
TARGET_SPECS: dict[str, dict] = {
    "obs_nfip_event_claims": {
        "task": "regression",
        "metric": "r2",
        "scenarios": [
            "houston", "southwest_florida", "nyc",
            "riverside_coachella", "new_orleans",
        ],
    },
    "obs_has_311": {
        "task": "binary_classification",
        "metric": "roc_auc",
        "scenarios": ["houston", "nyc"],
    },
    "obs_has_hwm": {
        "task": "binary_classification",
        "metric": "roc_auc",
        "scenarios": ["houston", "new_orleans"],
    },
}
