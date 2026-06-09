"""Verify constants mirror _coverage_common.py."""
from __future__ import annotations

from georsct.experiment_audit.constants import (
    BUCKET,
    LEVEL_PREFIXES,
    OUTPUT_KEYS,
    SCENARIO_KEYS,
    CONTRACTED_CELLS,
)


def test_output_keys_all_five_scenarios():
    assert set(OUTPUT_KEYS.keys()) == {
        "houston", "southwest_florida", "nyc",
        "riverside_coachella", "new_orleans",
    }


def test_scenario_keys_derived():
    # new_orleans -> "no", riverside_coachella -> "rc", etc.
    assert SCENARIO_KEYS["new_orleans"] == "no"
    assert SCENARIO_KEYS["riverside_coachella"] == "rc"
    assert SCENARIO_KEYS["houston"] == "houston"


def test_level_prefixes():
    assert LEVEL_PREFIXES["r0"] == "r0"
    assert LEVEL_PREFIXES["r1"] == "r1_hydrology"
    assert LEVEL_PREFIXES["r2"] == "r2"


def test_bucket():
    assert BUCKET == "swarm-floodrsct-data"


def test_contracted_cells_count():
    # 5 nfip + 2 has_311 + 2 has_hwm = 9
    assert len(CONTRACTED_CELLS) == 9


def test_contracted_cells_includes_new_orleans_nfip():
    from georsct.experiment_audit.models import CellKey
    assert CellKey("new_orleans", "obs_nfip_event_claims") in CONTRACTED_CELLS


def test_contracted_cells_excludes_swfl_hwm():
    from georsct.experiment_audit.models import CellKey
    assert CellKey("southwest_florida", "obs_has_hwm") not in CONTRACTED_CELLS
