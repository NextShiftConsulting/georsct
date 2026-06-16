"""GeoRSCT-X gearbox — maps certificate weakness to expert admission mode.

The gearbox reads the certificate's ranked weakness vector and selects
the appropriate gear (intervention mode). Experts are then ranked by
expected compatibility gain within that gear.

Import rule: depends only on contracts, provenance, and ports.
"""

from __future__ import annotations

from georsct.contracts.task_contract import TaskContract
from georsct.ports.spatial_expert import SpatialExpert
from georsct.provenance.trace import ExecutionCertificate, Weakness


# ---------------------------------------------------------------------------
# Gear vocabulary
# ---------------------------------------------------------------------------

GEAR_FOR_WEAKNESS: dict[str, str] = {
    "low_target_coverage": "G1_observe",
    "under_supported_geometry": "G2_enrich",
    "residual_spatial_structure": "G3_diagnose",  # diagnose first, NOT enrich
    "leakage": "R_reverse",
    "none": "G0_base",
}


def select_gear(cert: ExecutionCertificate) -> str:
    """Select gear from the primary (highest severity) weakness."""
    wv = cert.weakness_vector()
    if not wv:
        return "G0_base"
    return GEAR_FOR_WEAKNESS.get(wv[0].weakness_type, "G0_base")


def select_gear_from_weakness(weakness: Weakness) -> str:
    """Select gear from a specific weakness."""
    return GEAR_FOR_WEAKNESS.get(weakness.weakness_type, "G0_base")


# ---------------------------------------------------------------------------
# Expert ranking (not first-match)
# ---------------------------------------------------------------------------

def rank_experts(
    candidates: list[SpatialExpert],
    cert: ExecutionCertificate,
    contract: TaskContract,
    already_run: frozenset[str] = frozenset(),
) -> list[SpatialExpert]:
    """Rank admissible experts by expected compatibility gain.

    Steps:
      1. Hard filter: remove already-run and non-admissible experts.
      2. Soft rank: sort by expected_delta descending.

    Args:
        candidates: All registered experts.
        cert: Current certificate state.
        contract: The task contract.
        already_run: Expert IDs that have already been activated.

    Returns:
        Ranked list of admissible experts (best first).
    """
    admissible = [
        e for e in candidates
        if e.expert_id not in already_run
        and e.admissible_for(contract, cert)
    ]
    admissible.sort(
        key=lambda e: e.expected_delta(contract.geometry),
        reverse=True,
    )
    return admissible
