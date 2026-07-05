"""Use case: Gate 3B spatial recoverability decision.

Control layer (not measurement). Emits EXECUTE or RE_ENCODE based on
forward accuracy and spatial_recoverability thresholds.

Moved from georsct.domain.spatial_recoverability to maintain P4 separation:
domain = measurement only, application = decisions.
"""


def gate_3b_decision(
    forward_score: float,
    spatial_recoverability: float,
    forward_floor: float = 0.0,
    reconstruct_floor: float = 0.3,
) -> str:
    """Gate 3B: spatial recoverability.

    Forward accuracy is not enough. A representation must also preserve
    enough backward structure to recover the geography it claims to
    reason over.

    Fires only on the taxi-map quadrant: high forward, low recoverability.

    Returns:
        "EXECUTE" or "RE_ENCODE"
    """
    if forward_score >= forward_floor and spatial_recoverability < reconstruct_floor:
        return "RE_ENCODE"
    return "EXECUTE"
