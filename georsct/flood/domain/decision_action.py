"""Decision-to-action mapping at the adapter boundary.

Maps RSCT gate outcomes to floodcaster-specific operational guidance.
Per ADR-029/034: domain produces PublicDecision, this module translates
to actionable text for dashboards/reports.
"""

from georsct.domain.certificate import PublicDecision


GUIDANCE = {
    PublicDecision.EXECUTE: (
        "Model output is operationally usable. Spatial quality, compatibility, "
        "and turbulence gates all pass. Proceed with certificate issuance."
    ),
    PublicDecision.CAUTION: (
        "Model output may be used with documented limitations. One or more "
        "quality gates show marginal performance. Review residual hotspots "
        "and evidence warnings before operational use."
    ),
    PublicDecision.REFUSE: (
        "Model output should NOT be used operationally. Quality gates fail. "
        "Investigate spatial coverage gaps, data source issues, or model "
        "misspecification before re-running."
    ),
}


def get_operational_guidance(decision: PublicDecision) -> str:
    """Translate certificate decision to operational guidance text."""
    return GUIDANCE.get(decision, GUIDANCE[PublicDecision.REFUSE])
