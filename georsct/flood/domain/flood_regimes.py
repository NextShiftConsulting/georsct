"""Flood-specific deployment-domain regime registry.

Defines the five hydrometeorological regimes for FloodRSCT deployment-aligned
validation. Each regime declares hazard-eligibility rules over the ZCTA-event
universe; the framework classes (RegimeDomain, EligibilityRule) live in
rsct.validation.deployment_domains.

Column names in EligibilityRule are the data contract the feature/universe
table must satisfy. Thresholds and units are first-draft and MUST be
pre-committed before computing results.
"""

from __future__ import annotations

from rsct.validation.deployment_domains import EligibilityRule, RegimeDomain


FLOOD_REGIMES: dict[str, RegimeDomain] = {
    "houston_pluvial": RegimeDomain(
        regime_id="houston_pluvial",
        name="Houston urban pluvial",
        hazard_mechanism="rainfall ponding on impervious/low-drainage surfaces",
        label_source="NFIP claims (urban/insured sampling bias)",
        full_universe=True,
        note=(
            "Pluvial flooding is not confined to historically-claimed ZCTAs; the "
            "deployment domain is the full metro ZCTA universe for the event. The "
            "'universe' table MUST therefore be the full metro ZCTA set, not the "
            "claimed subset, or the shift is understated."
        ),
    ),
    "nola_levee": RegimeDomain(
        regime_id="nola_levee",
        name="New Orleans levee-protected",
        hazard_mechanism="basin inundation conditional on levee/pump performance",
        label_source="NFIP claims + stream/pump gauges",
        rules=(
            EligibilityRule(
                "protected_basin", "truthy", None,
                layer="levee/pump footprint",
            ),
            EligibilityRule(
                "hand_m", "le", 5.0,
                layer="HAND (HUC-conditioned)", units="m",
            ),
        ),
        combine="and",
        note=(
            "Blocked until levee footprint + HAND/HUC are ingested; "
            "resolve() will raise MissingHazardLayer."
        ),
    ),
    "riverside_flash": RegimeDomain(
        regime_id="riverside_flash",
        name="Riverside-Coachella canyon flash",
        hazard_mechanism="flash concentration in washes / canyon outlets / alluvial fans",
        label_source="sparse flash-event labels",
        rules=(
            EligibilityRule(
                "flow_accum", "ge", 1000.0,
                layer="flow accumulation / wash network", units="cells (pre-commit)",
            ),
            EligibilityRule(
                "hand_m", "le", 10.0,
                layer="HAND", units="m",
            ),
        ),
        combine="or",
        note="Either high upstream accumulation or low HAND marks a flash-eligible ZCTA.",
    ),
    "swfl_surge": RegimeDomain(
        regime_id="swfl_surge",
        name="SW Florida coastal surge",
        hazard_mechanism="storm-surge inundation conditional on storm category",
        label_source="claims + coastal surge observations",
        rules=(
            EligibilityRule(
                "surge_inundated", "truthy", None,
                layer="SLOSH/coastal DEM envelope",
            ),
        ),
        combine="and",
        note=(
            "surge_inundated is category-conditional; materialize it per storm "
            "category from SLOSH MEOW/MOM or a DEM elevation band before resolving."
        ),
    ),
    "nyc_sewershed": RegimeDomain(
        regime_id="nyc_sewershed",
        name="NYC/NJ urban sewer-shed",
        hazard_mechanism="combined-sewer-overflow surcharge within sewershed catchments",
        label_source="311 complaints (reporting bias)",
        rules=(
            EligibilityRule(
                "sewershed_id", "notnull", None,
                layer="CSO sewershed boundaries (urban HUC-like)",
            ),
        ),
        combine="and",
        note=(
            "Blocked until sewershed boundaries are ingested; "
            "resolve() will raise MissingHazardLayer."
        ),
    ),
}


def get_flood_domain(regime_id: str) -> RegimeDomain:
    """Look up a flood regime by ID."""
    if regime_id not in FLOOD_REGIMES:
        raise KeyError(
            f"unknown flood regime '{regime_id}'; "
            f"known: {sorted(FLOOD_REGIMES)}"
        )
    return FLOOD_REGIMES[regime_id]
