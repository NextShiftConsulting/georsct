-- F3: list_zctas(scenario_id, snapshot_t, embedding_arm)
-- Returns all in-scope ZCTAs with their certificates for a given scenario/snapshot.
-- Ordered by zcta_id (alphabetic) — NOT by kappa_compat.
-- Ordering by kappa_compat would constitute an implicit scalar ranking (ADR-033 D1).
-- Use policy_view_queue for any operator-facing prioritisation.
-- Params: $scenario_id, $snapshot_t, $embedding_arm (default: 'graphsage_v1')
-- ADR refs: ADR-029 D1, ADR-033 D1, ADR-034 D1/D2/D3, five-scenario delta §I9

SELECT
    sz.zcta_id,
    sz.state_fips,
    sz.county_fips,
    sz.huc8_id,

    -- Certificate (may be null if not yet computed for this ZCTA)
    c.certificate_id,

    -- Core simplex
    c.r,
    c.s_sup,
    c.n,
    c.alpha,

    -- Kappa (ADR-034 D1 canonical names)
    c.kappa_compat,
    c.kappa_modal_min,

    -- Geo-specific
    c.sigma,
    c.n_ceiling,
    c.n_ceiling_unavailable_reason,

    -- Public projection (ADR-034 D2)
    c.public_decision,
    c.gate_reached,
    c.gate_reason,

    -- Reason codes as array (from normalised table)
    array_agg(rc.reason_code ORDER BY rc.sort_order) FILTER (WHERE rc.reason_code IS NOT NULL)
        AS reason_codes,

    -- Cert unavailable signal (I4: returned with certificate = null)
    CASE WHEN c.certificate_id IS NULL THEN 'NOT_COMPUTED' ELSE NULL END
        AS unavailable_reason

FROM scenario_zctas sz
LEFT JOIN certificate c
    ON  c.scenario_id    = sz.scenario_id
    AND c.zcta_id        = sz.zcta_id
    AND c.snapshot_t     = $snapshot_t
    AND c.embedding_arm  = $embedding_arm
LEFT JOIN certificate_reason_code rc
    ON  rc.certificate_id = c.certificate_id
WHERE sz.scenario_id = $scenario_id
GROUP BY
    sz.zcta_id, sz.state_fips, sz.county_fips, sz.huc8_id,
    c.certificate_id, c.r, c.s_sup, c.n, c.alpha,
    c.kappa_compat, c.kappa_modal_min,
    c.sigma, c.n_ceiling, c.n_ceiling_unavailable_reason,
    c.public_decision, c.gate_reached, c.gate_reason
ORDER BY sz.zcta_id;  -- alphabetic, not scalar-ranked
