-- F6: list_scenarios()
-- Returns supported scenarios with metadata.
-- Error: PORTFOLIO_UNAVAILABLE if fewer than 3 core scenarios are present.
-- Params: none
-- ADR refs: five-scenario delta §2 F6

SELECT
    s.scenario_id,
    s.display_name,
    s.anchor_storm,
    s.region,
    s.flood_archetype,
    s.primary_decision,
    s.build_phase,
    s.total_nfip_claims,
    s.total_loss_usd,
    s.affected_zctas,
    s.peak_extent_km2,
    s.fatality_count,
    -- Available snapshot hours as a sorted list
    array_agg(ss.snapshot_t ORDER BY ss.snapshot_t DESC) AS snapshot_hours,
    -- In-scope ZCTA count
    COUNT(DISTINCT sz.zcta_id)                            AS in_scope_zcta_count
FROM scenarios s
JOIN scenario_snapshots ss USING (scenario_id)
LEFT JOIN scenario_zctas sz USING (scenario_id)
GROUP BY
    s.scenario_id, s.display_name, s.anchor_storm, s.region,
    s.flood_archetype, s.primary_decision, s.build_phase,
    s.total_nfip_claims, s.total_loss_usd, s.affected_zctas,
    s.peak_extent_km2, s.fatality_count
ORDER BY s.build_phase, s.scenario_id;
