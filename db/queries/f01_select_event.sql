-- F1: select_event(scenario_id)
-- Validates scenario exists and returns its snapshot menu.
-- Invalidates all cached cert results from prior event (handled in app layer).
-- Params: $scenario_id
-- ADR refs: five-scenario delta §1.1, functional contract F1

SELECT
    s.scenario_id,
    s.display_name,
    s.anchor_storm,
    s.flood_archetype,
    s.build_phase,
    array_agg(ss.snapshot_t ORDER BY ss.snapshot_t DESC) AS available_snapshots
FROM scenarios s
JOIN scenario_snapshots ss USING (scenario_id)
WHERE s.scenario_id = $scenario_id
GROUP BY s.scenario_id, s.display_name, s.anchor_storm,
         s.flood_archetype, s.build_phase;

-- Returns 0 rows → raise UNKNOWN_SCENARIO
-- Returns a phase-2 scenario but only phase-1 is deployed → raise SCENARIO_NOT_DEPLOYED
-- (deployment-phase check is done in the Python wrapper, not in SQL)
