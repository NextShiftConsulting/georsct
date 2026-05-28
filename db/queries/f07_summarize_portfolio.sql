-- F7: summarize_portfolio(session_id)
-- Cross-scenario reliability statistics. The "benchmark not demo" function.
-- Precondition: at least one snapshot of each phase-1 scenario evaluated.
-- Params: $session_id
-- Error: INSUFFICIENT_SCENARIOS if fewer than 3 scenarios have decisions in session.
-- ADR refs: five-scenario delta §2 F7

-- Per-scenario metrics
WITH per_scenario AS (
    SELECT
        sd.scenario_id,
        COUNT(DISTINCT sd.zcta_id)                                              AS zctas_evaluated,
        COUNT(*) FILTER (WHERE sd.public_decision = 'EXECUTE')                  AS n_execute,
        COUNT(*) FILTER (WHERE sd.public_decision = 'CAUTION')                  AS n_caution,
        COUNT(*) FILTER (WHERE sd.public_decision = 'REFUSE')                   AS n_refuse,
        -- precision@EXECUTE
        AVG(CASE WHEN sd.public_decision = 'EXECUTE' AND sd.actually_flooded     THEN 1.0
                 WHEN sd.public_decision = 'EXECUTE' AND NOT sd.actually_flooded THEN 0.0
                 ELSE NULL END)                                                  AS precision_at_execute,
        -- recall@EXECUTE
        AVG(CASE WHEN sd.actually_flooded AND sd.public_decision = 'EXECUTE'     THEN 1.0
                 WHEN sd.actually_flooded AND sd.public_decision != 'EXECUTE'    THEN 0.0
                 ELSE NULL END)                                                  AS recall_at_execute,
        -- ground truth alignment: fraction of decisions that matched outcome
        AVG(CASE WHEN (sd.public_decision = 'EXECUTE' AND sd.actually_flooded)
                   OR (sd.public_decision != 'EXECUTE' AND NOT sd.actually_flooded)
                 THEN 1.0 ELSE 0.0 END)                                         AS ground_truth_alignment,
        -- dominant failure gate: most frequent gate_reached among non-EXECUTE decisions
        mode(c.gate_reached) FILTER (WHERE sd.public_decision != 'EXECUTE')     AS dominant_failure_gate
    FROM session_decision sd
    JOIN certificate c USING (certificate_id)
    WHERE sd.session_id = $session_id
    GROUP BY sd.scenario_id
),
scenario_count AS (
    SELECT COUNT(DISTINCT scenario_id) AS n_scenarios FROM per_scenario
),
-- Schema invariance: check that all certs share the same non-null field set
schema_invariance AS (
    SELECT
        COUNT(DISTINCT sd.scenario_id)                                AS n_scenarios_with_certs,
        -- All certs should have the six core fields non-null (I7)
        BOOL_AND(c.r IS NOT NULL AND c.s_sup IS NOT NULL AND c.n IS NOT NULL
                 AND c.alpha IS NOT NULL AND c.kappa_compat IS NOT NULL
                 AND c.sigma IS NOT NULL)                             AS core_fields_complete,
        -- Gate ordering invariance (I8): all gate_reached values should come from the same set
        COUNT(DISTINCT c.gate_reached) FILTER (WHERE c.gate_reached IS NOT NULL) AS distinct_gate_reached_values
    FROM session_decision sd
    JOIN certificate c USING (certificate_id)
    WHERE sd.session_id = $session_id
)
SELECT
    (SELECT n_scenarios FROM scenario_count)           AS scenarios_evaluated,
    (SELECT COUNT(*) FROM scenarios WHERE build_phase = 'core') AS scenarios_available,
    -- Per-scenario rows
    json_agg(json_object(
        'scenario_id',           ps.scenario_id,
        'n_execute',             ps.n_execute,
        'n_caution',             ps.n_caution,
        'n_refuse',              ps.n_refuse,
        'precision_at_execute',  ps.precision_at_execute,
        'recall_at_execute',     ps.recall_at_execute,
        'ground_truth_alignment',ps.ground_truth_alignment,
        'dominant_failure_gate', ps.dominant_failure_gate
    ) ORDER BY ps.scenario_id)                         AS per_scenario,
    -- Schema invariance block (I7, I8)
    (SELECT core_fields_complete FROM schema_invariance)   AS schema_fields_identical,
    (SELECT n_scenarios_with_certs FROM schema_invariance) AS scenarios_with_certs
FROM per_scenario ps;
