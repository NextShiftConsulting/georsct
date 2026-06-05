-- Ablation queries: compare a ZCTA's certificate across embedding arms
-- Used by POST /certify/geo/ablation (ADR-030 §endpoint definitions)
-- Scenario-specific stress tests are parameterised via $perturbation_arm
-- Params: $scenario_id, $snapshot_t, $zcta_id
-- ADR refs: ADR-030, five-scenario delta §3.3

-- All arms for a given ZCTA/scenario/snapshot, with change-of-decision flag
-- and the kappa_compat / public_decision delta vs the default arm (graphsage_v1)
WITH default_arm AS (
    SELECT certificate_id, kappa_compat, public_decision
    FROM certificate
    WHERE scenario_id   = $scenario_id
      AND snapshot_t    = $snapshot_t
      AND zcta_id       = $zcta_id
      AND embedding_arm = 'graphsage_v1'
)
SELECT
    c.embedding_arm,
    c.r, c.s_sup, c.n, c.alpha,
    c.kappa_compat,
    c.kappa_compat - d.kappa_compat            AS kappa_compat_delta,
    c.kappa_modal_min,
    c.sigma, c.task_residual_floor,
    c.public_decision,
    c.gate_reached,
    c.gate_reason,
    (c.public_decision != d.public_decision)   AS would_change_decision,
    c.live_sensors_used,
    c.computed_at
FROM certificate c, default_arm d
WHERE c.scenario_id   = $scenario_id
  AND c.snapshot_t    = $snapshot_t
  AND c.zcta_id       = $zcta_id
ORDER BY c.embedding_arm;

-- Scenario-specific stress test summary
-- Shows how many ZCTAs change decision when the perturbation arm is used
-- Params: $scenario_id, $snapshot_t, $perturbation_arm
WITH baseline AS (
    SELECT zcta_id, public_decision AS baseline_decision
    FROM certificate
    WHERE scenario_id   = $scenario_id
      AND snapshot_t    = $snapshot_t
      AND embedding_arm = 'graphsage_v1'
),
perturbed AS (
    SELECT zcta_id, public_decision AS perturbed_decision, gate_reached
    FROM certificate
    WHERE scenario_id   = $scenario_id
      AND snapshot_t    = $snapshot_t
      AND embedding_arm = $perturbation_arm
)
SELECT
    COUNT(*)                                                               AS total_zctas,
    COUNT(*) FILTER (WHERE b.baseline_decision != p.perturbed_decision)    AS decision_changes,
    COUNT(*) FILTER (WHERE b.baseline_decision = 'EXECUTE'
                      AND p.perturbed_decision != 'EXECUTE')               AS execute_to_other,
    COUNT(*) FILTER (WHERE b.baseline_decision != 'EXECUTE'
                      AND p.perturbed_decision = 'EXECUTE')                AS other_to_execute,
    ROUND(COUNT(*) FILTER (WHERE b.baseline_decision != p.perturbed_decision)::DOUBLE
          / NULLIF(COUNT(*), 0), 4)                                        AS instability_rate
FROM baseline b
JOIN perturbed p USING (zcta_id);
