-- F4: select_zcta(scenario_id, snapshot_t, zcta_id, embedding_arm)
-- Full certificate drill-down for one ZCTA:
--   certificate + gate results + reason codes + infrastructure evidence + alternative arms
-- Params: $scenario_id, $snapshot_t, $zcta_id, $embedding_arm
-- ADR refs: ADR-030 D1/D4, ADR-033 §3.2 gate narrative, five-scenario delta §4

-- 1. Core certificate
SELECT
    c.certificate_id,
    c.scenario_id,
    c.snapshot_t,
    c.zcta_id,
    c.embedding_arm,
    c.target,
    c.snapshot_year,

    c.r, c.s_sup, c.n, c.alpha,
    c.kappa_compat, c.kappa_h, c.kappa_l, c.kappa_int, c.kappa_modal_min,
    c.sigma, c.n_ceiling, c.n_ceiling_unavailable_reason,

    c.public_decision,
    c.public_decision_source,
    c.gate_reached,
    c.gate_reason,

    c.policy_id,
    c.certificate_schema_version,
    c.dataset_version,
    c.live_sensors_used,
    c.computed_at
FROM certificate c
WHERE c.scenario_id   = $scenario_id
  AND c.snapshot_t    = $snapshot_t
  AND c.zcta_id       = $zcta_id
  AND c.embedding_arm = $embedding_arm;

-- 2. Reason codes
SELECT rc.reason_code, rc.sort_order
FROM certificate_reason_code rc
JOIN certificate c USING (certificate_id)
WHERE c.scenario_id   = $scenario_id
  AND c.snapshot_t    = $snapshot_t
  AND c.zcta_id       = $zcta_id
  AND c.embedding_arm = $embedding_arm
ORDER BY rc.sort_order;

-- 3. Gate results (decision walkthrough)
SELECT
    gr.gate_name,
    gr.gate_version,
    gr.gate_decision,
    gr.active_metric,
    gr.active_value,
    gr.threshold_metric,
    gr.threshold_value,
    gr.passed,
    gr.gate_reason,
    gr.evaluated_at
FROM gate_result gr
JOIN certificate c USING (certificate_id)
WHERE c.scenario_id   = $scenario_id
  AND c.snapshot_t    = $snapshot_t
  AND c.zcta_id       = $zcta_id
  AND c.embedding_arm = $embedding_arm
ORDER BY gr.evaluated_at;

-- 4. Infrastructure evidence (scenario-discriminated §4)
SELECT ie.*
FROM infrastructure_evidence ie
JOIN certificate c USING (certificate_id)
WHERE c.scenario_id   = $scenario_id
  AND c.snapshot_t    = $snapshot_t
  AND c.zcta_id       = $zcta_id
  AND c.embedding_arm = $embedding_arm;

-- 5. Alternative arms
SELECT
    aa.embedding_arm,
    aa.r, aa.s_sup, aa.n, aa.alpha,
    aa.kappa_compat, aa.sigma, aa.n_ceiling,
    aa.public_decision, aa.gate_reached,
    aa.would_change_decision
FROM alternative_arm aa
JOIN certificate c USING (certificate_id)
WHERE c.scenario_id   = $scenario_id
  AND c.snapshot_t    = $snapshot_t
  AND c.zcta_id       = $zcta_id
  AND c.embedding_arm = $embedding_arm
ORDER BY aa.embedding_arm;

-- 6. Modal source ETags (audit block per ADR-030 D4)
SELECT mse.source_name, mse.etag, mse.source_tier
FROM modal_source_etag mse
JOIN certificate c USING (certificate_id)
WHERE c.scenario_id   = $scenario_id
  AND c.snapshot_t    = $snapshot_t
  AND c.zcta_id       = $zcta_id
  AND c.embedding_arm = $embedding_arm
ORDER BY mse.source_tier, mse.source_name;
