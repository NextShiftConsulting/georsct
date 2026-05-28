-- Policy-view queue queries
-- Ordering of ZCTAs lives ONLY here — not in certificates or F40 vectors.
-- ADR refs: ADR-033 D1, five-scenario delta §1.3 (ESCALATE verdict mapping)
-- Params: $scenario_id, $snapshot_t, $policy_view_queue_id

-- Build a new policy-view queue from F40 vectors (ordering declared here)
-- Example rule: risk_high_first_then_equity_q1
-- The ordering_rule field is the audit record; the ORDER BY clause implements it.
-- Params: $scenario_id, $snapshot_t, $policy_view, $ordering_rule, $created_by
WITH ranked AS (
    SELECT
        f.zcta_id,
        f.risk_level,
        f.reliability_action,
        f.population_context,
        f.access_fragility,
        f.equity_context,
        -- Floodcaster operator verdict (five-scenario delta §1.3)
        CASE
            WHEN f.reliability_action = 'suppress'  THEN 'SUPPRESS'
            WHEN f.reliability_action = 'escalate'  THEN 'ESCALATE'
            WHEN f.reliability_action = 'review'    THEN 'REVIEW'
            ELSE 'TRUST'
        END AS operator_verdict,
        -- Ordinal for risk_high_first_then_equity_q1 rule
        ROW_NUMBER() OVER (
            ORDER BY
                CASE f.risk_level
                    WHEN 'high'     THEN 1
                    WHEN 'moderate' THEN 2
                    ELSE 3
                END,
                CASE f.equity_context
                    WHEN 'q1' THEN 1
                    WHEN 'q2' THEN 2
                    WHEN 'q3' THEN 3
                    WHEN 'q4' THEN 4
                    ELSE 5
                END,
                f.zcta_id
        ) AS ordinal_position
    FROM f40_vector f
    WHERE f.scenario_id = $scenario_id
      AND f.snapshot_t  = $snapshot_t
      -- Suppress ZCTAs with reliability_action = suppress
      AND f.reliability_action != 'suppress'
)
SELECT
    zcta_id,
    ordinal_position,
    risk_level,
    reliability_action,
    operator_verdict,
    population_context,
    access_fragility,
    equity_context
FROM ranked
ORDER BY ordinal_position;

-- Retrieve an existing queue with its items
SELECT
    pvq.policy_view_queue_id,
    pvq.policy_view,
    pvq.ordering_rule,
    pvq.source,
    pvq.policy_view_version,
    pvq.note,
    pvq.created_by,
    pvq.created_at,
    pvqi.zcta_id,
    pvqi.ordinal_position,
    pvqi.recommended_action,
    pvqi.action_rationale
FROM policy_view_queue pvq
JOIN policy_view_queue_item pvqi USING (policy_view_queue_id)
WHERE pvq.policy_view_queue_id = $policy_view_queue_id
ORDER BY pvqi.ordinal_position;

-- Quick decision counts for a queue (for F5-style summary per queue)
SELECT
    pvqi.zcta_id,
    pvqi.ordinal_position,
    c.public_decision,
    c.gate_reached,
    f.risk_level,
    f.reliability_action,
    f.equity_context
FROM policy_view_queue pvq
JOIN policy_view_queue_item pvqi USING (policy_view_queue_id)
JOIN f40_vector f
    ON  f.scenario_id = pvq.scenario_id
    AND f.snapshot_t  = pvq.snapshot_t
    AND f.zcta_id     = pvqi.zcta_id
JOIN certificate c ON c.certificate_id = f.certificate_id
WHERE pvq.policy_view_queue_id = $policy_view_queue_id
ORDER BY pvqi.ordinal_position;
