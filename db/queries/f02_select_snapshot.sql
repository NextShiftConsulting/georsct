-- F2: select_snapshot(scenario_id, snapshot_t)
-- Validates that snapshot_t is available for the given scenario.
-- Params: $scenario_id, $snapshot_t
-- ADR refs: five-scenario delta §1.2, functional contract F2

SELECT
    ss.scenario_id,
    ss.snapshot_t,
    COUNT(c.certificate_id) AS precomputed_cert_count
FROM scenario_snapshots ss
LEFT JOIN certificate c
    ON  c.scenario_id = ss.scenario_id
    AND c.snapshot_t  = ss.snapshot_t
WHERE ss.scenario_id = $scenario_id
  AND ss.snapshot_t  = $snapshot_t
GROUP BY ss.scenario_id, ss.snapshot_t;

-- Returns 0 rows → raise SNAPSHOT_NOT_AVAILABLE
-- precomputed_cert_count = 0 → certs not yet loaded; app layer should fetch from DynamoDB
