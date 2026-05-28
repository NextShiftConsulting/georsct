-- F5: summarize_session(session_id)
-- Computes precision/recall/F1 for the active session.
-- Reads session_decision (written by every F3 call).
-- Precondition: list_zctas() called at least once in this session.
-- Params: $session_id
-- ADR refs: functional contract F5

WITH decisions AS (
    SELECT
        sd.public_decision,
        sd.actually_flooded,
        sd.zcta_id
    FROM session_decision sd
    WHERE sd.session_id = $session_id
),
counts AS (
    SELECT
        COUNT(*)                                                             AS total_zctas,
        COUNT(*) FILTER (WHERE public_decision = 'EXECUTE')                 AS zctas_targeted,
        COUNT(*) FILTER (WHERE public_decision = 'EXECUTE' AND actually_flooded)  AS zctas_that_flooded,
        COUNT(*) FILTER (WHERE public_decision = 'EXECUTE' AND NOT actually_flooded) AS zctas_that_didnt,
        COUNT(*) FILTER (WHERE actually_flooded AND public_decision != 'EXECUTE')    AS flooded_zctas_missed
    FROM decisions
),
metrics AS (
    SELECT *,
        CASE WHEN zctas_targeted > 0
             THEN zctas_that_flooded::DOUBLE / zctas_targeted
             ELSE NULL END                              AS precision_at_execute,
        CASE WHEN (zctas_that_flooded + flooded_zctas_missed) > 0
             THEN zctas_that_flooded::DOUBLE / (zctas_that_flooded + flooded_zctas_missed)
             ELSE NULL END                              AS recall_at_execute
    FROM counts
)
SELECT
    total_zctas,
    zctas_targeted,
    zctas_that_flooded,
    zctas_that_didnt,
    flooded_zctas_missed,
    precision_at_execute,
    recall_at_execute,
    CASE WHEN precision_at_execute IS NOT NULL
              AND recall_at_execute IS NOT NULL
              AND (precision_at_execute + recall_at_execute) > 0
         THEN 2 * precision_at_execute * recall_at_execute
              / (precision_at_execute + recall_at_execute)
         ELSE NULL END                                  AS f1_at_execute
FROM metrics;
