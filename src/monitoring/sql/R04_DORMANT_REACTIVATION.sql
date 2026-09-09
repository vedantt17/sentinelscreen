-- R04: dormant account reactivation.
--
-- Dormancy is measured against the account's own prior activity, not against a
-- portfolio average, because a low-activity account is not dormant and alerting
-- on it would drown the queue. The gap is taken from the previous transaction
-- of any kind, so a single small card payment breaks dormancy exactly as a
-- reviewer would expect.
WITH p AS (
    SELECT
        MAX(CASE WHEN param_name = 'dormancy_days'            THEN param_value END) AS dormancy_days,
        MAX(CASE WHEN param_name = 'min_reactivation_amount'  THEN param_value END) AS min_reactivation_amount,
        MAX(CASE WHEN param_name = 'min_reactivation_txns'    THEN param_value END) AS min_reactivation_txns
    FROM rule_params WHERE rule_id = 'R04_DORMANT_REACTIVATION'
),
gaps AS (
    SELECT
        t.customer_id,
        t.txn_ts,
        t.amount,
        strftime(t.txn_ts, '%Y-%m') AS period,
        date_diff(
            'day',
            LAG(t.txn_ts) OVER (PARTITION BY t.customer_id ORDER BY t.txn_ts),
            t.txn_ts
        ) AS days_since_prior
    FROM transactions t
),
reactivations AS (
    SELECT g.customer_id, g.period
    FROM gaps g, p
    WHERE g.days_since_prior >= p.dormancy_days
    GROUP BY 1, 2
)
SELECT
    'R04_DORMANT_REACTIVATION' AS rule_id,
    r.customer_id,
    r.period,
    SUM(g.amount)  AS metric_value,
    SUM(g.amount)  AS alert_amount,
    COUNT(*)       AS txn_count
FROM reactivations r
JOIN gaps g ON g.customer_id = r.customer_id AND g.period = r.period
CROSS JOIN p
GROUP BY 1, 2, 3, p.min_reactivation_amount, p.min_reactivation_txns
HAVING SUM(g.amount) >= p.min_reactivation_amount
   AND COUNT(*) >= p.min_reactivation_txns
