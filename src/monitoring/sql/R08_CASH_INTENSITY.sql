-- R08: cash intensity relative to the declared customer profile.
--
-- Peer-relative by construction. An absolute cash-share threshold alerts every
-- restaurant and every car wash every month, which is how a monitoring
-- programme ends up with a queue nobody can work. The comparison is against the
-- 75th percentile of the customer's own declared behavioural segment, so a cash
-- business is only flagged when it is cash-heavy even for a cash business.
WITH p AS (
    SELECT
        MAX(CASE WHEN param_name = 'min_cash_share'  THEN param_value END) AS min_cash_share,
        MAX(CASE WHEN param_name = 'min_cash_amount' THEN param_value END) AS min_cash_amount,
        MAX(CASE WHEN param_name = 'min_txn_count'   THEN param_value END) AS min_txn_count
    FROM rule_params WHERE rule_id = 'R08_CASH_INTENSITY'
),
monthly AS (
    SELECT
        t.customer_id,
        c.behaviour_segment,
        strftime(t.txn_ts, '%Y-%m') AS period,
        SUM(CASE WHEN t.channel = 'CASH' THEN t.amount ELSE 0.0 END) AS cash_amount,
        SUM(t.amount)                                               AS total_amount,
        COUNT(*)                                                    AS txn_count
    FROM transactions t
    JOIN customers c USING (customer_id)
    GROUP BY 1, 2, 3
),
with_share AS (
    SELECT *, cash_amount / NULLIF(total_amount, 0) AS cash_share FROM monthly
),
peer AS (
    SELECT
        behaviour_segment,
        QUANTILE_CONT(cash_share, 0.75) AS peer_p75
    FROM with_share
    GROUP BY 1
)
SELECT
    'R08_CASH_INTENSITY' AS rule_id,
    w.customer_id,
    w.period,
    w.cash_share  AS metric_value,
    w.cash_amount AS alert_amount,
    w.txn_count
FROM with_share w
JOIN peer pe USING (behaviour_segment)
CROSS JOIN p
WHERE w.txn_count >= p.min_txn_count
  AND w.cash_amount >= p.min_cash_amount
  AND w.cash_share >= p.min_cash_share
  AND w.cash_share >= pe.peer_p75
