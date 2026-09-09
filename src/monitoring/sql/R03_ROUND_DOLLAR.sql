-- R03: round-dollar layering.
--
-- Commercial payments settle invoices and therefore carry cents. A book of
-- business that is overwhelmingly round-valued is either payroll (which the
-- min_amount floor and the count threshold exist to exclude) or value being
-- moved for its own sake.
WITH p AS (
    SELECT
        MAX(CASE WHEN param_name = 'min_round_share' THEN param_value END) AS min_round_share,
        MAX(CASE WHEN param_name = 'min_txn_count'   THEN param_value END) AS min_txn_count,
        MAX(CASE WHEN param_name = 'min_amount'      THEN param_value END) AS min_amount
    FROM rule_params WHERE rule_id = 'R03_ROUND_DOLLAR'
),
scoped AS (
    SELECT
        t.customer_id,
        strftime(t.txn_ts, '%Y-%m') AS period,
        t.amount,
        CASE WHEN t.amount = ROUND(t.amount / 1000.0) * 1000.0 THEN 1 ELSE 0 END AS is_round
    FROM transactions t, p
    WHERE t.amount >= p.min_amount
      AND t.channel IN ('WIRE', 'ACH')
)
SELECT
    'R03_ROUND_DOLLAR' AS rule_id,
    s.customer_id,
    s.period,
    SUM(s.is_round)::DOUBLE / COUNT(*)                     AS metric_value,
    SUM(CASE WHEN s.is_round = 1 THEN s.amount ELSE 0 END) AS alert_amount,
    COUNT(*)                                               AS txn_count
FROM scoped s, p
GROUP BY 1, 2, 3, p.min_round_share, p.min_txn_count
HAVING COUNT(*) >= p.min_txn_count
   AND SUM(s.is_round)::DOUBLE / COUNT(*) >= p.min_round_share
