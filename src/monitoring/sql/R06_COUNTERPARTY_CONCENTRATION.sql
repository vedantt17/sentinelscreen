-- R06: counterparty concentration.
--
-- Concentration is measured as the largest single counterparty's share of the
-- month's value rather than a Herfindahl index over all counterparties: the
-- typology being targeted is one dominant conduit, and HHI would also fire on a
-- customer with three equal partners, which is ordinary commercial behaviour.
WITH p AS (
    SELECT
        MAX(CASE WHEN param_name = 'min_concentration' THEN param_value END) AS min_concentration,
        MAX(CASE WHEN param_name = 'min_txn_count'     THEN param_value END) AS min_txn_count,
        MAX(CASE WHEN param_name = 'min_total_amount'  THEN param_value END) AS min_total_amount
    FROM rule_params WHERE rule_id = 'R06_COUNTERPARTY_CONCENTRATION'
),
by_counterparty AS (
    SELECT
        t.customer_id,
        strftime(t.txn_ts, '%Y-%m')            AS period,
        COALESCE(t.counterparty_name, 'UNKNOWN') AS counterparty,
        SUM(t.amount)                          AS cp_amount,
        COUNT(*)                               AS cp_count
    FROM transactions t
    WHERE t.counterparty_name IS NOT NULL
    GROUP BY 1, 2, 3
),
totals AS (
    SELECT
        customer_id,
        period,
        SUM(cp_amount) AS total_amount,
        SUM(cp_count)  AS total_count,
        MAX(cp_amount) AS top_amount
    FROM by_counterparty
    GROUP BY 1, 2
)
SELECT
    'R06_COUNTERPARTY_CONCENTRATION' AS rule_id,
    t.customer_id,
    t.period,
    t.top_amount / NULLIF(t.total_amount, 0) AS metric_value,
    t.top_amount                             AS alert_amount,
    t.total_count                            AS txn_count
FROM totals t, p
WHERE t.total_count >= p.min_txn_count
  AND t.total_amount >= p.min_total_amount
  AND t.top_amount / NULLIF(t.total_amount, 0) >= p.min_concentration
