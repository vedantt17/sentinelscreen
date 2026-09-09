-- R05: rapid throughput on a newly onboarded account.
--
-- New accounts have no behavioural baseline, so the usual peer-relative and
-- self-relative scenarios are blind to them. This scenario covers that gap for
-- the onboarding window only, which is why the account-age filter is a hard
-- predicate rather than a scoring feature.
WITH p AS (
    SELECT
        MAX(CASE WHEN param_name = 'account_age_days'  THEN param_value END) AS account_age_days,
        MAX(CASE WHEN param_name = 'min_txn_count'     THEN param_value END) AS min_txn_count,
        MAX(CASE WHEN param_name = 'min_total_amount'  THEN param_value END) AS min_total_amount
    FROM rule_params WHERE rule_id = 'R05_NEW_ACCOUNT_BURST'
),
early AS (
    SELECT
        t.customer_id,
        strftime(t.txn_ts, '%Y-%m') AS period,
        t.amount
    FROM transactions t
    JOIN customers c USING (customer_id)
    CROSS JOIN p
    WHERE date_diff('day', c.account_open_date, t.txn_ts) BETWEEN 0 AND p.account_age_days
)
SELECT
    'R05_NEW_ACCOUNT_BURST' AS rule_id,
    e.customer_id,
    e.period,
    SUM(e.amount) AS metric_value,
    SUM(e.amount) AS alert_amount,
    COUNT(*)      AS txn_count
FROM early e, p
GROUP BY 1, 2, 3, p.min_txn_count, p.min_total_amount
HAVING COUNT(*) >= p.min_txn_count
   AND SUM(e.amount) >= p.min_total_amount
