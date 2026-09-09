-- R01: cash structuring beneath the CTR reporting threshold.
--
-- Deposits are counted inside a rolling window rather than per calendar month:
-- a launderer splitting 40,000 dollars does not respect month boundaries, and a
-- calendar-month count would miss a run that straddles the 31st.
WITH p AS (
    SELECT
        MAX(CASE WHEN param_name = 'lower_bound'    THEN param_value END) AS lower_bound,
        MAX(CASE WHEN param_name = 'upper_bound'    THEN param_value END) AS upper_bound,
        MAX(CASE WHEN param_name = 'min_txn_count'  THEN param_value END) AS min_txn_count,
        MAX(CASE WHEN param_name = 'window_days'    THEN param_value END) AS window_days
    FROM rule_params WHERE rule_id = 'R01_STRUCTURING'
),
banded AS (
    SELECT t.customer_id, t.txn_ts, t.amount
    FROM transactions t, p
    WHERE t.channel = 'CASH'
      AND t.direction = 'CREDIT'
      AND t.amount BETWEEN p.lower_bound AND p.upper_bound
),
windowed AS (
    SELECT
        a.customer_id,
        strftime(a.txn_ts, '%Y-%m')                AS period,
        COUNT(*)                                   AS window_count,
        SUM(b.amount)                              AS window_amount
    FROM banded a
    JOIN banded b
      ON b.customer_id = a.customer_id
     AND b.txn_ts >= a.txn_ts
     AND date_diff('day', a.txn_ts, b.txn_ts) < (SELECT window_days FROM p)
    GROUP BY 1, 2
)
-- One alert per customer-month, reporting the single strongest window and the
-- amount that belongs to *that* window. Taking MAX() of the count and MAX() of
-- the amount independently would pair a count from one window with a total from
-- another, and the alert's own detail would then be internally inconsistent.
SELECT
    'R01_STRUCTURING'      AS rule_id,
    w.customer_id,
    w.period,
    w.window_count::DOUBLE AS metric_value,
    w.window_amount        AS alert_amount,
    w.window_count         AS txn_count
FROM windowed w, p
WHERE w.window_count >= p.min_txn_count
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY w.customer_id, w.period
    ORDER BY w.window_count DESC, w.window_amount DESC
) = 1
