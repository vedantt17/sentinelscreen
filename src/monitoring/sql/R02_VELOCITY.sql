-- R02: rapid movement of funds (pass-through / funnel behaviour).
--
-- The signal is not volume but *retention*: money that arrives and leaves at
-- near parity inside a short holding period has no economic purpose for the
-- account holder. Debits are attributed to a credit only when they fall inside
-- the holding window, so a routine monthly outflow does not count.
WITH p AS (
    SELECT
        MAX(CASE WHEN param_name = 'max_hold_hours'        THEN param_value END) AS max_hold_hours,
        MAX(CASE WHEN param_name = 'min_passthrough_ratio' THEN param_value END) AS min_passthrough_ratio,
        MAX(CASE WHEN param_name = 'min_amount'            THEN param_value END) AS min_amount
    FROM rule_params WHERE rule_id = 'R02_VELOCITY'
),
credits AS (
    -- The window is materialised as an absolute expiry timestamp rather than
    -- evaluated with date_diff() in the join predicate. Two reasons, both
    -- load-bearing: DuckDB cannot evaluate a scalar subquery inside a LEFT JOIN
    -- condition at all, and wrapping the join key in a function call defeats
    -- its range-join optimiser, collapsing the join to a nested loop. On the
    -- 500k-row feed that difference was 382 seconds versus under two.
    SELECT
        t.customer_id, t.txn_ts, t.amount,
        strftime(t.txn_ts, '%Y-%m') AS period,
        t.txn_ts + to_hours(CAST(p.max_hold_hours AS INTEGER)) AS hold_expires_at
    FROM transactions t, p
    WHERE t.direction = 'CREDIT' AND t.amount >= p.min_amount
),
matched AS (
    SELECT
        c.customer_id,
        c.period,
        c.txn_ts                       AS credit_ts,
        c.amount                       AS credit_amount,
        COALESCE(SUM(d.amount), 0.0)   AS debit_amount
    FROM credits c
    LEFT JOIN transactions d
      ON d.customer_id = c.customer_id
     AND d.direction = 'DEBIT'
     AND d.txn_ts > c.txn_ts
     AND d.txn_ts <= c.hold_expires_at
    GROUP BY 1, 2, 3, 4
),
aggregated AS (
    SELECT
        customer_id,
        period,
        SUM(credit_amount)                            AS credit_total,
        SUM(LEAST(debit_amount, credit_amount))       AS passthrough_total,
        COUNT(*)                                      AS credit_count
    FROM matched
    GROUP BY 1, 2
)
SELECT
    'R02_VELOCITY' AS rule_id,
    a.customer_id,
    a.period,
    a.passthrough_total / NULLIF(a.credit_total, 0) AS metric_value,
    a.passthrough_total                             AS alert_amount,
    a.credit_count                                  AS txn_count
FROM aggregated a, p
WHERE a.credit_total > 0
  AND a.passthrough_total / a.credit_total >= p.min_passthrough_ratio
