-- R07: exposure to high-risk and comprehensively sanctioned jurisdictions.
--
-- Share as well as absolute amount, because a single 12,000 dollar payment to a
-- high-risk corridor from a large importer is materially different from the
-- same payment being that customer's entire month. The country tiering lives in
-- a table so that a FATF grey-list revision is a data change, not a code change.
WITH p AS (
    SELECT
        MAX(CASE WHEN param_name = 'min_corridor_share'  THEN param_value END) AS min_corridor_share,
        MAX(CASE WHEN param_name = 'min_corridor_amount' THEN param_value END) AS min_corridor_amount,
        MAX(CASE WHEN param_name = 'min_txn_count'       THEN param_value END) AS min_txn_count
    FROM rule_params WHERE rule_id = 'R07_HIGH_RISK_CORRIDOR'
),
scoped AS (
    SELECT
        t.customer_id,
        strftime(t.txn_ts, '%Y-%m') AS period,
        t.amount,
        CASE WHEN r.country IS NOT NULL THEN t.amount ELSE 0.0 END AS corridor_amount,
        CASE WHEN r.country IS NOT NULL THEN 1 ELSE 0 END          AS is_corridor
    FROM transactions t
    LEFT JOIN risk_countries r
           ON r.country = t.counterparty_country AND r.tier = 'HIGH'
)
SELECT
    'R07_HIGH_RISK_CORRIDOR' AS rule_id,
    s.customer_id,
    s.period,
    SUM(s.corridor_amount) / NULLIF(SUM(s.amount), 0) AS metric_value,
    SUM(s.corridor_amount)                            AS alert_amount,
    SUM(s.is_corridor)                                AS txn_count
FROM scoped s, p
GROUP BY 1, 2, 3, p.min_corridor_share, p.min_corridor_amount, p.min_txn_count
HAVING SUM(s.is_corridor) >= p.min_txn_count
   AND SUM(s.corridor_amount) >= p.min_corridor_amount
   AND SUM(s.corridor_amount) / NULLIF(SUM(s.amount), 0) >= p.min_corridor_share
