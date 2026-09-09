-- Customer-month feature matrix.
--
-- Every column here is a function of the transaction feed and the customer
-- master only. The label table is deliberately not referenced: it is a separate
-- table precisely so that a careless `SELECT *` cannot pull the target into the
-- design matrix. Peer- and self-relative features that require a baseline are
-- NOT computed here; they are fitted on the training split alone in Python,
-- because a baseline fitted over the full dataset leaks test-period behaviour
-- into training features.
WITH monthly AS (
    SELECT
        t.customer_id,
        strftime(t.txn_ts, '%Y-%m')                                             AS period,
        COUNT(*)                                                                AS txn_count,
        SUM(t.amount)                                                           AS total_amount,
        AVG(t.amount)                                                           AS avg_amount,
        MAX(t.amount)                                                           AS max_amount,
        COALESCE(STDDEV_SAMP(t.amount), 0.0)                                    AS std_amount,
        QUANTILE_CONT(t.amount, 0.5)                                            AS median_amount,

        SUM(CASE WHEN t.direction = 'CREDIT' THEN t.amount ELSE 0 END)          AS credit_amount,
        SUM(CASE WHEN t.direction = 'DEBIT'  THEN t.amount ELSE 0 END)          AS debit_amount,
        SUM(CASE WHEN t.direction = 'CREDIT' THEN 1 ELSE 0 END)                 AS credit_count,

        SUM(CASE WHEN t.channel = 'CASH' THEN t.amount ELSE 0 END)              AS cash_amount,
        SUM(CASE WHEN t.channel = 'CASH' THEN 1 ELSE 0 END)                     AS cash_count,
        SUM(CASE WHEN t.channel = 'WIRE' THEN t.amount ELSE 0 END)              AS wire_amount,
        SUM(CASE WHEN t.channel = 'CARD' THEN t.amount ELSE 0 END)              AS card_amount,

        -- Amount-shape signals. The near-CTR band is deliberately wider than
        -- the structuring rule's band so that the model can see the shoulder
        -- of the distribution the rule cuts off.
        SUM(CASE WHEN t.amount BETWEEN 7500 AND 9999.99 THEN 1 ELSE 0 END)      AS near_ctr_count,
        SUM(CASE WHEN t.amount = ROUND(t.amount / 1000.0) * 1000.0 THEN 1 ELSE 0 END)
                                                                                AS round_count,
        SUM(CASE WHEN t.amount >= 10000 THEN 1 ELSE 0 END)                      AS ctr_count,

        SUM(CASE WHEN t.is_cross_border THEN t.amount ELSE 0 END)               AS cross_border_amount,
        COUNT(DISTINCT t.counterparty_country)                                  AS distinct_countries,
        COUNT(DISTINCT t.counterparty_name)                                     AS distinct_counterparties,

        SUM(CASE WHEN HOUR(t.txn_ts) < 7 OR HOUR(t.txn_ts) >= 20 THEN 1 ELSE 0 END)
                                                                                AS night_count,
        SUM(CASE WHEN DAYOFWEEK(t.txn_ts) IN (0, 6) THEN 1 ELSE 0 END)          AS weekend_count,

        MIN(t.txn_ts)                                                           AS first_ts,
        MAX(t.txn_ts)                                                           AS last_ts
    FROM transactions t
    GROUP BY 1, 2
),
corridor AS (
    SELECT
        t.customer_id,
        strftime(t.txn_ts, '%Y-%m') AS period,
        SUM(CASE WHEN r.tier = 'HIGH'   THEN t.amount ELSE 0 END) AS high_risk_amount,
        SUM(CASE WHEN r.tier = 'MEDIUM' THEN t.amount ELSE 0 END) AS medium_risk_amount,
        SUM(CASE WHEN r.tier = 'HIGH'   THEN 1 ELSE 0 END)        AS high_risk_count
    FROM transactions t
    LEFT JOIN risk_countries r ON r.country = t.counterparty_country
    GROUP BY 1, 2
),
top_counterparty AS (
    SELECT customer_id, period, MAX(cp_amount) AS top_cp_amount
    FROM (
        SELECT
            t.customer_id,
            strftime(t.txn_ts, '%Y-%m') AS period,
            t.counterparty_name,
            SUM(t.amount)               AS cp_amount
        FROM transactions t
        WHERE t.counterparty_name IS NOT NULL
        GROUP BY 1, 2, 3
    )
    GROUP BY 1, 2
),
-- Self-relative dormancy: the gap between this month's first transaction and
-- the account's immediately preceding one. Computed with a window function
-- over the customer's own history, so it never looks forward.
gaps AS (
    SELECT
        customer_id,
        period,
        MAX(days_since_prior) AS max_gap_days,
        MIN(days_since_prior) AS min_gap_days
    FROM (
        SELECT
            t.customer_id,
            strftime(t.txn_ts, '%Y-%m') AS period,
            date_diff(
                'day',
                LAG(t.txn_ts) OVER (PARTITION BY t.customer_id ORDER BY t.txn_ts),
                t.txn_ts
            ) AS days_since_prior
        FROM transactions t
    )
    GROUP BY 1, 2
)
SELECT
    m.customer_id,
    m.period,
    CAST(SUBSTR(m.period, 6, 2) AS INTEGER)                          AS month_of_year,
    m.txn_count,
    m.total_amount,
    m.avg_amount,
    m.max_amount,
    m.std_amount,
    m.median_amount,
    m.credit_amount,
    m.debit_amount,
    m.credit_count::DOUBLE / m.txn_count                             AS credit_share,
    -- Pass-through proxy: how much of what came in also went out this month.
    LEAST(m.debit_amount, m.credit_amount)
        / NULLIF(m.credit_amount, 0)                                 AS passthrough_ratio,
    m.cash_amount,
    m.cash_amount / NULLIF(m.total_amount, 0)                        AS cash_share,
    m.cash_count::DOUBLE / m.txn_count                               AS cash_txn_share,
    m.wire_amount / NULLIF(m.total_amount, 0)                        AS wire_share,
    m.card_amount / NULLIF(m.total_amount, 0)                        AS card_share,
    m.near_ctr_count,
    m.near_ctr_count::DOUBLE / m.txn_count                           AS near_ctr_share,
    m.round_count::DOUBLE / m.txn_count                              AS round_share,
    m.ctr_count,
    m.cross_border_amount / NULLIF(m.total_amount, 0)                AS cross_border_share,
    m.distinct_countries,
    m.distinct_counterparties,
    m.distinct_counterparties::DOUBLE / m.txn_count                  AS counterparty_diversity,
    m.night_count::DOUBLE / m.txn_count                              AS night_share,
    m.weekend_count::DOUBLE / m.txn_count                            AS weekend_share,
    date_diff('day', m.first_ts, m.last_ts)                          AS active_span_days,
    COALESCE(c.high_risk_amount, 0.0)                                AS high_risk_amount,
    COALESCE(c.high_risk_amount, 0.0) / NULLIF(m.total_amount, 0)    AS high_risk_share,
    COALESCE(c.medium_risk_amount, 0.0) / NULLIF(m.total_amount, 0)  AS medium_risk_share,
    COALESCE(c.high_risk_count, 0)                                   AS high_risk_count,
    COALESCE(tc.top_cp_amount, 0.0) / NULLIF(m.total_amount, 0)      AS top_counterparty_share,
    COALESCE(g.max_gap_days, 0)                                      AS max_gap_days,
    date_diff('day', cu.account_open_date, m.first_ts)               AS account_age_days,
    CASE cu.kyc_risk_rating WHEN 'LOW' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END AS kyc_risk_ordinal,
    CAST(cu.is_pep AS INTEGER)                                       AS is_pep,
    cu.behaviour_segment,
    CASE WHEN rn.country IS NOT NULL THEN 1 ELSE 0 END               AS nationality_high_risk
FROM monthly m
JOIN customers cu USING (customer_id)
LEFT JOIN corridor c          ON c.customer_id = m.customer_id AND c.period = m.period
LEFT JOIN top_counterparty tc ON tc.customer_id = m.customer_id AND tc.period = m.period
LEFT JOIN gaps g              ON g.customer_id = m.customer_id AND g.period = m.period
LEFT JOIN risk_countries rn   ON rn.country = cu.nationality AND rn.tier = 'HIGH'
ORDER BY m.customer_id, m.period
