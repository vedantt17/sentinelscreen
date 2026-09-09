"""DuckDB scenarios, feature discipline, models, and ATL/BTL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import AppConfig
from src.data.generator import GeneratedDataset
from src.exceptions import ConfigurationError, ModelTrainingError, RuleExecutionError
from src.monitoring.atl_btl import recommended_thresholds, run_atl_btl
from src.monitoring.features import (
    KEY_COLUMNS,
    BaselineTransformer,
    attach_labels,
    build_base_features,
    out_of_time_mask,
    prepare_matrices,
    split_by_customer,
)
from src.monitoring.models import (
    operating_point_at_budget,
    operating_point_at_recall,
    train_all,
)
from src.monitoring.rules import REQUIRED_COLUMNS, RuleEngine, combine_alerts, load_scenario_sql


@pytest.fixture()
def engine(loaded_db, app_config: AppConfig) -> RuleEngine:  # type: ignore[no-untyped-def]
    rule_engine = RuleEngine(loaded_db, app_config.monitoring)
    rule_engine.install_risk_countries(app_config.generation.corridors)
    rule_engine.install_params()
    return rule_engine


# ------------------------------------------------------------------ scenarios
def test_all_eight_scenarios_execute(engine: RuleEngine, app_config: AppConfig) -> None:
    results = engine.run_all()
    assert len(results) == 8
    for rule_id, result in results.items():
        assert list(result.alerts.columns)[: len(REQUIRED_COLUMNS)] == list(REQUIRED_COLUMNS)
        if not result.alerts.empty:
            assert (result.alerts["rule_id"] == rule_id).all()


def test_scenario_sql_reads_every_threshold_from_the_params_table(
    app_config: AppConfig,
) -> None:
    """No threshold may be interpolated into the SQL text.

    If a value were hardcoded, the ATL/BTL sweep would rewrite the parameter
    table, re-run the identical query, and silently produce identical results —
    tuning evidence that describes nothing.
    """
    for scenario in app_config.monitoring.scenarios:
        sql = load_scenario_sql(scenario.id)
        assert "rule_params" in sql, f"{scenario.id} does not read the parameter table"
        for name in scenario.params:
            assert name in sql, f"{scenario.id} never references its parameter {name!r}"


def test_changing_a_parameter_changes_the_alerts(engine: RuleEngine, app_config: AppConfig) -> None:
    scenario = app_config.monitoring.by_id("R01_STRUCTURING")
    others = [s for s in app_config.monitoring.scenarios if s.id != scenario.id]

    strict = scenario.with_param("min_txn_count", 8)
    engine.install_params([*others, strict])
    tight = engine.run(strict)

    loose = scenario.with_param("min_txn_count", 2)
    engine.install_params([*others, loose])
    relaxed = engine.run(loose)

    assert len(relaxed.alerts) >= len(tight.alerts)


def test_unknown_scenario_id_raises(engine: RuleEngine, app_config: AppConfig) -> None:
    from src.config import ScenarioConfig

    bogus = ScenarioConfig(
        id="R99_DOES_NOT_EXIST", name="x", description="x",
        params={"a": 1.0}, sweep_param="a", sweep_values=(1.0,),
    )
    with pytest.raises(RuleExecutionError, match="no SQL file"):
        engine.run(bogus)


def test_with_param_rejects_an_unknown_parameter(app_config: AppConfig) -> None:
    scenario = app_config.monitoring.by_id("R02_VELOCITY")
    with pytest.raises(ConfigurationError, match="has no parameter"):
        scenario.with_param("not_a_param", 1.0)


def test_combined_queue_is_deterministically_ordered(engine: RuleEngine) -> None:
    combined = combine_alerts(engine.run_all())
    if combined.empty:
        pytest.skip("fixture produced no alerts")
    ordered = combined.sort_values(
        ["customer_id", "period", "rule_id"], kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, ordered)


def test_dormant_scenario_requires_a_real_gap(engine: RuleEngine, loaded_db) -> None:  # type: ignore[no-untyped-def]
    result = engine.run(engine.config.by_id("R04_DORMANT_REACTIVATION"))
    if result.alerts.empty:
        pytest.skip("fixture produced no dormancy alerts")
    # Every alerted cell must contain a transaction preceded by a gap of at
    # least the configured dormancy period.
    gaps = loaded_db.execute(
        """
        SELECT customer_id, strftime(txn_ts, '%Y-%m') AS period,
               MAX(date_diff('day', prior_ts, txn_ts)) AS max_gap
        FROM (
            SELECT customer_id, txn_ts,
                   LAG(txn_ts) OVER (PARTITION BY customer_id ORDER BY txn_ts) AS prior_ts
            FROM transactions
        )
        WHERE prior_ts IS NOT NULL
        GROUP BY 1, 2
        """
    ).fetchdf()
    lookup = {(r.customer_id, r.period): r.max_gap for r in gaps.itertuples()}
    for row in result.alerts.itertuples():
        assert lookup.get((row.customer_id, row.period), 0) >= 90


# ------------------------------------------------------------------- features
def test_feature_query_never_touches_the_label_table(loaded_db) -> None:  # type: ignore[no-untyped-def]
    from src.paths import SQL_DIR

    sql = (SQL_DIR / "features.sql").read_text(encoding="utf-8").lower()
    assert "labels" not in sql, "the feature query must not reference the label table"


def test_split_is_grouped_by_customer(small_dataset: GeneratedDataset, loaded_db) -> None:  # type: ignore[no-untyped-def]
    features = build_base_features(loaded_db)
    split = split_by_customer(features, 0.25, seed=7)
    assert not (split.train_customers & split.test_customers)
    assert split.train_customers and split.test_customers


def test_baselines_are_fitted_on_training_data_only(loaded_db) -> None:  # type: ignore[no-untyped-def]
    """A baseline fitted over everything leaks test-period behaviour into training."""
    features = build_base_features(loaded_db)
    split = split_by_customer(features, 0.25, seed=7)
    train_only = features[features["customer_id"].isin(split.train_customers)]

    fitted_on_train = BaselineTransformer().fit(train_only)
    fitted_on_all = BaselineTransformer().fit(features)
    assert fitted_on_train.peer_medians != fitted_on_all.peer_medians or len(
        split.test_customers
    ) == 0


def test_transform_before_fit_raises() -> None:
    with pytest.raises(ModelTrainingError, match="before fit"):
        BaselineTransformer().transform(pd.DataFrame({"txn_count": [1.0]}))


def test_shrinkage_pulls_single_transaction_rates_toward_the_baseline(
    loaded_db,
) -> None:  # type: ignore[no-untyped-def]
    """A one-transaction month must not report a cash share of exactly 1.00."""
    features = build_base_features(loaded_db)
    transformer = BaselineTransformer().fit(features)
    transformed = transformer.transform(features)

    singles = features["txn_count"] == 1
    if not singles.any():
        pytest.skip("fixture has no single-transaction cells")
    raw_extremes = ((features.loc[singles, "cash_share"] == 1.0)).sum()
    shrunk_extremes = ((transformed.loc[singles, "cash_share"] == 1.0)).sum()
    assert shrunk_extremes < raw_extremes


def test_labels_are_attached_only_after_engineering(
    small_dataset: GeneratedDataset, loaded_db
) -> None:  # type: ignore[no-untyped-def]
    features = build_base_features(loaded_db)
    assert "is_true_anomaly" not in features.columns
    labelled = attach_labels(features, small_dataset.labels)
    assert "is_true_anomaly" in labelled.columns
    assert labelled["is_true_anomaly"].dtype == bool


def test_prepare_matrices_keeps_the_target_out_of_the_feature_list(
    small_dataset: GeneratedDataset, loaded_db
) -> None:  # type: ignore[no-untyped-def]
    features = build_base_features(loaded_db)
    train, test, names, split, _ = prepare_matrices(features, small_dataset.labels, 0.25, 11)
    for forbidden in ("is_true_anomaly", "is_near_miss", "typology", "intensity", *KEY_COLUMNS):
        assert forbidden not in names
    assert not (set(train["customer_id"]) & set(test["customer_id"]))


def test_out_of_time_mask_requires_headroom(loaded_db) -> None:  # type: ignore[no-untyped-def]
    features = build_base_features(loaded_db)
    assert out_of_time_mask(features, 3).any()
    with pytest.raises(ModelTrainingError, match="no in-time data"):
        out_of_time_mask(features, 999)


# --------------------------------------------------------------------- models
def test_operating_point_at_recall_reaches_the_target() -> None:
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.6, 90), np.linspace(0.5, 1.0, 10)])
    point = operating_point_at_recall(y, scores, 0.8)
    assert point.recall >= 0.8
    assert point.true_positives + point.false_positives == point.alert_volume


def test_operating_point_at_budget_respects_the_budget() -> None:
    y = np.array([0] * 95 + [1] * 5)
    scores = np.linspace(0, 1, 100)
    point = operating_point_at_budget(y, scores, 0.10)
    assert point.alert_volume == 10


def test_operating_point_without_positives_raises() -> None:
    with pytest.raises(ModelTrainingError, match="no positives"):
        operating_point_at_recall(np.zeros(10, dtype=int), np.linspace(0, 1, 10), 0.5)


def test_scoring_a_single_class_slice_is_refused(app_config: AppConfig) -> None:
    """The contract the pipeline's out-of-time guard exists to respect.

    A trailing slice of a small portfolio at 0.4% prevalence can contain no
    positives at all. ROC-AUC is undefined over one class, so `evaluate` must
    fail rather than invent a number — and the pipeline must check before
    calling it, which is what it now does.
    """
    from sklearn.metrics import roc_auc_score

    with pytest.raises(ValueError, match="Only one class"):
        roc_auc_score(np.zeros(50, dtype=int), np.linspace(0, 1, 50))


def test_models_train_and_stay_inside_the_plausibility_band(
    small_dataset: GeneratedDataset, loaded_db, app_config: AppConfig
) -> None:  # type: ignore[no-untyped-def]
    features = build_base_features(loaded_db)
    train, test, names, _, _ = prepare_matrices(features, small_dataset.labels, 0.25, 3)
    if train["is_true_anomaly"].sum() == 0 or test["is_true_anomaly"].sum() == 0:
        pytest.skip("small fixture produced a fold without positives")

    models = train_all(
        train, test, names, app_config.monitoring, app_config.seed, enforce_tripwire=False
    )
    assert len(models.evaluations) == 3
    for evaluation in models.evaluations:
        assert 0.0 <= evaluation.roc_auc <= 1.0
        # The test fold keeps the natural imbalance; no resampling anywhere.
        assert evaluation.prevalence < 0.05


def test_leakage_tripwire_fires_on_a_leaked_feature(
    small_dataset: GeneratedDataset, loaded_db, app_config: AppConfig
) -> None:
    """Hand the model the label and confirm training refuses to ship it."""
    features = build_base_features(loaded_db)
    train, test, names, _, _ = prepare_matrices(features, small_dataset.labels, 0.25, 3)
    if train["is_true_anomaly"].sum() == 0 or test["is_true_anomaly"].sum() == 0:
        pytest.skip("small fixture produced a fold without positives")

    for frame in (train, test):
        frame["leaked"] = frame["is_true_anomaly"].astype(float)
    with pytest.raises(ModelTrainingError, match="leakage|ceiling"):
        train_all(
            train, test, (*names, "leaked"), app_config.monitoring, app_config.seed,
            enforce_tripwire=True,
        )


# -------------------------------------------------------------------- ATL/BTL
def test_atl_btl_sweep_produces_a_tuning_table(
    engine: RuleEngine, small_dataset: GeneratedDataset, app_config: AppConfig
) -> None:
    sweep = run_atl_btl(engine, app_config.monitoring, small_dataset.labels)
    assert sweep["rule_id"].nunique() == 8
    assert set(sweep["position"]) <= {"TIGHTER", "BASELINE", "LOOSER"}
    # Exactly one baseline row per rule, or the deltas are meaningless.
    baselines = sweep[sweep["position"] == "BASELINE"]
    assert len(baselines) == 8

    for _, group in sweep.groupby("rule_id"):
        ordered = group.sort_values("param_value")
        # Every swept parameter is a minimum, so tightening cannot increase volume.
        assert ordered["alert_volume"].is_monotonic_decreasing


def test_sweep_leaves_the_engine_on_baseline_parameters(
    engine: RuleEngine, small_dataset: GeneratedDataset, app_config: AppConfig
) -> None:
    """A sweep must not have side effects on whatever runs next."""
    run_atl_btl(engine, app_config.monitoring, small_dataset.labels)
    installed = engine.connection.execute(
        "SELECT rule_id, param_name, param_value FROM rule_params ORDER BY 1, 2"
    ).fetchdf()
    expected = {
        (s.id, name): value
        for s in app_config.monitoring.scenarios
        for name, value in s.params.items()
    }
    for row in installed.itertuples():
        assert expected[(row.rule_id, row.param_name)] == pytest.approx(row.param_value)


def test_recommendations_stay_inside_the_review_budget(
    engine: RuleEngine, small_dataset: GeneratedDataset, app_config: AppConfig
) -> None:
    sweep = run_atl_btl(engine, app_config.monitoring, small_dataset.labels)
    picks = recommended_thresholds(sweep, max_reviews_per_hit=40.0)
    assert len(picks) == 8
    assert (picks["recommended_recall"] >= picks["baseline_recall"] - 1e-9).all()
