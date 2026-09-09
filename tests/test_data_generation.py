"""Generator contracts: volumes, schema conformance, and injected structure."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import GenerationConfig
from src.data.generator import GeneratedDataset, generate_dataset
from src.data.schemas import (
    CUSTOMERS_SCHEMA,
    LABELS_SCHEMA,
    TRANSACTIONS_SCHEMA,
    WATCHLIST_SCHEMA,
    erd_mermaid,
)
from src.data.validation import coerce_frame, validate_frame
from src.exceptions import SchemaValidationError


def test_declared_volumes_are_exact(
    small_dataset: GeneratedDataset, small_generation_config: GenerationConfig
) -> None:
    assert len(small_dataset.watchlist) == small_generation_config.n_watchlist
    assert len(small_dataset.customers) == small_generation_config.n_customers
    assert len(small_dataset.transactions) == small_generation_config.n_transactions


@pytest.mark.parametrize(
    ("table", "schema"),
    [
        ("watchlist", WATCHLIST_SCHEMA),
        ("customers", CUSTOMERS_SCHEMA),
        ("transactions", TRANSACTIONS_SCHEMA),
        ("labels", LABELS_SCHEMA),
    ],
)
def test_frames_satisfy_their_schema(
    small_dataset: GeneratedDataset, table: str, schema
) -> None:  # type: ignore[no-untyped-def]
    frame = getattr(small_dataset, table)
    validate_frame(coerce_frame(frame, schema), schema)


def test_validation_actually_rejects_a_broken_frame(small_dataset: GeneratedDataset) -> None:
    broken = coerce_frame(small_dataset.transactions, TRANSACTIONS_SCHEMA).copy()
    broken.loc[broken.index[0], "amount"] = -5.0
    with pytest.raises(SchemaValidationError, match="below minimum"):
        validate_frame(broken, TRANSACTIONS_SCHEMA)


def test_validation_rejects_an_out_of_domain_value(small_dataset: GeneratedDataset) -> None:
    broken = coerce_frame(small_dataset.transactions, TRANSACTIONS_SCHEMA).copy()
    broken.loc[broken.index[0], "channel"] = "CHEQUE"
    with pytest.raises(SchemaValidationError, match="outside domain"):
        validate_frame(broken, TRANSACTIONS_SCHEMA)


def test_transactions_fall_inside_the_observation_window(
    small_dataset: GeneratedDataset, small_generation_config: GenerationConfig
) -> None:
    start = pd.Timestamp(small_generation_config.start_date)
    end = start + pd.DateOffset(months=small_generation_config.months)
    timestamps = small_dataset.transactions["txn_ts"]
    assert timestamps.min() >= start
    assert timestamps.max() < end


def test_every_transaction_resolves_to_a_customer(small_dataset: GeneratedDataset) -> None:
    known = set(small_dataset.customers["customer_id"])
    assert set(small_dataset.transactions["customer_id"]) <= known


def test_watchlist_carries_realistic_data_quality_defects(
    small_dataset: GeneratedDataset,
) -> None:
    watchlist = small_dataset.watchlist
    individuals = watchlist[watchlist["entity_type"] == "INDIVIDUAL"]
    # Missing DOB is the defect that most degrades corroboration in production,
    # so the generator must reproduce it rather than emit a tidy list.
    assert individuals["dob"].isna().mean() > 0.20
    assert watchlist["address_line"].isna().mean() > 0.20
    assert (watchlist["aliases"] != "").mean() > 0.30
    assert set(watchlist["name_script"].unique()) <= {"LATIN", "ARABIC", "CYRILLIC", "HAN"}
    assert watchlist["name_script"].nunique() == 4


def test_screening_ground_truth_is_planted(
    small_dataset: GeneratedDataset, small_generation_config: GenerationConfig
) -> None:
    customers = small_dataset.customers
    truth = small_generation_config.screening_ground_truth
    assert int(customers["watchlist_true_match_uid"].notna().sum()) == truth["true_match_customers"]
    assert int(customers["is_screening_near_miss"].sum()) == truth["near_miss_customers"]
    # A planted hit and a planted near miss must never be the same record, or
    # the precision denominator becomes ambiguous.
    overlap = customers["watchlist_true_match_uid"].notna() & customers["is_screening_near_miss"]
    assert not overlap.any()


def test_label_rates_match_the_configured_injection(
    small_dataset: GeneratedDataset, small_generation_config: GenerationConfig
) -> None:
    labels = small_dataset.labels
    typologies = small_generation_config.typologies
    assert labels["is_true_anomaly"].mean() == pytest.approx(
        float(typologies["anomaly_rate"]), rel=0.35
    )
    assert labels["is_near_miss"].mean() == pytest.approx(
        float(typologies["near_miss_rate"]), rel=0.35
    )
    # An anomalous cell is never also labelled a near miss.
    assert not (labels["is_true_anomaly"] & labels["is_near_miss"]).any()


def test_anomaly_detectability_is_spread_not_uniform(
    small_dataset: GeneratedDataset,
) -> None:
    """The knob that keeps the problem hard.

    Detectability must span a range. If every planted typology were executed
    loudly the model would separate trivially and the reported ROC-AUC would be
    an artefact of the generator; if none were, the problem would be unlearnable.
    All three bands must therefore be populated, with the quiet end carrying a
    substantial share — a rendered typology is capped at the cell's own size, so
    a quiet month can only ever host a quiet typology.
    """
    anomalies = small_dataset.labels[small_dataset.labels["is_true_anomaly"]]
    assert len(anomalies) > 0
    bands = anomalies["intensity"].value_counts(normalize=True)
    assert set(bands.index) == {"LOUD", "ATTENUATED", "INVISIBLE"}
    assert bands["INVISIBLE"] > 0.02, "some positives must leave no observable trace"
    assert bands["LOUD"] > 0.10, "some positives must be plainly visible"
    assert bands["ATTENUATED"] + bands["INVISIBLE"] > 0.35, "the hard tail must dominate"


def test_all_six_typologies_are_represented(small_dataset: GeneratedDataset) -> None:
    present = set(small_dataset.labels["typology"].dropna().unique())
    assert present == {
        "structuring", "rapid_movement", "high_risk_corridor",
        "sanctioned_counterparty", "dormant_reactivation", "layering",
    }


def test_labels_cover_exactly_the_active_customer_months(
    small_dataset: GeneratedDataset,
) -> None:
    transactions = small_dataset.transactions
    active = set(
        zip(transactions["customer_id"], transactions["txn_ts"].dt.strftime("%Y-%m"))
    )
    labelled = set(zip(small_dataset.labels["customer_id"], small_dataset.labels["period"]))
    assert active == labelled


def test_erd_is_generated_from_the_schemas(small_dataset: GeneratedDataset) -> None:
    diagram = erd_mermaid()
    assert diagram.startswith("erDiagram")
    for schema in (WATCHLIST_SCHEMA, CUSTOMERS_SCHEMA, TRANSACTIONS_SCHEMA, LABELS_SCHEMA):
        assert schema.name.upper() in diagram
    assert "CUSTOMERS ||--o{ TRANSACTIONS : customer_id" in diagram


def test_generation_is_reproducible_for_a_fixed_seed(
    small_generation_config: GenerationConfig,
) -> None:
    first = generate_dataset(small_generation_config)
    second = generate_dataset(small_generation_config)
    pd.testing.assert_frame_equal(first.transactions, second.transactions)
    pd.testing.assert_frame_equal(first.customers, second.customers)
    pd.testing.assert_frame_equal(first.watchlist, second.watchlist)
    assert first.manifest == second.manifest


def test_a_different_seed_produces_different_data(
    small_generation_config: GenerationConfig,
) -> None:
    import dataclasses

    other = generate_dataset(dataclasses.replace(small_generation_config, seed=987654321))
    baseline = generate_dataset(small_generation_config)
    assert not baseline.transactions["amount"].equals(other.transactions["amount"])
