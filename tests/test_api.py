"""API surface: the four endpoints, and the controls behind them."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import (
    AlertNotFoundError,
    DispositionNotPermittedError,
    ScreeningService,
    WatchlistUnavailableError,
)
from src.config import AppConfig
from src.data.generator import GeneratedDataset
from src.data.loaders import write_parquet


@pytest.fixture(scope="module")
def service_paths(tmp_path_factory, request) -> tuple[Path, Path]:  # type: ignore[no-untyped-def]
    """A small on-disk dataset plus a hand-built alert file, as the API expects."""
    dataset: GeneratedDataset = request.getfixturevalue("small_dataset")
    data_dir = tmp_path_factory.mktemp("api_data")
    output_dir = tmp_path_factory.mktemp("api_outputs")
    write_parquet(
        {
            "watchlist": dataset.watchlist,
            "customers": dataset.customers,
            "transactions": dataset.transactions,
            "labels": dataset.labels,
        },
        data_dir,
    )
    watchlist = dataset.watchlist.iloc[:3]
    pd.DataFrame(
        [
            {
                "subject_id": "CUST-000001",
                "subject_name": "Test Subject",
                "watchlist_uid": watchlist.iloc[0]["sdn_uid"],
                "watchlist_name": watchlist.iloc[0]["primary_name"],
                "program": watchlist.iloc[0]["program"],
                "matched_on": "primary",
                "name_score": 0.97,
                "corroboration_delta": 0.0,
                "final_score": 0.97,
                "decision": "ESCALATE",
                "jaro_winkler": 0.97,
                "token_set": 0.95,
                "phonetic": 1.0,
                "initials": 1.0,
                "order_swapped": False,
            },
            {
                "subject_id": "CUST-000002",
                "subject_name": "Other Subject",
                "watchlist_uid": watchlist.iloc[1]["sdn_uid"],
                "watchlist_name": watchlist.iloc[1]["primary_name"],
                "program": watchlist.iloc[1]["program"],
                "matched_on": "alias",
                "name_score": 0.87,
                "corroboration_delta": -0.02,
                "final_score": 0.85,
                "decision": "REVIEW",
                "jaro_winkler": 0.88,
                "token_set": 0.85,
                "phonetic": 0.5,
                "initials": 0.5,
                "order_swapped": True,
            },
        ]
    ).to_csv(output_dir / "screening_alerts.csv", index=False)
    pd.DataFrame(
        [
            {
                "rule_id": "R01_STRUCTURING",
                "customer_id": "CUST-000003",
                "period": "2024-05",
                "metric_value": 5.0,
                "alert_amount": 46000.0,
                "txn_count": 5,
            }
        ]
    ).to_csv(output_dir / "rule_alerts.csv", index=False)
    return data_dir, output_dir


@pytest.fixture(scope="module")
def service(service_paths: tuple[Path, Path], app_config: AppConfig) -> ScreeningService:
    data_dir, output_dir = service_paths
    return ScreeningService.from_disk(app_config, data_dir=data_dir, output_dir=output_dir)


@pytest.fixture()
def client(service: ScreeningService) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        # Swap in the fixture-scoped service so tests do not depend on whether
        # `make run` has been executed in this working tree.
        test_client.app.state.service = service
        yield test_client


# --------------------------------------------------------------------- health
def test_health_reports_the_loaded_state(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["watchlist_entities"] > 0
    assert body["alerts_loaded"] == 3


def test_missing_watchlist_raises_a_clear_error(tmp_path: Path, app_config: AppConfig) -> None:
    with pytest.raises(WatchlistUnavailableError, match="make run"):
        ScreeningService.from_disk(app_config, data_dir=tmp_path, output_dir=tmp_path)


# ------------------------------------------------------------ /screen/name
def test_screen_name_returns_the_scoring_breakdown(
    client: TestClient, service: ScreeningService
) -> None:
    target = service.entries[0]
    response = client.post("/screen/name", json={"name": target.name, "top_n": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["rule_version"] == service.config.screening.rule_version
    assert body["matches"], "screening an exact watchlist name must return a match"
    top = body["matches"][0]
    assert top["watchlist_uid"] == target.uid
    assert set(top["signals"]) == {
        "jaro_winkler", "token_set", "phonetic", "initials", "order_swapped"
    }


def test_screen_name_reports_transliteration_context(client: TestClient) -> None:
    body = client.post("/screen/name", json={"name": "Владимир Иванов"}).json()
    assert body["detected_script"] == "CYRILLIC"
    assert body["latin_form"].lower().startswith("vladimir")


def test_screen_name_rejects_a_malformed_dob(client: TestClient) -> None:
    response = client.post("/screen/name", json={"name": "Test", "dob": "31-12-1980"})
    assert response.status_code == 422


def test_screen_name_rejects_unknown_fields(client: TestClient) -> None:
    response = client.post("/screen/name", json={"name": "Test", "surprise": 1})
    assert response.status_code == 422


def test_unrelated_name_returns_no_matches(client: TestClient) -> None:
    body = client.post("/screen/name", json={"name": "Qqqzzz Vvvxxx"}).json()
    assert body["matches"] == []


# ----------------------------------------------------- /screen/transaction
def test_transaction_flags_near_threshold_cash(client: TestClient) -> None:
    body = client.post(
        "/screen/transaction",
        json={
            "txn_id": "T1", "customer_id": "C1", "amount": 9400.0,
            "channel": "CASH", "direction": "CREDIT",
        },
    ).json()
    assert "NEAR_CTR_CASH" in {f["code"] for f in body["flags"]}


def test_transaction_flags_ctr_reportable(client: TestClient) -> None:
    body = client.post(
        "/screen/transaction",
        json={
            "txn_id": "T2", "customer_id": "C1", "amount": 25000.0,
            "channel": "CASH", "direction": "CREDIT",
        },
    ).json()
    assert "CTR_REPORTABLE" in {f["code"] for f in body["flags"]}


def test_high_risk_corridor_forces_review(client: TestClient) -> None:
    body = client.post(
        "/screen/transaction",
        json={
            "txn_id": "T3", "customer_id": "C1", "amount": 5000.0,
            "channel": "WIRE", "direction": "DEBIT", "counterparty_country": "IR",
        },
    ).json()
    assert body["requires_review"] is True


def test_sanctioned_counterparty_is_matched(
    client: TestClient, service: ScreeningService
) -> None:
    target = service.entries[0]
    body = client.post(
        "/screen/transaction",
        json={
            "txn_id": "T4", "customer_id": "C1", "amount": 20000.0,
            "channel": "WIRE", "direction": "DEBIT", "counterparty_name": target.name,
        },
    ).json()
    assert body["counterparty_matches"]
    assert body["requires_review"] is True


def test_transaction_rejects_a_bad_channel(client: TestClient) -> None:
    response = client.post(
        "/screen/transaction",
        json={
            "txn_id": "T5", "customer_id": "C1", "amount": 100.0,
            "channel": "CHEQUE", "direction": "DEBIT",
        },
    )
    assert response.status_code == 422


# ------------------------------------------------------------------- /alerts
def test_alerts_are_ranked_by_score(client: TestClient) -> None:
    alerts = client.get("/alerts").json()["alerts"]
    scores = [a["score"] for a in alerts]
    assert scores == sorted(scores, reverse=True)


def test_alerts_filter_by_type(client: TestClient) -> None:
    body = client.get("/alerts", params={"alert_type": "TRANSACTION_MONITORING"}).json()
    assert body["total"] == 1
    assert body["alerts"][0]["period"] == "2024-05"


def test_alerts_paginate(client: TestClient) -> None:
    body = client.get("/alerts", params={"limit": 1, "offset": 1}).json()
    assert len(body["alerts"]) == 1
    assert body["total"] == 3


def test_unknown_alert_returns_404(client: TestClient) -> None:
    assert client.get("/alerts/NOPE-1").status_code == 404


# -------------------------------------------------------------- disposition
def test_disposition_is_written_to_the_audit_chain(
    client: TestClient, service: ScreeningService
) -> None:
    before = len(service.audit)
    response = client.post(
        "/alerts/SCR-0000002/disposition",
        json={"disposition": "FALSE_POSITIVE", "analyst_id": "an-17", "note": "different DOB"},
    )
    assert response.status_code == 200
    assert len(service.audit) == before + 1
    service.audit.verify()
    assert response.json()["audit_hash"] == service.audit.head_hash


def test_l1_cannot_close_an_escalated_sanctions_match(client: TestClient) -> None:
    response = client.post(
        "/alerts/SCR-0000001/disposition",
        json={"disposition": "FALSE_POSITIVE", "analyst_id": "an-17", "analyst_level": "L1"},
    )
    assert response.status_code == 403
    assert "L2" in response.json()["detail"]


def test_l2_can_close_an_escalated_sanctions_match(client: TestClient) -> None:
    response = client.post(
        "/alerts/SCR-0000001/disposition",
        json={"disposition": "TRUE_POSITIVE", "analyst_id": "l2-3", "analyst_level": "L2"},
    )
    assert response.status_code == 200
    assert client.get("/alerts/SCR-0000001").json()["disposition"] == "TRUE_POSITIVE"


def test_disposition_on_unknown_alert_returns_404(client: TestClient) -> None:
    response = client.post(
        "/alerts/NOPE-1/disposition",
        json={"disposition": "FALSE_POSITIVE", "analyst_id": "an-1"},
    )
    assert response.status_code == 404


def test_service_level_errors_are_typed(service: ScreeningService) -> None:
    with pytest.raises(AlertNotFoundError):
        service.get_alert("NOPE-1")
    with pytest.raises(DispositionNotPermittedError):
        service.disposition("SCR-0000001", "FALSE_POSITIVE", "an-1", "", "L1")
