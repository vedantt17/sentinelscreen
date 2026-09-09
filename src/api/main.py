"""FastAPI surface.

Four endpoints, matching how a screening platform is actually consumed:
real-time name screening at onboarding, per-transaction screening in the payment
path, the analyst queue, and disposition. Routes stay thin — every control lives
in `service.py` so it can be tested without HTTP.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status

from src.api.schemas import (
    Alert,
    AlertPage,
    DispositionRequest,
    DispositionResponse,
    HealthResponse,
    NameScreenRequest,
    NameScreenResponse,
    TransactionScreenRequest,
    TransactionScreenResponse,
)
from src.api.service import (
    AlertNotFoundError,
    DispositionNotPermittedError,
    ScreeningService,
    WatchlistUnavailableError,
    alert_payload,
    match_payload,
)
from src.exceptions import SanctionsMatchingError
from src.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Index the watchlist once at startup, not per request.

    Building the blocking index over ~9,000 reference names takes a moment; a
    per-request rebuild would dominate latency and make the service useless for
    the payment path it is meant to sit in.
    """
    configure_logging(logging.INFO)
    try:
        app.state.service = ScreeningService.from_disk()
        logger.info("service ready", extra=app.state.service.health())
    except WatchlistUnavailableError as exc:
        # Start anyway so /health can explain the problem. Screening endpoints
        # return 503 until the dataset exists.
        app.state.service = None
        app.state.startup_error = str(exc)
        logger.error("service started without a watchlist", extra={"detail": str(exc)})
    yield


app = FastAPI(
    title="SentinelScreen",
    version="1.0.0",
    summary="Sanctions screening and transaction monitoring",
    description=(
        "Reference implementation of a sanctions screening and AML transaction "
        "monitoring stack. All data is synthetic. This is a portfolio project and "
        "is not legal or compliance advice."
    ),
    lifespan=lifespan,
)


def get_service(request: Request) -> ScreeningService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "startup_error", "screening service unavailable"),
        )
    assert isinstance(service, ScreeningService)
    return service


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "service", None)
    if service is None:
        return {
            "status": "degraded",
            "watchlist_entities": 0,
            "reference_names": 0,
            "alerts_loaded": 0,
            "screening_rule_version": "unavailable",
            "monitoring_rule_version": "unavailable",
            "detail": getattr(request.app.state, "startup_error", "not initialised"),
        }
    return dict(service.health())


@app.post("/screen/name", response_model=NameScreenResponse, tags=["screening"])
def screen_name(
    payload: NameScreenRequest, service: ScreeningService = Depends(get_service)
) -> dict[str, Any]:
    try:
        matches, context = service.screen_name(
            name=payload.name,
            subject_id=payload.subject_id,
            dob=payload.dob,
            nationality=payload.nationality,
            national_id=payload.national_id,
            top_n=payload.top_n,
            min_score=payload.min_score,
        )
    except SanctionsMatchingError as exc:
        # A name the engine cannot reduce to a Latin form is a client-side data
        # problem, not a server fault: 422 rather than 500.
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {
        "subject_id": payload.subject_id,
        "query_name": payload.name,
        "rule_version": service.config.screening.rule_version,
        "matches": [match_payload(m) for m in matches],
        **context,
    }


@app.post("/screen/transaction", response_model=TransactionScreenResponse, tags=["screening"])
def screen_transaction(
    payload: TransactionScreenRequest, service: ScreeningService = Depends(get_service)
) -> dict[str, Any]:
    try:
        matches, flags = service.screen_transaction(
            txn_id=payload.txn_id,
            customer_id=payload.customer_id,
            amount=payload.amount,
            channel=payload.channel,
            direction=payload.direction,
            counterparty_name=payload.counterparty_name,
            counterparty_country=payload.counterparty_country,
        )
    except SanctionsMatchingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    requires_review = bool(matches) or any(f["severity"] == "HIGH" for f in flags)
    return {
        "txn_id": payload.txn_id,
        "customer_id": payload.customer_id,
        "rule_version": service.config.screening.rule_version,
        "counterparty_matches": [match_payload(m) for m in matches],
        "flags": flags,
        "requires_review": requires_review,
    }


@app.get("/alerts", response_model=AlertPage, tags=["alerts"])
def list_alerts(
    service: ScreeningService = Depends(get_service),
    alert_type: Literal["SCREENING", "TRANSACTION_MONITORING"] | None = None,
    subject_id: str | None = None,
    decision: str | None = None,
    undispositioned_only: bool = False,
    min_score: float | None = Query(default=None, ge=0.0),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    total, records = service.list_alerts(
        alert_type=alert_type,
        subject_id=subject_id,
        decision=decision,
        undispositioned_only=undispositioned_only,
        min_score=min_score,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "alerts": [alert_payload(r) for r in records],
    }


@app.get("/alerts/{alert_id}", response_model=Alert, tags=["alerts"])
def get_alert(alert_id: str, service: ScreeningService = Depends(get_service)) -> Any:
    try:
        return alert_payload(service.get_alert(alert_id))
    except AlertNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.post(
    "/alerts/{alert_id}/disposition",
    response_model=DispositionResponse,
    tags=["alerts"],
)
def disposition_alert(
    alert_id: str,
    payload: DispositionRequest,
    service: ScreeningService = Depends(get_service),
) -> dict[str, Any]:
    try:
        _, sequence, content_hash = service.disposition(
            alert_id=alert_id,
            disposition=payload.disposition,
            analyst_id=payload.analyst_id,
            note=payload.note,
            analyst_level=payload.analyst_level,
        )
    except AlertNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except DispositionNotPermittedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    return {
        "alert_id": alert_id,
        "disposition": payload.disposition,
        "analyst_id": payload.analyst_id,
        "audit_sequence": sequence,
        "audit_hash": content_hash,
    }
