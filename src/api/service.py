"""Application state and business logic behind the endpoints.

The routes in `main.py` are deliberately thin: they translate HTTP to calls on
this object and back. Everything that a compliance reviewer would call a control
— who may close what, what gets written to the audit trail, what counts as a
transaction risk flag — lives here, where it is testable without a web server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config import AppConfig, load_app_config
from src.data.loaders import PARQUET_FILES
from src.data.schemas import WATCHLIST_SCHEMA
from src.exceptions import SentinelScreenError
from src.logging_setup import get_logger
from src.paths import DATA_DIR, OUTPUT_DIR
from src.pipeline.audit import AuditLog, derive_run_id
from src.sanctions.engine import (
    MatchResult,
    ScreeningEngine,
    ScreeningSubject,
    WatchlistEntry,
)
from src.sanctions.normalize import normalize_name

logger = get_logger(__name__)


class AlertNotFoundError(SentinelScreenError):
    def __init__(self, alert_id: str) -> None:
        self.alert_id = alert_id
        super().__init__(f"no alert with id {alert_id!r}")


class DispositionNotPermittedError(SentinelScreenError):
    """An L1 analyst attempted to close an escalated sanctions alert."""


class WatchlistUnavailableError(SentinelScreenError):
    def __init__(self, path: Path) -> None:
        super().__init__(
            f"watchlist not found at {path}. Run `make run` to generate the dataset "
            "before starting the API."
        )


@dataclass
class AlertRecord:
    alert_id: str
    alert_type: str
    subject_id: str
    rule_id: str
    period: str | None
    score: float
    decision: str
    detail: dict[str, Any]
    disposition: str | None = None
    disposition_note: str | None = None
    disposition_analyst: str | None = None


@dataclass
class ScreeningService:
    config: AppConfig
    engine: ScreeningEngine
    entries: tuple[WatchlistEntry, ...]
    alerts: dict[str, AlertRecord] = field(default_factory=dict)
    audit: AuditLog = field(init=False)

    def __post_init__(self) -> None:
        self.audit = AuditLog(
            derive_run_id(
                self.config.seed,
                [self.config.screening.rule_version, self.config.monitoring.rule_version],
            )
        )

    # ------------------------------------------------------------ construction
    @classmethod
    def from_disk(
        cls,
        config: AppConfig | None = None,
        data_dir: Path = DATA_DIR,
        output_dir: Path = OUTPUT_DIR,
    ) -> "ScreeningService":
        app_config = config or load_app_config()
        watchlist_path = data_dir / PARQUET_FILES["watchlist"]
        if not watchlist_path.is_file():
            raise WatchlistUnavailableError(watchlist_path)

        watchlist = pd.read_parquet(watchlist_path)
        entries = tuple(
            WatchlistEntry(
                uid=str(row.sdn_uid),
                name=str(row.primary_name),
                program=str(row.program),
                entity_type=str(row.entity_type),
                dob=None if pd.isna(row.dob) else str(row.dob),
                nationality=None if pd.isna(row.nationality) else str(row.nationality),
                national_id=None if pd.isna(row.national_id) else str(row.national_id),
                aliases=tuple(a for a in str(row.aliases or "").split("|") if a),
            )
            for row in watchlist.itertuples()
        )
        engine = ScreeningEngine(config=app_config.screening)
        engine.index_watchlist(entries)
        service = cls(config=app_config, engine=engine, entries=entries)
        service._load_alerts(output_dir)
        return service

    def _load_alerts(self, output_dir: Path) -> None:
        """Hydrate the queue from the last pipeline run, if one exists.

        The API is a read/disposition surface over alerts the batch pipeline
        produced; it does not re-derive them. An empty queue is a valid state
        and is reported through /health rather than by failing requests.
        """
        screening_path = output_dir / "screening_alerts.csv"
        if screening_path.is_file():
            frame = pd.read_csv(screening_path)
            for position, row in enumerate(frame.itertuples(), start=1):
                alert_id = f"SCR-{position:07d}"
                self.alerts[alert_id] = AlertRecord(
                    alert_id=alert_id,
                    alert_type="SCREENING",
                    subject_id=str(row.subject_id),
                    rule_id=self.config.screening.rule_version,
                    period=None,
                    score=float(row.final_score),
                    decision=str(row.decision),
                    detail={
                        "subject_name": str(row.subject_name),
                        "watchlist_uid": str(row.watchlist_uid),
                        "watchlist_name": str(row.watchlist_name),
                        "program": str(row.program),
                        "matched_on": str(row.matched_on),
                        "name_score": float(row.name_score),
                    },
                )

        rules_path = output_dir / "rule_alerts.csv"
        if rules_path.is_file():
            frame = pd.read_csv(rules_path)
            for position, row in enumerate(frame.itertuples(), start=1):
                alert_id = f"TM-{position:07d}"
                self.alerts[alert_id] = AlertRecord(
                    alert_id=alert_id,
                    alert_type="TRANSACTION_MONITORING",
                    subject_id=str(row.customer_id),
                    rule_id=str(row.rule_id),
                    period=str(row.period),
                    score=float(row.metric_value),
                    decision="REVIEW",
                    detail={
                        "alert_amount": float(row.alert_amount),
                        "txn_count": int(row.txn_count),
                    },
                )
        logger.info("alert queue hydrated", extra={"alerts": len(self.alerts)})

    # -------------------------------------------------------------- screening
    def screen_name(
        self,
        name: str,
        subject_id: str,
        dob: str | None,
        nationality: str | None,
        national_id: str | None,
        top_n: int,
        min_score: float | None,
    ) -> tuple[list[MatchResult], dict[str, Any]]:
        subject = ScreeningSubject(
            subject_id=subject_id,
            name=name,
            dob=dob,
            nationality=nationality,
            national_id=national_id,
        )
        normalized = normalize_name(name, self.config.screening)
        candidates = (
            len(self.engine.blocking_index.candidates(normalized))
            if not normalized.is_empty()
            else 0
        )
        matches = self.engine.screen(subject, top_n=top_n, min_score=min_score)
        context = {
            "normalized_name": normalized.normalized,
            "detected_script": normalized.script.value,
            "latin_form": normalized.latin,
            "candidates_considered": candidates,
        }
        return matches, context

    def screen_transaction(
        self,
        txn_id: str,
        customer_id: str,
        amount: float,
        channel: str,
        direction: str,
        counterparty_name: str | None,
        counterparty_country: str | None,
    ) -> tuple[list[MatchResult], list[dict[str, str]]]:
        """Screen the counterparty and apply the single-transaction risk flags.

        Only the checks that can be evaluated from one transaction live here.
        Structuring, velocity and concentration are inherently multi-transaction
        and belong to the batch scenarios; approximating them per-transaction
        would produce a rule that looks the same and behaves differently.
        """
        matches: list[MatchResult] = []
        if counterparty_name:
            matches, _ = self.screen_name(
                name=counterparty_name,
                subject_id=customer_id,
                dob=None,
                nationality=counterparty_country,
                national_id=None,
                top_n=3,
                min_score=None,
            )

        flags: list[dict[str, str]] = []
        threshold = self.config.generation.ctr_threshold
        band = self.config.generation.amounts["structuring_band"]
        if amount >= threshold:
            flags.append(
                {
                    "code": "CTR_REPORTABLE",
                    "detail": f"Amount {amount:,.2f} meets the {threshold:,.0f} CTR threshold.",
                    "severity": "INFO",
                }
            )
        elif band[0] <= amount <= band[1] and channel == "CASH":
            flags.append(
                {
                    "code": "NEAR_CTR_CASH",
                    "detail": (
                        f"Cash amount {amount:,.2f} sits just below the "
                        f"{threshold:,.0f} threshold; single-transaction indicator only."
                    ),
                    "severity": "MEDIUM",
                }
            )
        if counterparty_country in self.config.generation.high_risk_countries:
            flags.append(
                {
                    "code": "HIGH_RISK_CORRIDOR",
                    "detail": f"Counterparty jurisdiction {counterparty_country} is high risk.",
                    "severity": "HIGH",
                }
            )
        if amount == round(amount / 1000.0) * 1000.0 and amount >= 1000 and channel in {"WIRE", "ACH"}:
            flags.append(
                {
                    "code": "ROUND_VALUE",
                    "detail": "Exact round-thousand value; weak on its own, corroborating in aggregate.",
                    "severity": "INFO",
                }
            )
        return matches, flags

    # ------------------------------------------------------------ alert queue
    def list_alerts(
        self,
        alert_type: str | None = None,
        subject_id: str | None = None,
        decision: str | None = None,
        undispositioned_only: bool = False,
        min_score: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[AlertRecord]]:
        selected = [
            record
            for record in self.alerts.values()
            if (alert_type is None or record.alert_type == alert_type)
            and (subject_id is None or record.subject_id == subject_id)
            and (decision is None or record.decision == decision)
            and (not undispositioned_only or record.disposition is None)
            and (min_score is None or record.score >= min_score)
        ]
        # Highest score first: the queue an analyst works, not insertion order.
        selected.sort(key=lambda r: (-r.score, r.alert_id))
        return len(selected), selected[offset : offset + limit]

    def get_alert(self, alert_id: str) -> AlertRecord:
        try:
            return self.alerts[alert_id]
        except KeyError as exc:
            raise AlertNotFoundError(alert_id) from exc

    def disposition(
        self,
        alert_id: str,
        disposition: str,
        analyst_id: str,
        note: str,
        analyst_level: str,
    ) -> tuple[AlertRecord, int, str]:
        record = self.get_alert(alert_id)
        if record.decision == "ESCALATE" and analyst_level != "L2":
            # Four-eyes in miniature: the band that goes straight to a sanctions
            # officer cannot be closed by the first-line queue.
            raise DispositionNotPermittedError(
                f"alert {alert_id} is an ESCALATE-band sanctions match and requires "
                "an L2 analyst to disposition"
            )
        record.disposition = disposition
        record.disposition_note = note
        record.disposition_analyst = analyst_id

        entry = self.audit.append(
            event_type="DISPOSITION",
            rule_version=record.rule_id,
            subject_id=record.subject_id,
            # Dispositions are the one event with a genuine external clock, but
            # the audit file is compared byte-for-byte across runs, so the
            # sequence number carries the ordering and the period carries the
            # business date.
            occurred_at=record.period or "batch",
            decision=disposition,
            input_snapshot={"alert_id": alert_id, "alert_type": record.alert_type, **record.detail},
            scoring_breakdown={
                "score": record.score,
                "original_decision": record.decision,
                "analyst_id": analyst_id,
                "analyst_level": analyst_level,
                "note": note,
            },
        )
        logger.info(
            "alert dispositioned",
            extra={
                "alert_id": alert_id,
                "disposition": disposition,
                "analyst_id": analyst_id,
                "audit_sequence": entry.sequence,
            },
        )
        return record, entry.sequence, entry.content_hash

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.alerts else "degraded",
            "watchlist_entities": len(self.entries),
            "reference_names": len(self.engine.reference_names),
            "alerts_loaded": len(self.alerts),
            "screening_rule_version": self.config.screening.rule_version,
            "monitoring_rule_version": self.config.monitoring.rule_version,
            "detail": None if self.alerts else "no batch alerts loaded; run `make run` first",
        }


def match_payload(match: MatchResult) -> dict[str, Any]:
    return {
        "watchlist_uid": match.watchlist_uid,
        "watchlist_name": match.watchlist_name,
        "program": match.program,
        "matched_on": match.matched_on,
        "name_score": round(match.name_score, 6),
        "corroboration_delta": round(match.corroboration_delta, 6),
        "final_score": round(match.final_score, 6),
        "decision": match.decision.value,
        "signals": {
            "jaro_winkler": round(match.signals.jaro_winkler, 6),
            "token_set": round(match.signals.token_set, 6),
            "phonetic": round(match.signals.phonetic, 6),
            "initials": round(match.signals.initials, 6),
            "order_swapped": match.signals.order_swapped,
        },
        "corroboration": {k: round(v, 6) for k, v in match.corroboration},
    }


def watchlist_columns() -> Sequence[str]:
    return WATCHLIST_SCHEMA.column_names


def alert_payload(record: AlertRecord) -> Mapping[str, Any]:
    return {
        "alert_id": record.alert_id,
        "alert_type": record.alert_type,
        "subject_id": record.subject_id,
        "rule_id": record.rule_id,
        "period": record.period,
        "score": round(record.score, 6),
        "decision": record.decision,
        "detail": record.detail,
        "disposition": record.disposition,
        "disposition_note": record.disposition_note,
    }
