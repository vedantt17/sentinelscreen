"""Request and response models.

Pydantic is used here and nowhere else. At the API boundary the input is
untrusted and arrives one at a time, which is exactly the situation pydantic is
good at; over a 500,000-row dataframe it would cost minutes and buy nothing the
column contracts in `src/data/schemas.py` do not already give.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Decision = Literal["ESCALATE", "REVIEW", "DISCARD"]
DispositionCode = Literal["TRUE_POSITIVE", "FALSE_POSITIVE", "ESCALATED_L2", "PENDING_INFO"]


class NameScreenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=300, description="Name as captured, any script")
    dob: str | None = Field(
        default=None,
        description="ISO date or year-only. Partial dates are normal on watchlist feeds.",
    )
    nationality: str | None = Field(default=None, min_length=2, max_length=2)
    national_id: str | None = Field(default=None, max_length=64)
    subject_id: str = Field(default="AD-HOC", max_length=64)
    top_n: int = Field(default=5, ge=1, le=25)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("nationality")
    @classmethod
    def _upper(cls, value: str | None) -> str | None:
        return value.upper() if value else value

    @field_validator("dob")
    @classmethod
    def _dob_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not (len(text) == 4 and text.isdigit()):
            # Anything other than a bare year must parse as a full ISO date;
            # a silently unparsed DOB would disable the strongest corroborator.
            try:
                datetime.strptime(text, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError("dob must be YYYY or YYYY-MM-DD") from exc
        return text


class SignalBreakdown(BaseModel):
    jaro_winkler: float
    token_set: float
    phonetic: float
    initials: float
    order_swapped: bool


class NameMatch(BaseModel):
    watchlist_uid: str
    watchlist_name: str
    program: str
    matched_on: str = Field(description="'primary' or 'alias' — which reference name matched")
    name_score: float
    corroboration_delta: float
    final_score: float
    decision: Decision
    signals: SignalBreakdown
    corroboration: dict[str, float]


class NameScreenResponse(BaseModel):
    subject_id: str
    query_name: str
    normalized_name: str
    detected_script: str
    latin_form: str
    rule_version: str
    candidates_considered: int
    matches: list[NameMatch]


class TransactionScreenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    txn_id: str = Field(max_length=64)
    customer_id: str = Field(max_length=64)
    amount: float = Field(gt=0, le=50_000_000)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    channel: Literal["WIRE", "ACH", "CARD", "CASH"]
    direction: Literal["CREDIT", "DEBIT"]
    counterparty_name: str | None = Field(default=None, max_length=300)
    counterparty_country: str | None = Field(default=None, min_length=2, max_length=2)

    @field_validator("counterparty_country")
    @classmethod
    def _upper(cls, value: str | None) -> str | None:
        return value.upper() if value else value


class TransactionRiskFlag(BaseModel):
    code: str
    detail: str
    severity: Literal["INFO", "MEDIUM", "HIGH"]


class TransactionScreenResponse(BaseModel):
    txn_id: str
    customer_id: str
    rule_version: str
    counterparty_matches: list[NameMatch]
    flags: list[TransactionRiskFlag]
    requires_review: bool


class Alert(BaseModel):
    alert_id: str
    alert_type: Literal["SCREENING", "TRANSACTION_MONITORING"]
    subject_id: str
    rule_id: str
    period: str | None = None
    score: float
    decision: str
    detail: dict[str, str | float | int | bool]
    disposition: DispositionCode | None = None
    disposition_note: str | None = None


class AlertPage(BaseModel):
    total: int
    limit: int
    offset: int
    alerts: list[Alert]


class DispositionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: DispositionCode
    analyst_id: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=2000)
    # An L1 analyst may not close a sanctions escalation; that is an L2 action.
    # The API records the claim and the service layer enforces the rule.
    analyst_level: Literal["L1", "L2"] = "L1"


class DispositionResponse(BaseModel):
    alert_id: str
    disposition: DispositionCode
    analyst_id: str
    audit_sequence: int
    audit_hash: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    watchlist_entities: int
    reference_names: int
    alerts_loaded: int
    screening_rule_version: str
    monitoring_rule_version: str
    detail: str | None = None
