"""Immutable audit records.

Every alert this system raises has to be explainable years later, to somebody
who was not there: which rule version fired, what the input looked like at the
time, and exactly how the score was arrived at. That is the whole content of
`AuditRecord`, and it is why the record carries an input *snapshot* rather than
a foreign key — the customer row will have changed by the time anyone asks.

Records are chained by hash. Each one commits to its predecessor, so removing or
editing an entry after the fact invalidates every record that follows it and
`AuditLog.verify()` says exactly where the chain broke. This is deliberately
cheap tamper *evidence*, not tamper proofing: it detects after-the-fact edits to
the file, which is the realistic threat for an audit artefact.

Nothing here reads the wall clock. `occurred_at` is derived from the data being
audited, so two seeded runs produce byte-identical audit files and `make
reproduce` can compare them directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from src.exceptions import AuditIntegrityError
from src.logging_setup import get_logger

logger = get_logger(__name__)

GENESIS_HASH: str = "0" * 64


def _canonical(payload: Mapping[str, Any]) -> str:
    """Stable JSON: sorted keys, no incidental whitespace, floats rounded.

    Float formatting has to be pinned or the hash chain becomes platform
    dependent — the same score can render as 0.8300000000000001 on one machine
    and 0.83 on another.
    """
    def normalise(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 8)
        if isinstance(value, dict):
            return {k: normalise(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalise(v) for v in value]
        return value

    return json.dumps(normalise(dict(payload)), sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class AuditRecord:
    sequence: int
    event_type: str
    rule_version: str
    subject_id: str
    occurred_at: str
    decision: str
    input_snapshot: Mapping[str, Any]
    scoring_breakdown: Mapping[str, Any]
    previous_hash: str
    content_hash: str = field(default="")

    def payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "rule_version": self.rule_version,
            "subject_id": self.subject_id,
            "occurred_at": self.occurred_at,
            "decision": self.decision,
            "input_snapshot": dict(self.input_snapshot),
            "scoring_breakdown": dict(self.scoring_breakdown),
            "previous_hash": self.previous_hash,
        }

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical(self.payload()).encode("utf-8")).hexdigest()

    def sealed(self) -> "AuditRecord":
        return AuditRecord(**{**self.payload(), "content_hash": self.compute_hash()})

    def to_json(self) -> str:
        return json.dumps(
            {**self.payload(), "content_hash": self.content_hash},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


class AuditLog:
    """Append-only hash chain of audit records."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._records: list[AuditRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[AuditRecord]:
        return iter(self._records)

    @property
    def head_hash(self) -> str:
        return self._records[-1].content_hash if self._records else GENESIS_HASH

    def append(
        self,
        event_type: str,
        rule_version: str,
        subject_id: str,
        occurred_at: str,
        decision: str,
        input_snapshot: Mapping[str, Any],
        scoring_breakdown: Mapping[str, Any],
    ) -> AuditRecord:
        record = AuditRecord(
            sequence=len(self._records),
            event_type=event_type,
            rule_version=rule_version,
            subject_id=subject_id,
            occurred_at=occurred_at,
            decision=decision,
            input_snapshot=dict(input_snapshot),
            scoring_breakdown=dict(scoring_breakdown),
            previous_hash=self.head_hash,
        ).sealed()
        self._records.append(record)
        return record

    def verify(self) -> None:
        """Recompute the chain; raise at the first record that does not agree."""
        expected_previous = GENESIS_HASH
        for index, record in enumerate(self._records):
            if record.sequence != index:
                raise AuditIntegrityError(
                    f"record at position {index} claims sequence {record.sequence}"
                )
            if record.previous_hash != expected_previous:
                raise AuditIntegrityError(
                    f"chain broken at sequence {record.sequence}: "
                    f"expected previous {expected_previous[:12]}..., "
                    f"found {record.previous_hash[:12]}..."
                )
            recomputed = record.compute_hash()
            if recomputed != record.content_hash:
                raise AuditIntegrityError(
                    f"record {record.sequence} content does not match its hash "
                    f"({recomputed[:12]}... vs {record.content_hash[:12]}...)"
                )
            expected_previous = record.content_hash

    def write_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in self._records:
                handle.write(record.to_json())
                handle.write("\n")
        logger.info(
            "audit log written",
            extra={"path": str(path), "records": len(self._records), "head": self.head_hash[:16]},
        )
        return path

    @classmethod
    def read_jsonl(cls, path: Path, run_id: str = "restored") -> "AuditLog":
        log = cls(run_id)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            content_hash = raw.pop("content_hash")
            log._records.append(AuditRecord(**raw, content_hash=content_hash))
        return log

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_decision: dict[str, int] = {}
        for record in self._records:
            by_type[record.event_type] = by_type.get(record.event_type, 0) + 1
            by_decision[record.decision] = by_decision.get(record.decision, 0) + 1
        return {
            "run_id": self.run_id,
            "records": len(self._records),
            "head_hash": self.head_hash,
            "by_event_type": dict(sorted(by_type.items())),
            "by_decision": dict(sorted(by_decision.items())),
        }


def derive_run_id(seed: int, versions: Sequence[str]) -> str:
    """Deterministic run identifier: same seed and same rule versions, same id.

    A UUID would be the obvious choice and would break `make reproduce`, since
    the run id is written into every audit record.
    """
    material = "|".join([str(seed), *sorted(versions)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
