"""Data-quality control execution.

The checks themselves live in `config/dq_checks.yaml` as SQL, not in Python.
That is a governance choice: a data-quality control set is reviewed and signed
off by people who read SQL and do not read Python, and a control that has to be
re-implemented in order to be understood will not be reviewed properly.

Severity decides what happens next. A BLOCKER failure aborts the pipeline
because continuing would produce artefacts that look authoritative and are not —
a referential-integrity break silently drops alerts, and nothing downstream
would show it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import duckdb
import pandas as pd

from src.config import DataQualityCheck
from src.exceptions import DataQualityError
from src.logging_setup import get_logger

logger = get_logger(__name__)

# `elapsed_seconds` is deliberately absent: the report is hashed by the
# reproducibility check, and a duration differs between two identical runs.
# Timings are collected separately into outputs/run_timings.json.
RESULT_COLUMNS: tuple[str, ...] = (
    "check_id", "dimension", "severity", "description", "failing_rows", "status",
)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    dimension: str
    severity: str
    description: str
    failing_rows: int
    elapsed_seconds: float
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "ERROR"
        return "PASS" if self.failing_rows == 0 else "FAIL"

    @property
    def is_blocking_failure(self) -> bool:
        return self.severity == "BLOCKER" and self.status != "PASS"

    def as_row(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "dimension": self.dimension,
            "severity": self.severity,
            "description": self.description,
            "failing_rows": self.failing_rows,
            "status": self.status,
        }


def run_checks(
    connection: duckdb.DuckDBPyConnection, checks: Sequence[DataQualityCheck]
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in checks:
        start = time.perf_counter()
        try:
            row = connection.execute(check.sql).fetchone()
            failing = int(row[0]) if row and row[0] is not None else 0
            error: str | None = None
        except duckdb.Error as exc:
            # A control that cannot execute is a failed control, not a skipped
            # one. Swallowing the error would report a clean bill of health.
            failing = -1
            error = str(exc)
        results.append(
            CheckResult(
                check_id=check.id,
                dimension=check.dimension,
                severity=check.severity,
                description=check.description,
                failing_rows=failing,
                elapsed_seconds=time.perf_counter() - start,
                error=error,
            )
        )
    return results


def to_frame(results: Sequence[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results], columns=list(RESULT_COLUMNS))


def timings(results: Sequence[CheckResult]) -> dict[str, float]:
    return {r.check_id: round(r.elapsed_seconds, 4) for r in results}


def enforce(results: Sequence[CheckResult]) -> None:
    blocking = [r.check_id for r in results if r.is_blocking_failure]
    failures = [r for r in results if r.status != "PASS"]
    for result in failures:
        logger.warning(
            "data-quality control failed",
            extra={
                "check_id": result.check_id,
                "severity": result.severity,
                "failing_rows": result.failing_rows,
                "error": result.error,
            },
        )
    logger.info(
        "data-quality run complete",
        extra={
            "checks": len(results),
            "passed": sum(1 for r in results if r.status == "PASS"),
            "failed": len(failures),
            "blocking": len(blocking),
        },
    )
    if blocking:
        raise DataQualityError(blocking)
