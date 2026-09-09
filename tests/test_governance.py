"""Audit chain, data-quality controls, and deterministic artefact writing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import AppConfig, DataQualityCheck
from src.exceptions import AuditIntegrityError, ConfigurationError, DataQualityError
from src.pipeline.artifacts import build_manifest, diff_manifests, sha256_file, write_csv, write_json
from src.pipeline.audit import GENESIS_HASH, AuditLog, AuditRecord, derive_run_id
from src.pipeline.quality import enforce, run_checks, to_frame


# ------------------------------------------------------------------ audit log
def _log(entries: int = 3) -> AuditLog:
    log = AuditLog("test-run")
    for index in range(entries):
        log.append(
            event_type="SCREENING_ALERT",
            rule_version="screening-v1.3.0",
            subject_id=f"CUST-{index:06d}",
            occurred_at="2024-12-31",
            decision="REVIEW",
            input_snapshot={"name": f"Subject {index}"},
            scoring_breakdown={"final_score": 0.9 + index / 100},
        )
    return log


def test_chain_verifies_when_untouched() -> None:
    log = _log()
    log.verify()
    assert len(log) == 3


def test_first_record_commits_to_the_genesis_hash() -> None:
    log = _log(1)
    assert next(iter(log)).previous_hash == GENESIS_HASH


def test_each_record_commits_to_its_predecessor() -> None:
    records = list(_log(4))
    for previous, current in zip(records, records[1:]):
        assert current.previous_hash == previous.content_hash


def test_editing_a_record_breaks_the_chain() -> None:
    log = _log()
    records = log._records
    tampered = AuditRecord(
        **{**records[1].payload(), "decision": "DISCARD"},
        content_hash=records[1].content_hash,
    )
    records[1] = tampered
    with pytest.raises(AuditIntegrityError, match="does not match its hash"):
        log.verify()


def test_removing_a_record_breaks_the_chain() -> None:
    log = _log(4)
    del log._records[1]
    with pytest.raises(AuditIntegrityError):
        log.verify()


def test_jsonl_round_trip_preserves_the_chain(tmp_path: Path) -> None:
    log = _log(5)
    path = log.write_jsonl(tmp_path / "audit.jsonl")
    restored = AuditLog.read_jsonl(path)
    restored.verify()
    assert restored.head_hash == log.head_hash


def test_audit_records_contain_no_wall_clock(tmp_path: Path) -> None:
    """Two runs must produce identical audit bytes, so nothing may read the clock."""
    first = _log().write_jsonl(tmp_path / "a.jsonl")
    second = _log().write_jsonl(tmp_path / "b.jsonl")
    assert sha256_file(first) == sha256_file(second)


def test_run_id_is_a_function_of_seed_and_versions() -> None:
    assert derive_run_id(42, ["a", "b"]) == derive_run_id(42, ["b", "a"])
    assert derive_run_id(42, ["a"]) != derive_run_id(43, ["a"])


def test_float_formatting_is_pinned() -> None:
    """Otherwise the same score hashes differently across platforms."""
    log = AuditLog("t")
    first = log.append(
        "X", "v1", "S1", "2024-01-01", "REVIEW", {}, {"score": 0.1 + 0.2}
    )
    other = AuditLog("t").append("X", "v1", "S1", "2024-01-01", "REVIEW", {}, {"score": 0.3})
    assert first.content_hash == other.content_hash


# --------------------------------------------------------------- data quality
def test_all_ten_controls_execute(loaded_db, app_config: AppConfig) -> None:  # type: ignore[no-untyped-def]
    results = run_checks(loaded_db, app_config.dq_checks)
    assert len(results) == 10
    assert {r.status for r in results} <= {"PASS", "FAIL", "ERROR"}


def test_controls_pass_on_generated_data(loaded_db, app_config: AppConfig) -> None:  # type: ignore[no-untyped-def]
    results = run_checks(loaded_db, app_config.dq_checks)
    failures = [(r.check_id, r.failing_rows, r.error) for r in results if r.status != "PASS"]
    assert not failures, f"generated data violates its own controls: {failures}"


def test_blocking_failure_aborts(loaded_db, app_config: AppConfig) -> None:  # type: ignore[no-untyped-def]
    """A BLOCKER failure must stop the pipeline, not be recorded and ignored.

    The violation used here is a negative amount rather than a duplicate key,
    because DuckDB enforces the primary key itself — the offending INSERT would
    be rejected by the database before the control ever ran.
    """
    loaded_db.execute(
        "UPDATE transactions SET amount = -1.0 "
        "WHERE txn_id = (SELECT MIN(txn_id) FROM transactions);"
    )
    results = run_checks(loaded_db, app_config.dq_checks)
    assert any(r.check_id == "DQ05_AMOUNT_RANGE" and r.status == "FAIL" for r in results)
    with pytest.raises(DataQualityError, match="DQ05"):
        enforce(results)


def test_a_control_that_cannot_execute_counts_as_failed(loaded_db) -> None:  # type: ignore[no-untyped-def]
    """Swallowing the error would report a clean bill of health."""
    broken = DataQualityCheck(
        id="DQ99_BROKEN", dimension="validity", severity="CRITICAL",
        description="references a table that does not exist",
        sql="SELECT COUNT(*) FROM table_that_does_not_exist",
    )
    result = run_checks(loaded_db, [broken])[0]
    assert result.status == "ERROR"
    assert result.error is not None


def test_unknown_severity_is_rejected_at_load() -> None:
    with pytest.raises(ConfigurationError, match="unknown severity"):
        DataQualityCheck(
            id="X", dimension="validity", severity="SEVERE", description="", sql="SELECT 0"
        )


def test_report_frame_has_a_row_per_control(loaded_db, app_config: AppConfig) -> None:  # type: ignore[no-untyped-def]
    frame = to_frame(run_checks(loaded_db, app_config.dq_checks))
    assert len(frame) == 10
    assert set(frame["check_id"]) == {c.id for c in app_config.dq_checks}


# ------------------------------------------------------------------ artefacts
def test_json_writes_are_byte_stable(tmp_path: Path) -> None:
    payload = {"b": 2, "a": 1.0 / 3.0, "nested": {"z": [1.5, 2.5]}}
    first = write_json(tmp_path / "a.json", payload)
    second = write_json(tmp_path / "b.json", dict(reversed(list(payload.items()))))
    assert sha256_file(first) == sha256_file(second)


def test_csv_writes_are_byte_stable(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1.0 / 3.0, 2.0 / 3.0], "y": ["a", "b"]})
    first = write_csv(tmp_path / "a.csv", frame)
    second = write_csv(tmp_path / "b.csv", frame.copy())
    assert sha256_file(first) == sha256_file(second)


def test_csv_line_endings_are_pinned(tmp_path: Path) -> None:
    """Windows would otherwise emit CRLF and CI would emit LF."""
    path = write_csv(tmp_path / "a.csv", pd.DataFrame({"x": [1, 2]}))
    assert b"\r\n" not in path.read_bytes()


def test_manifest_excludes_volatile_artefacts(tmp_path: Path) -> None:
    write_json(tmp_path / "run_info.json", {"generated_at": "now"})
    write_json(tmp_path / "metrics.json", {"a": 1})
    manifest = build_manifest(list(tmp_path.glob("*.json")), tmp_path)
    assert "metrics.json" in manifest["artefacts"]
    assert "run_info.json" not in manifest["artefacts"]


def test_manifest_diff_detects_a_changed_artefact(tmp_path: Path) -> None:
    left_dir = tmp_path / "l"
    right_dir = tmp_path / "r"
    left_dir.mkdir()
    right_dir.mkdir()
    write_json(left_dir / "m.json", {"a": 1})
    write_json(right_dir / "m.json", {"a": 2})
    left = build_manifest(list(left_dir.glob("*")), left_dir)
    right = build_manifest(list(right_dir.glob("*")), right_dir)
    problems = diff_manifests(left, right)
    assert problems and "differs: m.json" in problems[0]


def test_manifest_diff_is_clean_for_identical_runs(tmp_path: Path) -> None:
    left_dir = tmp_path / "l"
    right_dir = tmp_path / "r"
    left_dir.mkdir()
    right_dir.mkdir()
    for directory in (left_dir, right_dir):
        write_json(directory / "m.json", {"a": 1, "b": [1, 2, 3]})
    left = build_manifest(list(left_dir.glob("*")), left_dir)
    right = build_manifest(list(right_dir.glob("*")), right_dir)
    assert diff_manifests(left, right) == []
