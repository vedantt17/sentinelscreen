"""DuckDB scenario engine.

Every threshold a scenario uses is read from the `rule_params` table at query
time. Nothing is interpolated into the SQL string and nothing is hardcoded in
it. That is not a stylistic preference: the ATL/BTL analyzer re-runs the exact
same SQL text against a rewritten parameter table, so any threshold that lived
in the query would silently escape the sweep and the resulting tuning evidence
would be wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import duckdb
import pandas as pd

from src.config import MonitoringConfig, ScenarioConfig
from src.exceptions import RuleExecutionError
from src.logging_setup import get_logger
from src.paths import SQL_DIR

logger = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "rule_id", "customer_id", "period", "metric_value", "alert_amount", "txn_count",
)


@lru_cache(maxsize=32)
def load_scenario_sql(rule_id: str, sql_dir: Path = SQL_DIR) -> str:
    path = sql_dir / f"{rule_id}.sql"
    if not path.is_file():
        raise RuleExecutionError(rule_id, f"no SQL file at {path}")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class ScenarioResult:
    rule_id: str
    alerts: pd.DataFrame
    params: Mapping[str, float]
    elapsed_seconds: float

    @property
    def alert_count(self) -> int:
        return len(self.alerts)


class RuleEngine:
    """Executes the eight monitoring scenarios against a loaded DuckDB session."""

    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection,
        config: MonitoringConfig,
        sql_dir: Path = SQL_DIR,
    ) -> None:
        self.connection = connection
        self.config = config
        self.sql_dir = sql_dir

    # ------------------------------------------------------------- parameters
    def install_params(self, scenarios: Sequence[ScenarioConfig] | None = None) -> None:
        """(Re)write the parameter table that every scenario joins against."""
        target = scenarios if scenarios is not None else self.config.scenarios
        rows = [
            {"rule_id": scenario.id, "param_name": name, "param_value": float(value)}
            for scenario in target
            for name, value in sorted(scenario.params.items())
        ]
        frame = pd.DataFrame(rows, columns=["rule_id", "param_name", "param_value"])
        self.connection.execute(
            "CREATE OR REPLACE TABLE rule_params ("
            " rule_id VARCHAR NOT NULL, param_name VARCHAR NOT NULL, param_value DOUBLE NOT NULL);"
        )
        self.connection.register("_stage_rule_params", frame)
        self.connection.execute("INSERT INTO rule_params SELECT * FROM _stage_rule_params;")
        self.connection.unregister("_stage_rule_params")

    def install_risk_countries(self, corridors: Mapping[str, Sequence[str]]) -> None:
        """Country risk tiering as data.

        A FATF list revision should be a row change, not a code deployment, and
        keeping it in a table is what lets the corridor scenario be re-tuned
        without touching the SQL under change control.
        """
        rows = [
            {"country": country, "tier": tier.replace("_risk", "").upper()}
            for tier, countries in corridors.items()
            for country in countries
        ]
        frame = pd.DataFrame(rows, columns=["country", "tier"])
        self.connection.execute(
            "CREATE OR REPLACE TABLE risk_countries (country VARCHAR NOT NULL, tier VARCHAR NOT NULL);"
        )
        self.connection.register("_stage_risk_countries", frame)
        self.connection.execute("INSERT INTO risk_countries SELECT * FROM _stage_risk_countries;")
        self.connection.unregister("_stage_risk_countries")

    # -------------------------------------------------------------- execution
    def run(self, scenario: ScenarioConfig) -> ScenarioResult:
        sql = load_scenario_sql(scenario.id, self.sql_dir)
        start = time.perf_counter()
        try:
            alerts = self.connection.execute(sql).fetchdf()
        except duckdb.Error as exc:
            raise RuleExecutionError(scenario.id, str(exc)) from exc
        elapsed = time.perf_counter() - start

        missing = [c for c in REQUIRED_COLUMNS if c not in alerts.columns]
        if missing:
            raise RuleExecutionError(
                scenario.id, f"result is missing required columns: {missing}"
            )
        if not alerts.empty and (alerts["rule_id"] != scenario.id).any():
            raise RuleExecutionError(scenario.id, "result rows carry a foreign rule_id")

        # Deterministic ordering: DuckDB makes no ordering guarantee without an
        # ORDER BY, and the alert file is hashed by the determinism check.
        alerts = alerts.sort_values(["customer_id", "period"], kind="mergesort").reset_index(
            drop=True
        )
        logger.info(
            "scenario executed",
            extra={
                "rule_id": scenario.id,
                "alerts": len(alerts),
                "elapsed_seconds": round(elapsed, 3),
                "params": dict(scenario.params),
            },
        )
        return ScenarioResult(
            rule_id=scenario.id,
            alerts=alerts,
            params=dict(scenario.params),
            elapsed_seconds=elapsed,
        )

    def run_all(self, scenarios: Sequence[ScenarioConfig] | None = None) -> dict[str, ScenarioResult]:
        target = list(scenarios if scenarios is not None else self.config.scenarios)
        self.install_params(target)
        return {scenario.id: self.run(scenario) for scenario in target}


def combine_alerts(results: Mapping[str, ScenarioResult]) -> pd.DataFrame:
    """Stack every scenario's alerts into one queue, ordered deterministically."""
    frames = [r.alerts for r in results.values() if not r.alerts.empty]
    if not frames:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(
        ["customer_id", "period", "rule_id"], kind="mergesort"
    ).reset_index(drop=True)
