"""Above-the-line / below-the-line threshold analysis.

ATL/BTL testing is how a monitoring programme defends its thresholds to a
regulator. Above the line asks "of the alerts this threshold produces, how many
are worth working?"; below the line asks the harder question, "what did this
threshold let through, and what would it cost to catch it?".

Both are answered here by re-executing the scenario's *own SQL* against a
rewritten parameter table. Re-implementing the rule logic in pandas to sweep it
would be the obvious shortcut and a serious one: the tuning evidence would then
describe a Python reimplementation rather than the query that actually runs in
production, and the two drift apart the first time either is edited.

The column a tuning committee actually decides on is
`marginal_reviews_per_extra_hit`: relaxing a threshold always finds more, and
the only question is what each additional confirmed hit costs in analyst time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.config import MonitoringConfig, ScenarioConfig
from src.logging_setup import get_logger
from src.monitoring.rules import RuleEngine

logger = get_logger(__name__)

SWEEP_COLUMNS: tuple[str, ...] = (
    "rule_id", "param_name", "param_value", "position", "alert_volume",
    "true_positives", "false_positives", "precision", "recall",
    "false_positive_ratio", "alert_rate", "delta_alerts", "delta_true_positives",
    "marginal_reviews_per_extra_hit",
)


@dataclass(frozen=True)
class SweepRow:
    rule_id: str
    param_name: str
    param_value: float
    position: str          # TIGHTER | BASELINE | LOOSER
    alert_volume: int
    true_positives: int
    false_positives: int
    precision: float
    recall: float
    false_positive_ratio: float
    alert_rate: float
    delta_alerts: int
    delta_true_positives: int
    marginal_reviews_per_extra_hit: float


def _score(
    alerts: pd.DataFrame, truth: set[tuple[str, str]], population: int, total_positives: int
) -> tuple[int, int, int, float, float, float, float]:
    cells = set(zip(alerts["customer_id"], alerts["period"])) if not alerts.empty else set()
    volume = len(cells)
    true_positives = len(cells & truth)
    false_positives = volume - true_positives
    precision = true_positives / volume if volume else 0.0
    recall = true_positives / total_positives if total_positives else 0.0
    ratio = false_positives / true_positives if true_positives else float("inf")
    return (
        volume, true_positives, false_positives, precision, recall, ratio,
        volume / population if population else 0.0,
    )


def sweep_scenario(
    engine: RuleEngine,
    scenario: ScenarioConfig,
    all_scenarios: Sequence[ScenarioConfig],
    truth: set[tuple[str, str]],
    population: int,
) -> list[SweepRow]:
    """Evaluate one scenario across the configured values of its swept parameter."""
    baseline_value = float(scenario.params[scenario.sweep_param])
    values = sorted({*scenario.sweep_values, baseline_value})
    others = [s for s in all_scenarios if s.id != scenario.id]
    total_positives = len(truth)

    measured: dict[float, tuple[int, int, int, float, float, float, float]] = {}
    for value in values:
        variant = scenario.with_param(scenario.sweep_param, value)
        # The whole parameter table is rewritten so the swept scenario reads its
        # new value while every other scenario keeps its baseline.
        engine.install_params([*others, variant])
        result = engine.run(variant)
        measured[value] = _score(result.alerts, truth, population, total_positives)

    baseline = measured[baseline_value]
    rows: list[SweepRow] = []
    for value in values:
        volume, tps, fps, precision, recall, ratio, rate = measured[value]
        delta_alerts = volume - baseline[0]
        delta_tps = tps - baseline[1]
        # Only meaningful when relaxing: the cost of the alerts you buy by
        # loosening. Tightening is reported as the mirror image, which is what
        # an ATL review needs in order to argue a threshold can be raised.
        marginal = (
            abs(delta_alerts) / abs(delta_tps) if delta_tps else float("inf") if delta_alerts else 0.0
        )
        rows.append(
            SweepRow(
                rule_id=scenario.id,
                param_name=scenario.sweep_param,
                param_value=value,
                position=(
                    "BASELINE" if value == baseline_value
                    else "LOOSER" if volume > baseline[0]
                    else "TIGHTER"
                ),
                alert_volume=volume,
                true_positives=tps,
                false_positives=fps,
                precision=precision,
                recall=recall,
                false_positive_ratio=ratio,
                alert_rate=rate,
                delta_alerts=delta_alerts,
                delta_true_positives=delta_tps,
                marginal_reviews_per_extra_hit=marginal,
            )
        )
    logger.info(
        "threshold sweep complete",
        extra={
            "rule_id": scenario.id,
            "param": scenario.sweep_param,
            "values_tested": len(values),
            "baseline_alerts": baseline[0],
            "baseline_precision": round(baseline[3], 4),
        },
    )
    return rows


def run_atl_btl(
    engine: RuleEngine,
    config: MonitoringConfig,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Sweep every scenario and return the tuning table."""
    truth = set(
        zip(
            labels.loc[labels["is_true_anomaly"], "customer_id"],
            labels.loc[labels["is_true_anomaly"], "period"],
        )
    )
    population = len(labels)
    rows: list[SweepRow] = []
    for scenario in config.scenarios:
        rows.extend(sweep_scenario(engine, scenario, config.scenarios, truth, population))

    # Leave the engine holding the baseline parameters: a sweep must not have
    # side effects on whatever runs next.
    engine.install_params(config.scenarios)

    frame = pd.DataFrame([r.__dict__ for r in rows], columns=list(SWEEP_COLUMNS))
    for column in ("precision", "recall", "false_positive_ratio", "alert_rate",
                   "marginal_reviews_per_extra_hit"):
        frame[column] = frame[column].round(6)
    return frame.sort_values(["rule_id", "param_value"], kind="mergesort").reset_index(drop=True)


def recommended_thresholds(sweep: pd.DataFrame, max_reviews_per_hit: float = 40.0) -> pd.DataFrame:
    """Loosest setting per rule whose marginal cost stays inside the review budget.

    This is a decision *aid*, not a decision. A real tuning committee weighs
    typology coverage, regulatory expectation and analyst capacity; the point of
    surfacing it is that the recommendation is reproducible and its basis is
    visible in the same table.
    """
    picks: list[dict[str, object]] = []
    for rule_id, group in sweep.groupby("rule_id", sort=True):
        baseline = group[group["position"] == "BASELINE"].iloc[0]
        affordable = group[
            (group["marginal_reviews_per_extra_hit"] <= max_reviews_per_hit)
            & (group["true_positives"] >= baseline["true_positives"])
        ]
        chosen = (
            affordable.sort_values("recall", kind="mergesort").iloc[-1]
            if not affordable.empty
            else baseline
        )
        # What the cheapest genuine relaxation would cost, whether or not it was
        # affordable. Without this the report cannot distinguish "the baseline
        # is already optimal" from "every relaxation was priced out", and those
        # are very different findings for a tuning committee.
        relaxations = group[group["delta_true_positives"] > 0]
        cheapest = (
            float(relaxations["marginal_reviews_per_extra_hit"].min())
            if not relaxations.empty
            else float("inf")
        )
        picks.append(
            {
                "rule_id": rule_id,
                "param_name": str(baseline["param_name"]),
                "baseline_value": float(baseline["param_value"]),
                "recommended_value": float(chosen["param_value"]),
                "baseline_alerts": int(baseline["alert_volume"]),
                "recommended_alerts": int(chosen["alert_volume"]),
                "baseline_recall": float(baseline["recall"]),
                "recommended_recall": float(chosen["recall"]),
                "marginal_reviews_per_extra_hit": float(
                    chosen["marginal_reviews_per_extra_hit"]
                ),
                "cheapest_relaxation_reviews_per_hit": cheapest,
                "relaxation_affordable": bool(cheapest <= max_reviews_per_hit),
            }
        )
    return pd.DataFrame(picks)


def sweep_summary(sweep: pd.DataFrame) -> Mapping[str, object]:
    baseline = sweep[sweep["position"] == "BASELINE"]
    return {
        "rules_swept": int(sweep["rule_id"].nunique()),
        "settings_evaluated": int(len(sweep)),
        "baseline_total_alerts": int(baseline["alert_volume"].sum()),
        "baseline_mean_precision": round(float(baseline["precision"].mean()), 6),
        "baseline_union_recall": None,  # filled by the pipeline, which owns the union
    }


def union_recall(alert_cells: Iterable[tuple[str, str]], truth: set[tuple[str, str]]) -> float:
    cells = set(alert_cells)
    return len(cells & truth) / len(truth) if truth else 0.0
