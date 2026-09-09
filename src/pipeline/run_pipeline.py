"""End-to-end orchestration.

Stage order is a dependency order, not a preference:

    generate -> validate -> load -> data quality -> screening -> scenarios
             -> features -> split -> models -> ATL/BTL -> artefacts

Data quality runs before anything reads the tables, because every metric
downstream is meaningless if referential integrity is broken. The train/test
split happens before any baseline is fitted, and the ATL/BTL sweep runs last
because it rewrites the scenario parameter table and must not leave the engine
in a tuned state for anything that follows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config import AppConfig, load_app_config
from src.data.generator import GeneratedDataset, generate_dataset
from src.data.loaders import connect, create_run_window, load_frames, write_parquet
from src.logging_setup import get_logger
from src.monitoring.atl_btl import recommended_thresholds, run_atl_btl
from src.monitoring.diagnostics import separability_report
from src.monitoring.features import build_base_features, out_of_time_mask, prepare_matrices
from src.monitoring.models import TrainedModels, evaluate, train_all
from src.monitoring.rules import RuleEngine, ScenarioResult, combine_alerts
from src.paths import DATA_DIR, OUTPUT_DIR
from src.pipeline.artifacts import build_manifest, write_csv, write_json
from src.pipeline.audit import AuditLog, derive_run_id
from src.pipeline.quality import enforce, run_checks, timings as dq_timings, to_frame
from src.sanctions.engine import (
    MatchResult,
    ScreeningEngine,
    ScreeningRun,
    ScreeningSubject,
    WatchlistEntry,
)

logger = get_logger(__name__)

SCREENING_SWEEP_GRID: tuple[float, ...] = (
    0.70, 0.74, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94,
)

# Alerts are generated down to the bottom of the sweep grid, not down to the
# production threshold. Emitting only what already clears review would make the
# sweep flat below that point and hide exactly the trade-off it exists to show.
SCREENING_FLOOR: float = min(SCREENING_SWEEP_GRID)

# Below this the out-of-time ROC-AUC is noise rather than a measurement.
MIN_OUT_OF_TIME_POSITIVES: int = 5


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]
    artefacts: tuple[Path, ...]
    elapsed_seconds: float


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
def _watchlist_entries(watchlist: pd.DataFrame) -> list[WatchlistEntry]:
    return [
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
    ]


def _subjects(customers: pd.DataFrame) -> list[ScreeningSubject]:
    return [
        ScreeningSubject(
            subject_id=str(row.customer_id),
            name=str(row.full_name),
            dob=None if pd.isna(row.dob) else str(row.dob),
            nationality=None if pd.isna(row.nationality) else str(row.nationality),
            national_id=None if pd.isna(row.national_id) else str(row.national_id),
        )
        for row in customers.itertuples()
    ]


def _screening_frame(matches: Sequence[MatchResult]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame(
            columns=[
                "subject_id", "subject_name", "watchlist_uid", "watchlist_name", "program",
                "matched_on", "name_score", "corroboration_delta", "final_score", "decision",
                "jaro_winkler", "token_set", "phonetic", "initials", "order_swapped",
            ]
        )
    return pd.DataFrame(
        [
            {
                "subject_id": m.subject_id,
                "subject_name": m.subject_name,
                "watchlist_uid": m.watchlist_uid,
                "watchlist_name": m.watchlist_name,
                "program": m.program,
                "matched_on": m.matched_on,
                "name_score": round(m.name_score, 6),
                "corroboration_delta": round(m.corroboration_delta, 6),
                "final_score": round(m.final_score, 6),
                "decision": m.decision.value,
                "jaro_winkler": round(m.signals.jaro_winkler, 6),
                "token_set": round(m.signals.token_set, 6),
                "phonetic": round(m.signals.phonetic, 6),
                "initials": round(m.signals.initials, 6),
                "order_swapped": m.signals.order_swapped,
            }
            for m in matches
        ]
    )


def _screening_metrics(
    alerts: pd.DataFrame, customers: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Precision and recall against the planted screening ground truth.

    Measured at entity level: a customer counts as detected when the correct
    SDN uid appears among their alerts, not merely when they alerted at all.
    Alerting on the wrong designated party is a false positive, and a screening
    metric that ignores which entity matched would hide that.
    """
    truth = {
        str(row.customer_id): str(row.watchlist_true_match_uid)
        for row in customers.itertuples()
        if not pd.isna(row.watchlist_true_match_uid)
    }
    near_miss = {
        str(row.customer_id)
        for row in customers.itertuples()
        if bool(row.is_screening_near_miss)
    }
    surviving = alerts[alerts["final_score"] >= threshold]
    by_subject: dict[str, set[str]] = {}
    for subject_id, uid in zip(surviving["subject_id"], surviving["watchlist_uid"]):
        by_subject.setdefault(str(subject_id), set()).add(str(uid))

    detected = sum(1 for cid, uid in truth.items() if uid in by_subject.get(cid, set()))
    alerted_subjects = len(by_subject)
    false_positive_subjects = alerted_subjects - detected
    near_miss_alerted = sum(1 for cid in by_subject if cid in near_miss)
    return {
        "threshold": round(threshold, 4),
        "planted_true_matches": len(truth),
        "detected_true_matches": detected,
        "recall": round(detected / len(truth), 6) if truth else 0.0,
        "alerted_subjects": alerted_subjects,
        "alert_rows": int(len(surviving)),
        "alert_rate": round(alerted_subjects / len(customers), 6),
        "precision": round(detected / alerted_subjects, 6) if alerted_subjects else 0.0,
        "false_positive_ratio": (
            round(false_positive_subjects / detected, 4) if detected else float("inf")
        ),
        "planted_near_misses": len(near_miss),
        "near_misses_alerted": near_miss_alerted,
        "near_miss_alert_rate": (
            round(near_miss_alerted / len(near_miss), 6) if near_miss else 0.0
        ),
        "escalations": int((surviving["decision"] == "ESCALATE").sum()),
    }


def _screening_sweep(alerts: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """The screening equivalent of ATL/BTL: what each threshold buys and costs."""
    rows = [_screening_metrics(alerts, customers, t) for t in SCREENING_SWEEP_GRID]
    return pd.DataFrame(rows)


def _blocking_recall(
    engine: ScreeningEngine,
    customers: pd.DataFrame,
    entries: Sequence[WatchlistEntry],
) -> dict[str, Any]:
    """Share of planted hits that survive blocking, at entity level.

    Reported next to the reduction ratio because either number alone is
    meaningless: a blocker that returns nothing prunes 100% of the pair space.
    """
    from src.sanctions.normalize import normalize_name

    uid_by_position = [entries[engine.reference_owner(i)].uid for i in range(len(engine.reference_names))]
    index = engine.blocking_index
    retained = 0
    checked = 0
    for row in customers.itertuples():
        uid = row.watchlist_true_match_uid
        if pd.isna(uid):
            continue
        checked += 1
        normalized = normalize_name(str(row.full_name), engine.config)
        if any(uid_by_position[p] == uid for p in index.candidates(normalized)):
            retained += 1
    return {
        "planted_pairs": checked,
        "retained_after_blocking": retained,
        "blocking_recall": round(retained / checked, 6) if checked else 1.0,
    }


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
def _rule_performance(
    results: Mapping[str, ScenarioResult], labels: pd.DataFrame
) -> pd.DataFrame:
    truth = set(
        zip(
            labels.loc[labels["is_true_anomaly"], "customer_id"],
            labels.loc[labels["is_true_anomaly"], "period"],
        )
    )
    near = set(
        zip(
            labels.loc[labels["is_near_miss"], "customer_id"],
            labels.loc[labels["is_near_miss"], "period"],
        )
    )
    population = len(labels)
    rows: list[dict[str, Any]] = []
    for rule_id, result in sorted(results.items()):
        cells = set(zip(result.alerts["customer_id"], result.alerts["period"]))
        tps = len(cells & truth)
        rows.append(
            {
                "rule_id": rule_id,
                "alert_volume": len(cells),
                "alert_rate": round(len(cells) / population, 6) if population else 0.0,
                "true_positives": tps,
                "false_positives": len(cells) - tps,
                "precision": round(tps / len(cells), 6) if cells else 0.0,
                "recall": round(tps / len(truth), 6) if truth else 0.0,
                "false_positive_ratio": round((len(cells) - tps) / tps, 4) if tps else float("inf"),
                "near_misses_caught": len(cells & near),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def _period_end(period: str) -> str:
    return str(pd.Period(period, freq="M").end_time.strftime("%Y-%m-%d"))


def _write_audit(
    log: AuditLog,
    screening_alerts: pd.DataFrame,
    rule_alerts: pd.DataFrame,
    screening_version: str,
    tm_version: str,
    as_of: str,
) -> None:
    """One immutable record per alert, carrying its own scoring breakdown."""
    for row in screening_alerts.itertuples():
        log.append(
            event_type="SCREENING_ALERT",
            rule_version=screening_version,
            subject_id=str(row.subject_id),
            occurred_at=as_of,
            decision=str(row.decision),
            input_snapshot={
                "subject_name": str(row.subject_name),
                "watchlist_uid": str(row.watchlist_uid),
                "watchlist_name": str(row.watchlist_name),
                "program": str(row.program),
                "matched_on": str(row.matched_on),
            },
            scoring_breakdown={
                "name_score": float(row.name_score),
                "corroboration_delta": float(row.corroboration_delta),
                "final_score": float(row.final_score),
                "jaro_winkler": float(row.jaro_winkler),
                "token_set": float(row.token_set),
                "phonetic": float(row.phonetic),
                "initials": float(row.initials),
                "order_swapped": bool(row.order_swapped),
            },
        )
    for row in rule_alerts.itertuples():
        log.append(
            event_type="TM_ALERT",
            rule_version=tm_version,
            subject_id=str(row.customer_id),
            occurred_at=_period_end(str(row.period)),
            decision="REVIEW",
            input_snapshot={
                "rule_id": str(row.rule_id),
                "period": str(row.period),
                "txn_count": int(row.txn_count),
            },
            scoring_breakdown={
                "metric_value": float(row.metric_value),
                "alert_amount": float(row.alert_amount),
            },
        )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(
    config: AppConfig | None = None,
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
    dataset: GeneratedDataset | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    app_config = config or load_app_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    artefacts: list[Path] = []
    # Durations only. Excluded from the reproducibility diff by construction.
    run_timings: dict[str, Any] = {}

    run_id = derive_run_id(
        app_config.seed,
        [app_config.screening.rule_version, app_config.monitoring.rule_version],
    )
    logger.info("pipeline start", extra={"run_id": run_id, "seed": app_config.seed})

    # ---- 1. generate ------------------------------------------------------
    data = dataset or generate_dataset(app_config.generation)
    frames = {
        "watchlist": data.watchlist,
        "customers": data.customers,
        "transactions": data.transactions,
        "labels": data.labels,
    }
    written = write_parquet(frames, data_dir)
    artefacts.extend(written.values())
    artefacts.append(write_json(output_dir / "dataset_manifest.json", data.manifest))

    # ---- 2. load ----------------------------------------------------------
    connection = connect()
    try:
        load_frames(connection, frames, validate=False)
        window_start = app_config.generation.start_date.isoformat()
        window_end = (
            pd.Timestamp(app_config.generation.start_date)
            + pd.DateOffset(months=app_config.generation.months)
        ).date().isoformat()
        create_run_window(connection, window_start, window_end)

        rule_engine = RuleEngine(connection, app_config.monitoring)
        rule_engine.install_risk_countries(app_config.generation.corridors)
        rule_engine.install_params()

        # ---- 3. data quality ---------------------------------------------
        dq_results = run_checks(connection, app_config.dq_checks)
        run_timings["data_quality"] = dq_timings(dq_results)
        dq_frame = to_frame(dq_results)
        artefacts.append(write_csv(output_dir / "dq_report.csv", dq_frame))
        enforce(dq_results)

        # ---- 4. screening -------------------------------------------------
        entries = _watchlist_entries(data.watchlist)
        screener = ScreeningEngine(config=app_config.screening)
        screener.index_watchlist(entries)
        screening_run: ScreeningRun = screener.screen_many(
            _subjects(data.customers), top_n=3, min_score=SCREENING_FLOOR
        )
        screening_alerts = _screening_frame(screening_run.matches)
        artefacts.append(
            write_csv(
                output_dir / "screening_alerts.csv",
                screening_alerts,
                sort_by=["subject_id", "watchlist_uid"],
            )
        )

        run_timings["screening_seconds"] = round(screening_run.elapsed_seconds, 3)
        run_timings["blocking"] = screening_run.blocking.timings()
        blocking = dict(screening_run.blocking.as_dict())
        blocking.update(_blocking_recall(screener, data.customers, entries))
        blocking["prefiltered_pairs"] = screener.prefiltered_pairs
        blocking["alias_names"] = sum(len(e.aliases) for e in entries)
        blocking["fully_scored_pairs"] = screening_run.scored_pairs
        artefacts.append(write_json(output_dir / "blocking_benchmark.json", blocking))

        screening_sweep = _screening_sweep(screening_alerts, data.customers)
        artefacts.append(write_csv(output_dir / "screening_threshold_sweep.csv", screening_sweep))
        screening_metrics = _screening_metrics(
            screening_alerts, data.customers, app_config.screening.review
        )
        artefacts.append(write_json(output_dir / "screening_metrics.json", screening_metrics))

        # ---- 5. scenarios -------------------------------------------------
        scenario_results = rule_engine.run_all()
        run_timings["scenarios"] = {
            rule_id: round(result.elapsed_seconds, 4)
            for rule_id, result in sorted(scenario_results.items())
        }
        rule_alerts = combine_alerts(scenario_results)
        artefacts.append(
            write_csv(
                output_dir / "rule_alerts.csv", rule_alerts,
                sort_by=["customer_id", "period", "rule_id"],
            )
        )
        rule_performance = _rule_performance(scenario_results, data.labels)
        artefacts.append(write_csv(output_dir / "rule_performance.csv", rule_performance))

        # ---- 6. features, split, models -----------------------------------
        base_features = build_base_features(connection)
        train, test, feature_names, split, _ = prepare_matrices(
            base_features,
            data.labels,
            float(app_config.monitoring.split["test_fraction"]),
            app_config.seed,
        )
        models: TrainedModels = train_all(
            train, test, feature_names, app_config.monitoring, app_config.seed
        )

        # Secondary out-of-time slice: the grouped split answers "unseen
        # customers"; this answers "next quarter", which is where monitoring
        # models actually degrade first.
        oot_mask = out_of_time_mask(test, int(app_config.monitoring.split["out_of_time_months"]))
        best_name = models.best().model_name
        oot_labels = test.loc[oot_mask, "is_true_anomaly"].to_numpy(dtype=int)
        oot_positives = int(oot_labels.sum())

        # A trailing slice can be too thin to score. At 0.4% prevalence a few
        # months of a small portfolio may contain no positives at all, and an
        # ROC-AUC over one class is undefined — reporting the gap is correct,
        # crashing the pipeline over it is not.
        if MIN_OUT_OF_TIME_POSITIVES <= oot_positives < len(oot_labels):
            oot_metrics = evaluate(
                f"{best_name}_out_of_time",
                oot_labels,
                models.test_scores[best_name][oot_mask.to_numpy()],
                app_config.monitoring,
                enforce_tripwire=False,
            ).as_dict()
        else:
            oot_metrics = {
                "model": f"{best_name}_out_of_time",
                "n_test": int(len(oot_labels)),
                "n_positive": oot_positives,
                "roc_auc": None,
                "skipped_reason": (
                    f"only {oot_positives} positives in the out-of-time slice; "
                    f"at least {MIN_OUT_OF_TIME_POSITIVES} are needed for a stable estimate"
                ),
            }
            logger.warning("out-of-time evaluation skipped", extra=oot_metrics)

        model_metrics = {
            "split": dict(split.describe()),
            "feature_count": len(feature_names),
            "leakage_ceiling": app_config.monitoring.max_plausible_roc_auc,
            "unsupervised_ceiling": app_config.monitoring.max_unsupervised_roc_auc,
            "target_recall": app_config.monitoring.target_recall,
            "models": [e.as_dict() for e in models.evaluations],
            "out_of_time": oot_metrics,
        }
        artefacts.append(write_json(output_dir / "model_metrics.json", model_metrics))

        # Evidence for the non-separability claim, shipped as an artefact rather
        # than asserted in the README.
        separability = separability_report(test, models.test_scores, best_name)
        artefacts.append(
            write_json(output_dir / "separability_diagnostics.json", separability)
        )

        importance = pd.DataFrame(
            [
                {"model": e.model_name, "feature": name, "importance": round(value, 6)}
                for e in models.evaluations
                for name, value in e.top_features
            ]
        )
        artefacts.append(write_csv(output_dir / "feature_importance.csv", importance))

        # ---- 7. ATL/BTL ---------------------------------------------------
        sweep = run_atl_btl(rule_engine, app_config.monitoring, data.labels)
        artefacts.append(write_csv(output_dir / "atl_btl_sweep.csv", sweep))
        artefacts.append(
            write_csv(output_dir / "threshold_recommendations.csv", recommended_thresholds(sweep))
        )

        # ---- 8. audit -----------------------------------------------------
        audit = AuditLog(run_id)
        _write_audit(
            audit,
            screening_alerts[screening_alerts["final_score"] >= app_config.screening.review],
            rule_alerts,
            app_config.screening.rule_version,
            app_config.monitoring.rule_version,
            window_end,
        )
        audit.verify()
        artefacts.append(audit.write_jsonl(output_dir / "audit_log.jsonl"))
        artefacts.append(write_json(output_dir / "audit_summary.json", audit.summary()))
    finally:
        connection.close()

    # ---- 9. plots and manifest -------------------------------------------
    from src.pipeline.report import render_plots

    artefacts.extend(
        render_plots(
            output_dir=output_dir,
            model_metrics=model_metrics,
            sweep=sweep,
            rule_performance=rule_performance,
            screening_sweep=screening_sweep,
            scores=models.test_scores,
            y_test=test["is_true_anomaly"].to_numpy(dtype=int),
        )
    )

    elapsed = time.perf_counter() - started
    metrics = {
        "run_id": run_id,
        "seed": app_config.seed,
        "dataset": dict(data.manifest),
        "data_quality": {
            "checks": len(dq_results),
            "passed": int((dq_frame["status"] == "PASS").sum()),
            "failed": int((dq_frame["status"] != "PASS").sum()),
        },
        "screening": screening_metrics,
        "blocking": blocking,
        "rules": {
            "scenarios": len(scenario_results),
            "alert_rows": int(len(rule_alerts)),
            "union_recall": round(
                float(
                    len(
                        set(zip(rule_alerts["customer_id"], rule_alerts["period"]))
                        & set(
                            zip(
                                data.labels.loc[data.labels["is_true_anomaly"], "customer_id"],
                                data.labels.loc[data.labels["is_true_anomaly"], "period"],
                            )
                        )
                    )
                    / max(int(data.labels["is_true_anomaly"].sum()), 1)
                ),
                6,
            ),
        },
        "models": model_metrics,
        "separability": separability,
        "atl_btl": {
            "rules_swept": int(sweep["rule_id"].nunique()),
            "settings_evaluated": int(len(sweep)),
        },
        "audit": audit.summary(),
    }
    artefacts.append(write_json(output_dir / "pipeline_metrics.json", metrics))

    manifest = build_manifest(artefacts, output_dir.parent)
    artefacts.append(write_json(output_dir / "run_manifest.json", manifest))
    write_json(
        output_dir / "run_info.json",
        {
            "run_id": run_id,
            "elapsed_seconds": round(elapsed, 2),
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
    )
    write_json(output_dir / "run_timings.json", {**run_timings, "total_seconds": round(elapsed, 2)})
    logger.info(
        "pipeline complete",
        extra={"run_id": run_id, "elapsed_seconds": round(elapsed, 1),
               "artefacts": manifest["artefact_count"]},
    )
    return PipelineResult(
        output_dir=output_dir,
        metrics=metrics,
        manifest=manifest,
        artefacts=tuple(artefacts),
        elapsed_seconds=elapsed,
    )
