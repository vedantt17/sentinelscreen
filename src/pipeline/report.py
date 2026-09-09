"""Charts and the generated README.

Every number and every chart in the README is read from an artefact in
`outputs/`. Nothing is typed by hand. That is not tidiness for its own sake: a
hand-written metric is a metric that silently goes stale the first time a
threshold moves, and a README that disagrees with the artefacts is worse than
no README at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.schemas import erd_mermaid
from src.logging_setup import get_logger
from src.paths import OUTPUT_DIR, PROJECT_ROOT

logger = get_logger(__name__)

_PALETTE = {
    "primary": "#1f4e79",
    "secondary": "#c1443c",
    "tertiary": "#4c8f6b",
    "muted": "#8a8f98",
    "grid": "#dfe3e8",
}


def _style(axis: Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=11, loc="left", pad=10)
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(True, color=_PALETTE["grid"], linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(labelsize=8)


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    # metadata is pinned so the PNG bytes do not carry a creation timestamp,
    # which would break the byte-for-byte reproducibility check.
    figure.savefig(path, dpi=140, metadata={"Software": "SentinelScreen"})
    plt.close(figure)
    logger.info("plot written", extra={"artefact": path.name})
    return path


def render_plots(
    output_dir: Path,
    model_metrics: Mapping[str, Any],
    sweep: pd.DataFrame,
    rule_performance: pd.DataFrame,
    screening_sweep: pd.DataFrame,
    scores: Mapping[str, NDArray[Any]],
    y_test: NDArray[Any],
) -> list[Path]:
    plots_dir = output_dir / "plots"
    written: list[Path] = []

    # --- ROC and precision-recall ----------------------------------------
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = [_PALETTE["primary"], _PALETTE["secondary"], _PALETTE["tertiary"]]
    for colour, (name, score) in zip(colours, sorted(scores.items())):
        fpr, tpr, _ = roc_curve(y_test, score)
        auc = next(
            m["roc_auc"] for m in model_metrics["models"] if m["model"] == name
        )
        roc_axis.plot(fpr, tpr, color=colour, linewidth=1.6, label=f"{name} (AUC {auc:.3f})")
        precision, recall, _ = precision_recall_curve(y_test, score)
        pr_axis.plot(recall, precision, color=colour, linewidth=1.6, label=name)
    roc_axis.plot([0, 1], [0, 1], color=_PALETTE["muted"], linestyle="--", linewidth=0.9)
    _style(roc_axis, "ROC — held-out set, natural class balance", "False positive rate", "Recall")
    roc_axis.legend(fontsize=8, frameon=False)

    prevalence = float(y_test.mean())
    pr_axis.axhline(prevalence, color=_PALETTE["muted"], linestyle="--", linewidth=0.9)
    pr_axis.annotate(
        f"prevalence {prevalence:.3%}",
        xy=(0.55, prevalence),
        fontsize=8,
        color=_PALETTE["muted"],
        va="bottom",
    )
    _style(pr_axis, "Precision-recall — the curve that matters at 0.4% prevalence",
           "Recall", "Precision")
    pr_axis.set_yscale("log")
    pr_axis.legend(fontsize=8, frameon=False)
    written.append(_save(figure, plots_dir / "model_performance.png"))

    # --- ATL/BTL ----------------------------------------------------------
    rules = sorted(sweep["rule_id"].unique())
    columns = 4
    rows = int(np.ceil(len(rules) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.1 * rows))
    flat = np.atleast_1d(axes).ravel()
    for axis, rule_id in zip(flat, rules):
        group = sweep[sweep["rule_id"] == rule_id].sort_values("param_value")
        axis.plot(group["param_value"], group["alert_volume"], color=_PALETTE["primary"],
                  marker="o", markersize=3, linewidth=1.4, label="alert volume")
        twin = axis.twinx()
        twin.plot(group["param_value"], group["recall"], color=_PALETTE["secondary"],
                  marker="s", markersize=3, linewidth=1.4, label="recall")
        twin.set_ylim(0, max(0.05, float(group["recall"].max()) * 1.25))
        twin.tick_params(labelsize=7)
        baseline = group[group["position"] == "BASELINE"]
        if not baseline.empty:
            axis.axvline(float(baseline["param_value"].iloc[0]), color=_PALETTE["muted"],
                         linestyle="--", linewidth=0.9)
        _style(axis, f"{rule_id}\n{group['param_name'].iloc[0]}", "threshold", "alerts")
    for axis in flat[len(rules):]:
        axis.axis("off")
    figure.suptitle(
        "ATL/BTL threshold sweep — alert volume (blue) against recall (red); "
        "dashed line is the production baseline",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    written.append(_save(figure, plots_dir / "atl_btl_sweep.png"))

    # --- rule performance -------------------------------------------------
    figure, (volume_axis, precision_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    ordered = rule_performance.sort_values("alert_volume", ascending=True)
    labels = [r.replace("_", " ").title() for r in ordered["rule_id"]]
    volume_axis.barh(labels, ordered["alert_volume"], color=_PALETTE["primary"], height=0.6)
    _style(volume_axis, "Alert volume by scenario", "Alerted customer-months", "")
    precision_axis.barh(labels, ordered["precision"], color=_PALETTE["tertiary"], height=0.6)
    _style(precision_axis, "Precision by scenario", "Precision", "")
    written.append(_save(figure, plots_dir / "rule_performance.png"))

    # --- screening threshold ---------------------------------------------
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.plot(screening_sweep["threshold"], screening_sweep["recall"],
              color=_PALETTE["primary"], marker="o", markersize=3, label="recall")
    axis.plot(screening_sweep["threshold"], screening_sweep["near_miss_alert_rate"],
              color=_PALETTE["secondary"], marker="s", markersize=3,
              label="share of planted near misses alerted")
    axis.plot(screening_sweep["threshold"], screening_sweep["alert_rate"],
              color=_PALETTE["muted"], marker="^", markersize=3, label="portfolio alert rate")
    _style(axis, "Sanctions screening: recall against false-positive load",
           "Match-score threshold", "Rate")
    axis.legend(fontsize=8, frameon=False)
    written.append(_save(figure, plots_dir / "screening_threshold.png"))
    return written


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #
def _load(output_dir: Path, name: str) -> Any:
    path = output_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"missing artefact {path}; run `make run` first")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def _fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def _fmt_pct(value: Any, places: int = 2) -> str:
    return f"{float(value) * 100:.{places}f}%"


def _model_row(entry: Mapping[str, Any]) -> str:
    target = entry["at_target_recall"]
    return (
        f"| {entry['model']} | {entry['roc_auc']:.4f} | {entry['average_precision']:.4f} "
        f"| {target['recall']:.3f} | {target['precision']:.4f} "
        f"| {_fmt_int(target['alert_volume'])} | {target['false_positive_ratio']:.1f} : 1 |"
    )


def render_readme(output_dir: Path = OUTPUT_DIR, project_root: Path = PROJECT_ROOT) -> Path:
    """Assemble README.md from the artefacts produced by the last pipeline run."""
    metrics = _load(output_dir, "pipeline_metrics.json")
    manifest = _load(output_dir, "run_manifest.json")
    blocking = _load(output_dir, "blocking_benchmark.json")
    screening = _load(output_dir, "screening_metrics.json")
    model_metrics = _load(output_dir, "model_metrics.json")
    dataset = metrics["dataset"]
    dq = _load(output_dir, "dq_report.csv")
    rules = _load(output_dir, "rule_performance.csv")
    sweep = _load(output_dir, "atl_btl_sweep.csv")
    recommendations = _load(output_dir, "threshold_recommendations.csv")
    separability = _load(output_dir, "separability_diagnostics.json")

    best = max(model_metrics["models"], key=lambda m: m["average_precision"])
    unsupervised = next(
        m for m in model_metrics["models"] if m["model"] == "isolation_forest"
    )
    template = (project_root / "docs" / "README_TEMPLATE.md").read_text(encoding="utf-8")

    baseline_sweep = sweep[sweep["position"] == "BASELINE"]
    sample_rule = sweep[sweep["rule_id"] == "R01_STRUCTURING"].sort_values("param_value")
    sweep_table = "\n".join(
        f"| {row.param_value:g} | {row.position} | {row.precision:.4f} | {row.recall:.4f} "
        f"| {_fmt_int(row.alert_volume)} | "
        + (f"{row.false_positive_ratio:.1f} : 1" if np.isfinite(row.false_positive_ratio) else "n/a")
        + " |"
        for row in sample_rule.itertuples()
    )
    rule_table = "\n".join(
        f"| {row.rule_id} | {_fmt_int(row.alert_volume)} | {row.precision:.4f} "
        f"| {row.recall:.4f} | "
        + (f"{row.false_positive_ratio:.1f} : 1" if np.isfinite(row.false_positive_ratio) else "n/a")
        + " |"
        for row in rules.itertuples()
    )
    model_table = "\n".join(_model_row(m) for m in model_metrics["models"])
    recommendation_table = "\n".join(
        f"| {row.rule_id} | {row.param_name} | {row.baseline_value:g} | {row.recommended_value:g} "
        f"| {_fmt_int(row.baseline_alerts)} | {_fmt_int(row.recommended_alerts)} "
        f"| {row.baseline_recall:.4f} | {row.recommended_recall:.4f} |"
        for row in recommendations.itertuples()
    )
    dq_table = "\n".join(
        f"| {row.check_id} | {row.dimension} | {row.severity} | {row.status} "
        f"| {_fmt_int(row.failing_rows)} |"
        for row in dq.itertuples()
    )

    substitutions = {
        "{{ERD}}": erd_mermaid(),
        "{{SEED}}": str(metrics["seed"]),
        "{{RUN_ID}}": str(metrics["run_id"]),
        "{{ARTEFACT_COUNT}}": _fmt_int(manifest["artefact_count"]),
        "{{N_WATCHLIST}}": _fmt_int(dataset["watchlist_entities"]),
        "{{N_CUSTOMERS}}": _fmt_int(dataset["customers"]),
        "{{N_TRANSACTIONS}}": _fmt_int(dataset["transactions"]),
        "{{N_CELLS}}": _fmt_int(dataset["scored_customer_months"]),
        "{{ANOMALY_CELLS}}": _fmt_int(dataset["true_anomaly_cells"]),
        "{{ANOMALY_RATE}}": _fmt_pct(dataset["true_anomaly_rate"], 3),
        "{{NEAR_MISS_CELLS}}": _fmt_int(dataset["near_miss_cells"]),
        "{{NEAR_MISS_RATE}}": _fmt_pct(dataset["near_miss_rate"], 2),
        "{{INVISIBLE_ANOMALIES}}": _fmt_int(dataset["invisible_anomalies"]),
        "{{FULL_PAIR_SPACE}}": _fmt_int(blocking["full_pair_space"]),
        "{{REFERENCE_NAMES}}": _fmt_int(blocking["n_reference"]),
        "{{ALIAS_NAMES}}": _fmt_int(blocking["alias_names"]),
        "{{CANDIDATE_PAIRS}}": _fmt_int(blocking["candidate_pairs"]),
        "{{REDUCTION_PCT}}": f"{blocking['reduction_pct']:.3f}%",
        "{{SCORED_PAIRS}}": _fmt_int(blocking["fully_scored_pairs"]),
        "{{BLOCKING_RECALL}}": f"{blocking['blocking_recall']:.4f}",
        "{{MEAN_CANDIDATES}}": f"{blocking['mean_candidates_per_query']:.1f}",
        "{{LARGEST_BLOCK}}": _fmt_int(blocking["largest_block"]),
        "{{SCREEN_THRESHOLD}}": f"{screening['threshold']:.2f}",
        "{{SCREEN_RECALL}}": f"{screening['recall']:.4f}",
        "{{SCREEN_PRECISION}}": f"{screening['precision']:.4f}",
        "{{SCREEN_FP_RATIO}}": f"{screening['false_positive_ratio']:.1f}",
        "{{SCREEN_ALERTED}}": _fmt_int(screening["alerted_subjects"]),
        "{{SCREEN_ALERT_RATE}}": _fmt_pct(screening["alert_rate"]),
        "{{SCREEN_PLANTED}}": _fmt_int(screening["planted_true_matches"]),
        "{{SCREEN_DETECTED}}": _fmt_int(screening["detected_true_matches"]),
        "{{NEAR_MISS_ALERTED}}": _fmt_int(screening["near_misses_alerted"]),
        "{{NEAR_MISS_PLANTED}}": _fmt_int(screening["planted_near_misses"]),
        "{{BEST_MODEL}}": str(best["model"]),
        "{{BEST_AUC}}": f"{best['roc_auc']:.4f}",
        "{{BEST_AP}}": f"{best['average_precision']:.4f}",
        "{{BEST_RECALL}}": f"{best['at_target_recall']['recall']:.3f}",
        "{{BEST_PRECISION}}": f"{best['at_target_recall']['precision']:.4f}",
        "{{BEST_FP_RATIO}}": f"{best['at_target_recall']['false_positive_ratio']:.1f}",
        "{{BEST_ALERTS}}": _fmt_int(best["at_target_recall"]["alert_volume"]),
        "{{LEAKAGE_CEILING}}": f"{model_metrics['leakage_ceiling']:.2f}",
        "{{UNSUP_CEILING}}": f"{model_metrics['unsupervised_ceiling']:.2f}",
        "{{UNSUP_AUC}}": f"{unsupervised['roc_auc']:.4f}",
        "{{NEAR_MISS_AUC}}": f"{separability['anomaly_vs_near_miss_roc_auc']:.4f}",
        "{{NEAR_MISS_VERDICT}}": str(separability["near_miss_separability_verdict"]),
        "{{INJECTED_AUC}}": f"{separability['injected_vs_ordinary_roc_auc']:.4f}",
        "{{INVISIBLE_PCTL}}": f"{separability['mean_percentile']['invisible_positives']:.3f}",
        "{{NEAR_MISS_PCTL}}": f"{separability['mean_percentile']['near_miss_cells']:.3f}",
        "{{FEATURE_COUNT}}": str(model_metrics["feature_count"]),
        "{{TRAIN_ROWS}}": _fmt_int(model_metrics["split"]["train_rows"]),
        "{{TEST_ROWS}}": _fmt_int(model_metrics["split"]["test_rows"]),
        "{{TRAIN_CUSTOMERS}}": _fmt_int(model_metrics["split"]["train_customers"]),
        "{{TEST_CUSTOMERS}}": _fmt_int(model_metrics["split"]["test_customers"]),
        "{{OOT_AUC}}": (
            f"{model_metrics['out_of_time']['roc_auc']:.4f}"
            if model_metrics["out_of_time"].get("roc_auc") is not None
            else "not computed (too few positives in the slice)"
        ),
        "{{OOT_N}}": _fmt_int(model_metrics["out_of_time"]["n_test"]),
        "{{RULE_UNION_RECALL}}": f"{metrics['rules']['union_recall']:.4f}",
        "{{RULE_ALERT_ROWS}}": _fmt_int(metrics["rules"]["alert_rows"]),
        "{{BASELINE_ALERTS}}": _fmt_int(int(baseline_sweep["alert_volume"].sum())),
        "{{SWEEP_SETTINGS}}": _fmt_int(metrics["atl_btl"]["settings_evaluated"]),
        "{{DQ_CHECKS}}": str(metrics["data_quality"]["checks"]),
        "{{DQ_PASSED}}": str(metrics["data_quality"]["passed"]),
        "{{AUDIT_RECORDS}}": _fmt_int(metrics["audit"]["records"]),
        "{{AUDIT_HEAD}}": str(metrics["audit"]["head_hash"])[:16],
        "{{RULE_TABLE}}": rule_table,
        "{{MODEL_TABLE}}": model_table,
        "{{SWEEP_TABLE}}": sweep_table,
        "{{RECOMMENDATION_TABLE}}": recommendation_table,
        "{{DQ_TABLE}}": dq_table,
    }

    readme = template
    for placeholder, value in substitutions.items():
        readme = readme.replace(placeholder, value)

    leftovers = [
        token
        for token in readme.split("{{")[1:]
        if "}}" in token
    ]
    if leftovers:
        raise ValueError(
            "README template has unsubstituted placeholders: "
            + ", ".join(sorted({t.split('}}')[0] for t in leftovers}))
        )

    path = project_root / "README.md"
    path.write_text(readme, encoding="utf-8", newline="\n")
    logger.info("README generated", extra={"path": str(path), "bytes": len(readme)})
    return path
