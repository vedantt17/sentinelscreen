"""Separability diagnostics.

The README claims the generated data is genuinely hard. This module produces the
evidence for that claim so it is auditable rather than asserted, and it
decomposes the model's ROC-AUC into the three things that actually drive it:

* **injected vs. ordinary** — how distinguishable a cell carrying *any* injected
  pattern is from ordinary traffic. This is legitimate signal; a typology should
  look different from a quiet salary account.
* **anomaly vs. near miss** — how distinguishable a true typology is from a
  legitimate look-alike. This should sit at roughly 0.50. Anything materially
  above it means the two populations were not drawn from the same renderer, and
  every downstream precision figure is then an artefact of the generator.
* **where the invisible positives rank** — positives with no rendered typology
  should sit around the middle of the distribution. If they ranked high,
  something other than the injected pattern would be carrying the label.

The second number is the one to read first. It is the difference between a
dataset that measures a monitoring system and one that measures itself.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.logging_setup import get_logger

logger = get_logger(__name__)


def _safe_auc(y_true: NDArray[Any], scores: NDArray[Any]) -> float | None:
    """AUC, or None when a class is absent rather than a misleading 0.5."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    return round(float(roc_auc_score(y_true, scores)), 6)


def _mean_percentile(scores: NDArray[Any], subset: NDArray[Any]) -> float | None:
    if not subset.any():
        return None
    ranks = pd.Series(scores).rank(pct=True, method="average").to_numpy()
    return round(float(ranks[subset].mean()), 6)


def separability_report(
    test: pd.DataFrame, scores: Mapping[str, NDArray[Any]], model_name: str
) -> dict[str, Any]:
    score = np.asarray(scores[model_name], dtype=float)
    is_anomaly = test["is_true_anomaly"].to_numpy(dtype=bool)
    is_near_miss = test["is_near_miss"].to_numpy(dtype=bool)
    intensity = test["intensity"].fillna("NONE").to_numpy()

    injected = is_anomaly | is_near_miss
    visible = is_anomaly & (intensity != "INVISIBLE")
    invisible = is_anomaly & (intensity == "INVISIBLE")

    # Restricted to injected cells: can the model tell a real typology from a
    # legitimate look-alike, given that it has already spotted both?
    anomaly_vs_near_miss = _safe_auc(is_anomaly[injected], score[injected])

    # Visible positives against clean negatives, excluding the planted
    # look-alikes, which is the signal the model is genuinely entitled to.
    clean = ~injected
    visible_mask = visible | clean
    visible_vs_clean = _safe_auc(visible[visible_mask], score[visible_mask])

    report: dict[str, Any] = {
        "model": model_name,
        "n_test": int(len(test)),
        "n_anomaly": int(is_anomaly.sum()),
        "n_near_miss": int(is_near_miss.sum()),
        "n_invisible_positives": int(invisible.sum()),
        "overall_roc_auc": _safe_auc(is_anomaly, score),
        "injected_vs_ordinary_roc_auc": _safe_auc(injected, score),
        "anomaly_vs_near_miss_roc_auc": anomaly_vs_near_miss,
        "visible_positive_vs_clean_roc_auc": visible_vs_clean,
        "mean_percentile": {
            "visible_positives": _mean_percentile(score, visible),
            "invisible_positives": _mean_percentile(score, invisible),
            "near_miss_cells": _mean_percentile(score, is_near_miss),
            "ordinary_cells": _mean_percentile(score, clean),
        },
    }

    # A near-miss population the model can pick apart from the true anomalies
    # means the shared renderer has sprung a leak somewhere.
    if anomaly_vs_near_miss is not None:
        report["near_miss_separability_verdict"] = (
            "indistinguishable" if abs(anomaly_vs_near_miss - 0.5) <= 0.08
            else "SEPARABLE — investigate the injection path"
        )
    logger.info("separability diagnostics computed", extra=dict(report))
    return report
