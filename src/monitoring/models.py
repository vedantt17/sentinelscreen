"""Supervised and unsupervised models for the alert-triage layer.

Two things here are non-negotiable and both are enforced in code rather than
described in a docstring:

* **Held-out evaluation only, at the natural class balance.** No resampling of
  the test set. A precision figure computed after balancing the test set is not
  a precision figure — it is a statement about a population that does not exist.
* **Two leakage tripwires.** A supervised ceiling on held-out ROC-AUC, and a
  tighter one on the *unsupervised* Isolation Forest. The second is the sharper
  test and the one that caught a real bug here: a model that never sees a label
  can only separate the classes if the positives are outliers in raw feature
  space, which on synthetic data means the generator wrote the answer into the
  features. Training raises rather than returning such a model, because shipping
  it would mean publishing a number that cannot be reproduced on real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import MonitoringConfig
from src.exceptions import ModelTrainingError
from src.logging_setup import get_logger

logger = get_logger(__name__)

TARGET_COLUMN: str = "is_true_anomaly"


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    alert_volume: int
    alert_rate: float
    true_positives: int
    false_positives: int

    @property
    def false_positive_ratio(self) -> float:
        """Reviews per confirmed hit — the number an operations lead budgets against."""
        return self.false_positives / self.true_positives if self.true_positives else float("inf")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": round(self.threshold, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "alert_volume": self.alert_volume,
            "alert_rate": round(self.alert_rate, 6),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_positive_ratio": round(self.false_positive_ratio, 4),
        }


@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    roc_auc: float
    average_precision: float
    n_test: int
    n_positive: int
    prevalence: float
    at_target_recall: OperatingPoint
    at_alert_budget: OperatingPoint
    top_features: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "roc_auc": round(self.roc_auc, 6),
            "average_precision": round(self.average_precision, 6),
            "n_test": self.n_test,
            "n_positive": self.n_positive,
            "prevalence": round(self.prevalence, 6),
            "at_target_recall": self.at_target_recall.as_dict(),
            "at_alert_budget_5pct": self.at_alert_budget.as_dict(),
            "top_features": [
                {"feature": name, "importance": round(value, 6)} for name, value in self.top_features
            ],
        }


def operating_point_at_recall(
    y_true: NDArray[Any], scores: NDArray[Any], target_recall: float
) -> OperatingPoint:
    """Lowest-volume threshold that still reaches `target_recall`.

    Thresholds are taken from the observed score distribution rather than a
    fixed grid, so the reported operating point is one the model can actually
    produce instead of an interpolation between two it cannot.
    """
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = y_true[order]
    sorted_scores = scores[order]
    positives = int(y_true.sum())
    if positives == 0:
        raise ModelTrainingError("held-out set contains no positives; cannot form an operating point")

    cumulative_tp = np.cumsum(sorted_labels)
    recall_curve = cumulative_tp / positives
    reached = np.flatnonzero(recall_curve >= target_recall)
    index = int(reached[0]) if reached.size else len(sorted_labels) - 1

    volume = index + 1
    true_positives = int(cumulative_tp[index])
    return OperatingPoint(
        threshold=float(sorted_scores[index]),
        precision=true_positives / volume,
        recall=true_positives / positives,
        alert_volume=volume,
        alert_rate=volume / len(y_true),
        true_positives=true_positives,
        false_positives=volume - true_positives,
    )


def operating_point_at_budget(
    y_true: NDArray[Any], scores: NDArray[Any], budget_fraction: float
) -> OperatingPoint:
    """What the model delivers when the queue size is fixed by headcount."""
    volume = max(1, int(round(budget_fraction * len(y_true))))
    order = np.argsort(-scores, kind="mergesort")[:volume]
    true_positives = int(y_true[order].sum())
    positives = max(int(y_true.sum()), 1)
    return OperatingPoint(
        threshold=float(scores[order][-1]),
        precision=true_positives / volume,
        recall=true_positives / positives,
        alert_volume=volume,
        alert_rate=volume / len(y_true),
        true_positives=true_positives,
        false_positives=volume - true_positives,
    )


def _importances(model: Any, names: Sequence[str], limit: int = 15) -> tuple[tuple[str, float], ...]:
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return ()
    total = float(np.sum(raw)) or 1.0
    pairs = sorted(zip(names, (float(v) / total for v in raw)), key=lambda kv: (-kv[1], kv[0]))
    return tuple(pairs[:limit])


def _check_leakage(name: str, roc_auc: float, config: MonitoringConfig) -> None:
    if roc_auc > config.max_plausible_roc_auc:
        raise ModelTrainingError(
            f"{name} scored ROC-AUC {roc_auc:.4f} on held-out data, above the plausibility "
            f"ceiling of {config.max_plausible_roc_auc:.2f}. On a 0.4%-prevalence AML problem "
            "this indicates target leakage in a feature, not a good model. Fix the "
            "feature/label separation rather than shipping the number."
        )
    if roc_auc < config.min_plausible_roc_auc:
        logger.warning(
            "model is weaker than expected; check feature construction",
            extra={"model": name, "roc_auc": round(roc_auc, 4)},
        )


def _check_unsupervised_leakage(roc_auc: float, config: MonitoringConfig) -> None:
    """The sharper of the two tripwires.

    An unsupervised detector has no access to the target, so it can only
    separate the classes if the positives are outliers in raw feature space.
    On a synthetic dataset that means the injection is additive or otherwise
    distributionally distinct — the generator has written the answer into the
    features. This check is what caught exactly that during development.
    """
    if roc_auc > config.max_unsupervised_roc_auc:
        raise ModelTrainingError(
            f"unsupervised Isolation Forest scored ROC-AUC {roc_auc:.4f}, above the "
            f"{config.max_unsupervised_roc_auc:.2f} ceiling. A label-blind model cannot "
            "separate genuine AML anomalies this well; the positives are raw-space "
            "outliers, which means the data generation is leaking, not the model."
        )


def evaluate(
    model_name: str,
    y_true: NDArray[Any],
    scores: NDArray[Any],
    config: MonitoringConfig,
    top_features: tuple[tuple[str, float], ...] = (),
    enforce_tripwire: bool = True,
) -> ModelEvaluation:
    roc_auc = float(roc_auc_score(y_true, scores))
    if enforce_tripwire:
        _check_leakage(model_name, roc_auc, config)
    evaluation = ModelEvaluation(
        model_name=model_name,
        roc_auc=roc_auc,
        average_precision=float(average_precision_score(y_true, scores)),
        n_test=int(len(y_true)),
        n_positive=int(y_true.sum()),
        prevalence=float(y_true.mean()),
        at_target_recall=operating_point_at_recall(y_true, scores, config.target_recall),
        at_alert_budget=operating_point_at_budget(y_true, scores, 0.05),
        top_features=top_features,
    )
    logger.info("model evaluated", extra=evaluation.as_dict())
    return evaluation


@dataclass(frozen=True)
class TrainedModels:
    lightgbm: LGBMClassifier
    xgboost: XGBClassifier
    isolation_forest: IsolationForest
    evaluations: tuple[ModelEvaluation, ...]
    test_scores: Mapping[str, NDArray[Any]]

    def best(self) -> ModelEvaluation:
        return max(self.evaluations, key=lambda e: e.average_precision)


def train_all(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: Sequence[str],
    config: MonitoringConfig,
    seed: int,
    enforce_tripwire: bool = True,
) -> TrainedModels:
    x_train = train[list(feature_names)].to_numpy(dtype=np.float64)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
    x_test = test[list(feature_names)].to_numpy(dtype=np.float64)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=int)

    if y_train.sum() == 0 or y_test.sum() == 0:
        raise ModelTrainingError("split produced a fold with no positives")

    # Class weight is derived from the TRAINING fold only. Computing it over the
    # full dataset would let the test-set prevalence influence training.
    positives = int(y_train.sum())
    scale_pos_weight = float((len(y_train) - positives) / positives)

    lgbm_params = dict(config.lightgbm)
    lgbm = LGBMClassifier(
        random_state=seed, scale_pos_weight=scale_pos_weight, **lgbm_params
    )
    lgbm.fit(x_train, y_train)
    lgbm_scores = lgbm.predict_proba(x_test)[:, 1]

    xgb_params = dict(config.xgboost)
    xgb = XGBClassifier(
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        **xgb_params,
    )
    xgb.fit(x_train, y_train)
    xgb_scores = xgb.predict_proba(x_test)[:, 1]

    # Unsupervised, and fitted on TRAINING FEATURES ONLY WITHOUT LABELS — the
    # point of carrying it is to catch typologies the labelled history has never
    # seen, so training it on the labelled positives would defeat its purpose.
    iso_params = dict(config.isolation_forest)
    iso = IsolationForest(random_state=seed, **iso_params)
    iso.fit(x_train)
    iso_scores = -iso.score_samples(x_test)

    evaluations = (
        evaluate(
            "lightgbm", y_test, lgbm_scores, config,
            _importances(lgbm, feature_names), enforce_tripwire,
        ),
        evaluate(
            "xgboost", y_test, xgb_scores, config,
            _importances(xgb, feature_names), enforce_tripwire,
        ),
        # The isolation forest gets its own, tighter ceiling rather than the
        # supervised one, because a high score here means something different
        # and worse: it never saw a label, so separation implies the positives
        # are raw-space outliers and the generator is doing the model's work.
        evaluate("isolation_forest", y_test, iso_scores, config, (), enforce_tripwire=False),
    )
    if enforce_tripwire:
        _check_unsupervised_leakage(evaluations[2].roc_auc, config)
    return TrainedModels(
        lightgbm=lgbm,
        xgboost=xgb,
        isolation_forest=iso,
        evaluations=evaluations,
        test_scores={
            "lightgbm": lgbm_scores,
            "xgboost": xgb_scores,
            "isolation_forest": iso_scores,
        },
    )
