"""Customer-month feature construction and the train/test split.

The ordering enforced here is the whole point of the module:

    1. build label-free base features from the transaction feed
    2. split by customer  (nothing target-derived has been computed yet)
    3. fit peer and self baselines on the TRAINING customers only
    4. transform both sides with those fitted baselines
    5. only now attach labels

Fitting a peer baseline over the full dataset would let a test-period customer's
own behaviour influence the median it is later compared against. It is a subtle
leak, it inflates ROC-AUC by a few points, and it is the single most common way
a monitoring model looks better in a notebook than it does in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.exceptions import ModelTrainingError
from src.logging_setup import get_logger
from src.paths import SQL_DIR

logger = get_logger(__name__)

KEY_COLUMNS: tuple[str, str] = ("customer_id", "period")
SEGMENT_COLUMN: str = "behaviour_segment"

_SEGMENTS: tuple[str, ...] = (
    "DORMANT_RETAIL", "SALARIED", "SMALL_BUSINESS", "CASH_INTENSIVE", "TRADE_CORRIDOR",
)

# Columns the peer baseline is computed for. Ratios are excluded: a ratio is
# already scale-free, and dividing it by a peer median produces a quantity that
# is unstable when the median is near zero.
_PEER_LEVEL_COLUMNS: tuple[str, ...] = (
    "total_amount", "txn_count", "max_amount", "avg_amount", "cash_amount",
    "high_risk_amount", "distinct_counterparties",
)

# Rate features that are shrunk toward the portfolio mean in proportion to how
# little evidence the cell carries. A customer-month with one cash transaction
# has a cash_share of exactly 1.0, which is indistinguishable from a genuinely
# cash-dominated business month with forty transactions — and there are tens of
# thousands of the former. Left raw, these columns turn every quiet month into a
# feature-space outlier and the model spends its capacity on noise.
_SHRINK_COLUMNS: tuple[str, ...] = (
    "cash_share", "cash_txn_share", "wire_share", "card_share", "credit_share",
    "near_ctr_share", "round_share", "cross_border_share", "high_risk_share",
    "medium_risk_share", "top_counterparty_share", "night_share", "weekend_share",
    "passthrough_ratio", "counterparty_diversity",
)

# Pseudo-count for the shrinkage. Three is roughly the point at which a monthly
# rate starts to mean anything, and it matches the transaction floor most
# monitoring programmes use before scoring a cell at all.
_SHRINK_PRIOR: float = 3.0


def build_base_features(
    connection: duckdb.DuckDBPyConnection, sql_dir: Path = SQL_DIR
) -> pd.DataFrame:
    sql = (sql_dir / "features.sql").read_text(encoding="utf-8")
    frame = connection.execute(sql).fetchdf()
    if frame.empty:
        raise ModelTrainingError("feature query returned no rows")
    numeric = [c for c in frame.columns if c not in KEY_COLUMNS + (SEGMENT_COLUMN,)]
    frame[numeric] = frame[numeric].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    logger.info(
        "base features built",
        extra={"rows": len(frame), "columns": len(frame.columns)},
    )
    return frame


@dataclass(frozen=True)
class Split:
    train_keys: pd.DataFrame
    test_keys: pd.DataFrame
    train_customers: frozenset[str]
    test_customers: frozenset[str]

    def describe(self) -> dict[str, object]:
        return {
            "strategy": "grouped_by_customer",
            "train_rows": len(self.train_keys),
            "test_rows": len(self.test_keys),
            "train_customers": len(self.train_customers),
            "test_customers": len(self.test_customers),
        }


def split_by_customer(features: pd.DataFrame, test_fraction: float, seed: int) -> Split:
    """Group split on customer_id.

    A random row split would put January and February of the same customer on
    opposite sides, and since a customer's behaviour is highly autocorrelated
    the model would effectively be scoring accounts it had already memorised.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    groups = features["customer_id"].to_numpy()
    train_idx, test_idx = next(splitter.split(features, groups=groups))
    train = features.iloc[np.sort(train_idx)][list(KEY_COLUMNS)].reset_index(drop=True)
    test = features.iloc[np.sort(test_idx)][list(KEY_COLUMNS)].reset_index(drop=True)
    return Split(
        train_keys=train,
        test_keys=test,
        train_customers=frozenset(train["customer_id"]),
        test_customers=frozenset(test["customer_id"]),
    )


@dataclass
class BaselineTransformer:
    """Peer- and self-relative features, fitted on training customers only."""

    peer_medians: dict[str, dict[str, float]] = field(default_factory=dict)
    global_medians: dict[str, float] = field(default_factory=dict)
    shrink_targets: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, train_features: pd.DataFrame) -> "BaselineTransformer":
        grouped = train_features.groupby(SEGMENT_COLUMN)
        for column in _PEER_LEVEL_COLUMNS:
            medians = grouped[column].median()
            self.peer_medians[column] = {
                str(segment): float(value) for segment, value in medians.items()
            }
            self.global_medians[column] = float(train_features[column].median())

        # Shrinkage targets are the training-set means, weighted by transaction
        # count so the target reflects the rate of the portfolio's *activity*
        # rather than of its quietest accounts.
        weights = train_features["txn_count"].to_numpy(dtype=float)
        total_weight = float(weights.sum()) or 1.0
        for column in _SHRINK_COLUMNS:
            if column in train_features.columns:
                values = train_features[column].to_numpy(dtype=float)
                self.shrink_targets[column] = float((values * weights).sum() / total_weight)
        self.fitted = True
        logger.info(
            "peer baselines fitted",
            extra={
                "segments": sorted(self.peer_medians[_PEER_LEVEL_COLUMNS[0]]),
                "columns": list(_PEER_LEVEL_COLUMNS),
            },
        )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ModelTrainingError("BaselineTransformer.transform called before fit")
        out = features.copy()

        for column in _PEER_LEVEL_COLUMNS:
            lookup = self.peer_medians[column]
            fallback = self.global_medians[column]
            # An unseen segment falls back to the global training median rather
            # than to NaN: a missing peer baseline must not silently become an
            # extreme ratio that the model reads as an anomaly.
            denominator = out[SEGMENT_COLUMN].map(lookup).fillna(fallback).astype(float)
            denominator = denominator.replace(0.0, np.nan)
            out[f"peer_ratio_{column}"] = (out[column] / denominator).fillna(1.0).clip(0.0, 50.0)

        # Empirical-Bayes shrinkage on the rate features. A cell with n
        # transactions gets weight n/(n+k) on its own observed rate and the rest
        # on the fitted portfolio rate, so a one-transaction month no longer
        # reports a cash share of 1.00 as though it were established behaviour.
        counts = out["txn_count"].to_numpy(dtype=float)
        confidence = counts / (counts + _SHRINK_PRIOR)
        for column, target in self.shrink_targets.items():
            if column in out.columns:
                observed = out[column].to_numpy(dtype=float)
                out[column] = confidence * observed + (1.0 - confidence) * target

        # Self-relative: this month against the customer's own trailing history.
        # `shift(1)` inside the customer group keeps it strictly backward-looking.
        out = out.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
        grouped = out.groupby("customer_id", sort=False)
        prior_amount = grouped["total_amount"].shift(1)
        prior_count = grouped["txn_count"].shift(1)
        expanding_mean = (
            grouped["total_amount"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(
                level=0, drop=True
            )
        )
        out["prior_month_amount"] = prior_amount.fillna(0.0)
        out["amount_vs_prior"] = (
            out["total_amount"] / prior_amount.replace(0.0, np.nan)
        ).fillna(1.0).clip(0.0, 50.0)
        out["count_vs_prior"] = (
            out["txn_count"] / prior_count.replace(0.0, np.nan)
        ).fillna(1.0).clip(0.0, 50.0)
        out["amount_vs_trailing_mean"] = (
            out["total_amount"] / expanding_mean.replace(0.0, np.nan)
        ).fillna(1.0).clip(0.0, 50.0)
        out["is_first_active_month"] = grouped.cumcount().eq(0).astype(float)

        for segment in _SEGMENTS:
            out[f"segment_{segment.lower()}"] = (out[SEGMENT_COLUMN] == segment).astype(float)
        return out.drop(columns=[SEGMENT_COLUMN])


def feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(c for c in frame.columns if c not in KEY_COLUMNS and c != SEGMENT_COLUMN)


def attach_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Left-join the target. The only place the label table is ever touched."""
    merged = features.merge(
        labels[["customer_id", "period", "is_true_anomaly", "is_near_miss", "typology", "intensity"]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    merged["is_true_anomaly"] = merged["is_true_anomaly"].fillna(False).astype(bool)
    merged["is_near_miss"] = merged["is_near_miss"].fillna(False).astype(bool)
    return merged


def prepare_matrices(
    base_features: pd.DataFrame,
    labels: pd.DataFrame,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], Split, BaselineTransformer]:
    """Run the full ordered pipeline and return train/test frames.

    Returned frames carry the keys, the engineered features, and the label. The
    caller selects `feature_names` when fitting, so the label is never in X by
    construction rather than by convention.
    """
    split = split_by_customer(base_features, test_fraction, seed)
    train_mask = base_features["customer_id"].isin(split.train_customers)

    transformer = BaselineTransformer().fit(base_features.loc[train_mask])
    engineered = transformer.transform(base_features)

    names = feature_columns(engineered)
    labelled = attach_labels(engineered, labels)

    train = labelled.loc[labelled["customer_id"].isin(split.train_customers)].reset_index(drop=True)
    test = labelled.loc[labelled["customer_id"].isin(split.test_customers)].reset_index(drop=True)
    logger.info(
        "matrices prepared",
        extra={
            **split.describe(),
            "features": len(names),
            "train_positives": int(train["is_true_anomaly"].sum()),
            "test_positives": int(test["is_true_anomaly"].sum()),
            "train_prevalence": round(float(train["is_true_anomaly"].mean()), 6),
            "test_prevalence": round(float(test["is_true_anomaly"].mean()), 6),
        },
    )
    return train, test, names, split, transformer


def out_of_time_mask(frame: pd.DataFrame, months: int) -> pd.Series:
    """Trailing-months mask for the secondary out-of-time evaluation.

    The grouped split answers "does this generalise to unseen customers"; the
    out-of-time slice answers "does it still work next quarter". Real monitoring
    models degrade on the second question long before the first.
    """
    periods: Sequence[str] = sorted(frame["period"].unique())
    if months >= len(periods):
        raise ModelTrainingError(
            f"out_of_time_months={months} leaves no in-time data (only {len(periods)} periods)"
        )
    holdout = set(periods[-months:])
    return frame["period"].isin(holdout)


def feature_summary(names: Sequence[str]) -> Mapping[str, int]:
    return {"feature_count": len(names)}
