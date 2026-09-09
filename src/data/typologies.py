"""Typology and near-miss injection.

This module exists because of a measured failure. The first version of the
generator produced a dataset on which LightGBM scored 0.996 ROC-AUC and — far
more damning — an Isolation Forest with no access to the labels scored 0.979.
That is not a good model. It is a generator that wrote the answer into the
features: anomalous cells had simply been handed extra transactions, so "is
anomalous" collapsed into "is unusually busy".

Four properties fix it, and every one of them is a property the real problem
has:

1. **Injection is size-preserving.** A rendered typology replaces a share of the
   customer-month's existing transactions rather than adding to it. A month of
   laundering has roughly the number of transactions the account always had;
   what changes is their *shape* — all cash, all just under ten thousand, all to
   one corridor. Volume was the single largest leak, because a cell that held
   two transactions and was handed six could be detected by counting.

2. **True typologies and legitimate look-alikes come from ONE renderer.** A cash
   business banking five deposits of 9,200 dollars and a structurer banking five
   deposits of 9,200 dollars produce identical rows. The difference is intent,
   and intent is not in the transaction feed. The populations differ only in the
   mix of patterns and in an intensity multiplier whose distributions overlap.

3. **Intensity dilutes rather than amplifies.** It sets what *fraction* of the
   cell takes on the typology shape, so low-intensity cells shade continuously
   into ordinary traffic instead of forming a separable cluster.

4. **Some positives are rendered as nothing at all.** Laundering that never
   moves a monthly aggregate is invisible to a monthly model. Those cells are
   labelled positive and look like any other month, which is what puts a real
   ceiling on achievable recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.config import GenerationConfig
from src.logging_setup import get_logger

logger = get_logger(__name__)

PATTERNS: tuple[str, ...] = (
    "structuring", "rapid_movement", "high_risk_corridor",
    "sanctioned_counterparty", "dormant_reactivation", "layering",
)

# Segment weights for the two populations. They are close on purpose:
# laundering runs through cash businesses and trade corridors precisely because
# that is where it is hardest to see, so drawing anomalies from a visibly
# different segment mix would hand the model a shortcut it would not have.
_ANOMALY_SEGMENT_WEIGHT: Mapping[str, float] = {
    "DORMANT_RETAIL": 0.55, "SALARIED": 0.75, "SMALL_BUSINESS": 1.30,
    "CASH_INTENSIVE": 1.75, "TRADE_CORRIDOR": 1.65,
}
_NEAR_MISS_SEGMENT_WEIGHT: Mapping[str, float] = {
    "DORMANT_RETAIL": 0.45, "SALARIED": 0.70, "SMALL_BUSINESS": 1.35,
    "CASH_INTENSIVE": 1.90, "TRADE_CORRIDOR": 1.75,
}


@dataclass(frozen=True)
class InjectionCell:
    customer_id: str
    month: int
    pattern: str
    is_anomaly: bool
    intensity: float
    existing_count: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.customer_id, self.month)

    @property
    def rendered_rows(self) -> int:
        """How many of the cell's transactions take the typology's shape.

        Capped at the cell's own size: injection displaces traffic, it never
        adds to it. A quiet month can therefore only host a quiet typology,
        which is precisely why quiet months are hard.
        """
        if self.intensity <= 0.0:
            return 0
        wanted = int(round(self.existing_count * min(self.intensity, 1.0)))
        return int(np.clip(wanted, 1, self.existing_count))

    @property
    def intensity_band(self) -> str:
        if self.intensity <= 0.0:
            return "INVISIBLE"
        return "ATTENUATED" if self.rendered_rows <= 2 else "LOUD"


def _weighted_sample(
    rng: np.random.Generator,
    cells: pd.DataFrame,
    weights: Mapping[str, float],
    size: int,
) -> NDArray[Any]:
    probabilities = cells["behaviour_segment"].map(weights).to_numpy(dtype=float)
    probabilities = probabilities / probabilities.sum()
    return cast(
        "NDArray[Any]",
        rng.choice(len(cells), size=min(size, len(cells)), replace=False, p=probabilities),
    )


def _draw_pattern(rng: np.random.Generator, mix: Mapping[str, float]) -> str:
    names = [p for p in PATTERNS if p in mix and mix[p] > 0]
    probabilities = np.array([float(mix[p]) for p in names])
    probabilities = probabilities / probabilities.sum()
    return names[int(rng.choice(len(names), p=probabilities))]


def plan_injections(
    rng: np.random.Generator, config: GenerationConfig, active_cells: pd.DataFrame
) -> list[InjectionCell]:
    """Select and parameterise both populations in one pass.

    Both are drawn from the same pool of active customer-months and the draw is
    joint, so a cell can never be claimed by both.
    """
    cfg = config.typologies
    intensity_cfg = cfg["intensity"]
    n_cells = len(active_cells)
    n_anomaly = max(1, int(round(float(cfg["anomaly_rate"]) * n_cells)))
    n_near_miss = int(round(float(cfg["near_miss_rate"]) * n_cells))

    anomaly_mix = cfg["anomaly_mix"]
    near_miss_mix = cfg["near_miss_mix"]
    invisible_share = float(intensity_cfg["invisible_share"])
    log_sigma = float(intensity_cfg["log_sigma"])
    anomaly_centre = float(intensity_cfg["anomaly_log_mean"])
    near_miss_centre = float(intensity_cfg["near_miss_log_mean"])

    window = cfg["window_months"]
    min_window, max_window = int(window["min"]), int(window["max"])
    months = config.months

    customer_ids = active_cells["customer_id"].to_numpy()
    cell_months = active_cells["month"].to_numpy()
    cell_counts = active_cells["txn_count"].to_numpy()
    size_lookup = {
        (str(c), int(m)): int(n) for c, m, n in zip(customer_ids, cell_months, cell_counts)
    }
    taken: dict[tuple[str, int], InjectionCell] = {}

    def claim(
        position: int,
        is_anomaly: bool,
        mix: Mapping[str, float],
        centre: float,
        invisible: bool = False,
    ) -> None:
        customer_id = str(customer_ids[position])
        pattern = _draw_pattern(rng, mix)
        intensity = 0.0 if invisible else float(np.exp(rng.normal(centre, log_sigma)))
        # Campaign length is drawn identically for both populations. If only
        # true anomalies spanned consecutive months, the self-relative features
        # (this month versus last) would encode the label directly.
        span = int(rng.integers(min_window, max_window + 1))
        start = int(cell_months[position])
        for offset in range(span):
            month = start + offset
            if month >= months:
                break
            key = (customer_id, month)
            if key in taken:
                continue
            existing = size_lookup.get(key)
            if existing is None:
                # The campaign ran past the account's last active month; there
                # is nothing to displace, so there is nothing to render.
                continue
            taken[key] = InjectionCell(
                customer_id=customer_id,
                month=month,
                pattern=pattern,
                is_anomaly=is_anomaly,
                intensity=intensity,
                existing_count=existing,
            )

    n_invisible = int(round(invisible_share * n_anomaly))
    n_visible = n_anomaly - n_invisible

    for position in _weighted_sample(rng, active_cells, _ANOMALY_SEGMENT_WEIGHT, n_visible):
        if sum(1 for c in taken.values() if c.is_anomaly) >= n_visible:
            break
        claim(int(position), True, anomaly_mix, anomaly_centre)

    # Same segment weights as the visible cohort. Drawing this cohort from a
    # different mix would make segment membership predict the label.
    for position in _weighted_sample(
        rng, active_cells, _ANOMALY_SEGMENT_WEIGHT, max(n_invisible * 3, 1)
    ):
        if sum(1 for c in taken.values() if c.is_anomaly) >= n_anomaly:
            break
        if (str(customer_ids[position]), int(cell_months[position])) in taken:
            continue
        claim(int(position), True, anomaly_mix, anomaly_centre, invisible=True)

    for position in _weighted_sample(
        rng, active_cells, _NEAR_MISS_SEGMENT_WEIGHT, max(n_near_miss * 2, 1)
    ):
        # Count realised cells, not claims: a claim spans one or two months, so
        # counting claims would overshoot the configured near-miss rate.
        if sum(1 for c in taken.values() if not c.is_anomaly) >= n_near_miss:
            break
        if (str(customer_ids[position]), int(cell_months[position])) in taken:
            continue
        claim(int(position), False, near_miss_mix, near_miss_centre)

    cells = sorted(taken.values(), key=lambda c: (c.customer_id, c.month))
    logger.info(
        "injections planned",
        extra={
            "anomaly_cells": sum(1 for c in cells if c.is_anomaly),
            "near_miss_cells": sum(1 for c in cells if not c.is_anomaly),
            "invisible_anomalies": sum(1 for c in cells if c.is_anomaly and c.intensity == 0.0),
            "mean_rendered_rows": round(
                float(np.mean([c.rendered_rows for c in cells])) if cells else 0.0, 2
            ),
        },
    )
    return cells


class PatternRenderer:
    """Turns an `InjectionCell` into replacement transaction rows.

    One renderer serves both populations. Any branch on `cell.is_anomaly` here
    would be a hole in the non-separability guarantee, so there is exactly one —
    the counterparty pool — and it touches a field the monitoring feature set
    never reads. It matters only to the screening engine.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        config: GenerationConfig,
        window_start: pd.Timestamp,
        pools: Mapping[str, tuple[str, ...]],
    ) -> None:
        self.rng = rng
        self.config = config
        self.window_start = window_start
        self.pools = pools
        self.high_risk = tuple(config.corridors["high_risk"])
        self.medium_risk = tuple(config.corridors["medium_risk"])
        self.band = config.amounts["structuring_band"]
        self.multiples = list(config.amounts["round_dollar_multiples"])

    def _pick(self, pool_name: str) -> str:
        pool = self.pools[pool_name]
        return str(pool[int(self.rng.integers(0, len(pool)))])

    def _country(self, options: Sequence[str]) -> str:
        return str(options[int(self.rng.integers(0, len(options)))])

    def render(self, cell: InjectionCell, customer: pd.Series) -> list[dict[str, object]]:
        budget = cell.rendered_rows
        if budget <= 0:
            return []

        month_start = self.window_start + pd.DateOffset(months=cell.month)
        days = int(month_start.days_in_month)
        residence = str(customer["residence_country"])
        scale = min(cell.intensity, 1.6)
        # An injected row must not predate the account it sits in. Cells are
        # drawn from months the account was already active, but the opening
        # month itself can start part-way through.
        opened_at = pd.Timestamp(customer["account_open_date"])
        rows: list[dict[str, object]] = []

        def emit(
            amount: float,
            channel: str,
            direction: str,
            counterparty: str,
            country: str,
            day: int | None = None,
            hour: int | None = None,
        ) -> None:
            timestamp = month_start + pd.Timedelta(
                days=int(self.rng.integers(0, days)) if day is None else int(min(day, days - 1)),
                hours=int(self.rng.integers(8, 20)) if hour is None else int(hour),
                minutes=int(self.rng.integers(0, 60)),
                seconds=int(self.rng.integers(0, 60)),
            )
            if timestamp < opened_at:
                timestamp = opened_at + pd.Timedelta(
                    hours=int(self.rng.integers(1, 72)),
                    minutes=int(self.rng.integers(0, 60)),
                )
            rows.append(
                {
                    "customer_id": cell.customer_id,
                    "txn_ts": timestamp,
                    "amount": round(float(max(amount, 1.0)), 2),
                    "currency": "USD",
                    "channel": channel,
                    "direction": direction,
                    "counterparty_name": counterparty,
                    "counterparty_country": country,
                    "counterparty_bank": self._pick("banks"),
                    "is_cross_border": country != residence,
                    "_origin": "TYPOLOGY" if cell.is_anomaly else "NEAR_MISS",
                }
            )

        # The one substantive difference between the populations, and it is
        # invisible to the monitoring model: a true anomaly transacts with a
        # designated party, a near miss with someone whose name merely collides.
        counterparty_pool = "true_hit" if cell.is_anomaly else "confusable"

        if cell.pattern == "structuring":
            start_day = int(self.rng.integers(0, max(1, days - 10)))
            for _ in range(budget):
                emit(
                    float(self.rng.uniform(self.band[0], self.band[1])),
                    "CASH", "CREDIT", self._pick("benign"), residence,
                    day=start_day + int(self.rng.integers(0, 10)),
                    hour=int(self.rng.integers(9, 18)),
                )

        elif cell.pattern == "rapid_movement":
            emitted = 0
            while emitted < budget:
                amount = float(self.rng.uniform(6_000, 45_000) * scale)
                day = int(self.rng.integers(0, max(1, days - 3)))
                emit(amount, "WIRE", "CREDIT", self._pick("benign"), residence, day=day)
                emitted += 1
                if emitted >= budget:
                    break
                emit(
                    amount * float(self.rng.uniform(0.84, 0.98)),
                    "WIRE", "DEBIT", self._pick("benign"),
                    self._country(self.medium_risk + self.high_risk),
                    day=day + int(self.rng.integers(0, 3)),
                )
                emitted += 1

        elif cell.pattern == "high_risk_corridor":
            for _ in range(budget):
                emit(
                    float(self.rng.uniform(4_000, 40_000) * scale),
                    "WIRE", "DEBIT", self._pick("benign"),
                    self._country(self.high_risk),
                )

        elif cell.pattern == "sanctioned_counterparty":
            for _ in range(budget):
                emit(
                    float(self.rng.uniform(9_000, 80_000) * scale),
                    "WIRE",
                    "DEBIT" if self.rng.random() < 0.7 else "CREDIT",
                    self._pick(counterparty_pool),
                    self._country(self.high_risk + self.medium_risk),
                )

        elif cell.pattern == "dormant_reactivation":
            for _ in range(budget):
                emit(
                    float(self.rng.uniform(3_000, 30_000) * scale),
                    "WIRE" if self.rng.random() < 0.55 else "ACH",
                    "CREDIT" if self.rng.random() < 0.5 else "DEBIT",
                    self._pick("benign"),
                    self._country(self.medium_risk + self.high_risk)
                    if self.rng.random() < 0.35
                    else residence,
                )

        else:  # layering
            for _ in range(budget):
                multiple = float(self.multiples[int(self.rng.integers(0, len(self.multiples)))])
                emit(
                    multiple * float(self.rng.integers(1, 9)),
                    "WIRE" if self.rng.random() < 0.55 else "ACH",
                    "DEBIT" if self.rng.random() < 0.65 else "CREDIT",
                    self._pick("benign"),
                    self._country(self.medium_risk) if self.rng.random() < 0.35 else residence,
                )
        return rows
