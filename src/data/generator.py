"""Deterministic synthetic data generation.

Three properties matter more than volume here.

1. **Reproducibility.** One `numpy.random.Generator` seeded from config is
   threaded through every draw and consumed in a fixed order. No module-level
   `random`, no wall-clock, no dict iteration order dependence.

2. **Non-separability.** The typologies are injected into the same behavioural
   space that legitimate cash-intensive and trade-corridor customers already
   occupy, and a third of them are attenuated to the point of being invisible.
   A generator that plants loud anomalies produces a 0.99-AUC model that proves
   only that the generator wrote the answer into the features.

3. **Screening difficulty.** Planted watchlist hits carry the noise that breaks
   naive matchers — a different romanization, a swapped name order, an added
   honorific, a missing date of birth — while the planted near misses share a
   surname or a date of birth and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Final, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.config import GenerationConfig
from src.data import name_corpus as nc
from src.data.name_corpus import NameForm
from src.data.typologies import InjectionCell, PatternRenderer, plan_injections
from src.logging_setup import get_logger

logger = get_logger(__name__)

_SCRIPT_POOLS: Final[Mapping[str, tuple[tuple[NameForm, ...], tuple[NameForm, ...]]]] = {
    "LATIN": (nc.LATIN_GIVEN, nc.LATIN_FAMILY),
    "ARABIC": (nc.ARABIC_GIVEN, nc.ARABIC_FAMILY),
    "CYRILLIC": (nc.CYRILLIC_GIVEN, nc.CYRILLIC_FAMILY),
    "HAN": (nc.HAN_GIVEN, nc.HAN_SURNAME),
}

# Channel mix by behavioural segment. CASH_INTENSIVE and TRADE_CORRIDOR are the
# confusable cohorts: their legitimate baseline already looks like the shape of
# a cash-structuring or corridor typology, which is the point.
_CHANNEL_MIX: Final[Mapping[str, tuple[float, float, float, float]]] = {
    # (WIRE, ACH, CARD, CASH)
    "DORMANT_RETAIL": (0.05, 0.20, 0.65, 0.10),
    "SALARIED": (0.06, 0.34, 0.54, 0.06),
    "SMALL_BUSINESS": (0.16, 0.38, 0.28, 0.18),
    "CASH_INTENSIVE": (0.10, 0.20, 0.16, 0.54),
    "TRADE_CORRIDOR": (0.52, 0.22, 0.14, 0.12),
}

_CHANNEL_AMOUNT_SCALE: Final[Mapping[str, float]] = {
    "WIRE": 4.2, "ACH": 1.3, "CARD": 0.16, "CASH": 1.0,
}

# Probability that a given transaction settles outside the customer's country,
# and the risk tier it lands in when it does.
_CORRIDOR_PROFILE: Final[Mapping[str, tuple[float, float, float]]] = {
    # (p_cross_border, p_medium_given_cross, p_high_given_cross)
    "DORMANT_RETAIL": (0.04, 0.20, 0.03),
    "SALARIED": (0.06, 0.22, 0.03),
    "SMALL_BUSINESS": (0.14, 0.30, 0.05),
    "CASH_INTENSIVE": (0.10, 0.34, 0.07),
    "TRADE_CORRIDOR": (0.62, 0.46, 0.14),
}

_MONTH_SPREAD: Final[Mapping[str, tuple[int, ...]]] = {
    "DORMANT_RETAIL": (1, 2, 3),
    "SALARIED": (12,),
    "SMALL_BUSINESS": (9, 12),
    "CASH_INTENSIVE": (12,),
    "TRADE_CORRIDOR": (6, 9, 12),
}

# Config speaks of "chinese" because that is how a compliance stakeholder
# describes the romanization problem; the code speaks of HAN because that is the
# Unicode script the transliterator actually dispatches on.
_SCRIPT_ALIASES: Final[Mapping[str, str]] = {
    "LATIN": "LATIN", "ARABIC": "ARABIC", "CYRILLIC": "CYRILLIC",
    "CHINESE": "HAN", "HAN": "HAN",
}


def _script_key(name: str) -> str:
    try:
        return _SCRIPT_ALIASES[name.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported script {name!r} in generation config") from exc


@dataclass(frozen=True)
class GeneratedDataset:
    watchlist: pd.DataFrame
    customers: pd.DataFrame
    transactions: pd.DataFrame
    labels: pd.DataFrame
    manifest: dict[str, object]


class SyntheticDataGenerator:
    """Builds the watchlist, customer master, transaction feed and label table."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(np.random.PCG64(config.seed))
        self.window_start = datetime.combine(config.start_date, datetime.min.time())
        self.window_end = (
            pd.Timestamp(self.window_start) + pd.DateOffset(months=config.months)
        ).to_pydatetime()
        self.window_seconds = int((self.window_end - self.window_start).total_seconds())
        self.low_risk = tuple(config.corridors["low_risk"])
        self.medium_risk = tuple(config.corridors["medium_risk"])
        self.high_risk = tuple(config.corridors["high_risk"])

    # ------------------------------------------------------------ utilities
    def _choice(self, options: Sequence[str] | tuple[str, ...], size: int) -> NDArray[Any]:
        idx = self.rng.integers(0, len(options), size=size)
        return cast("NDArray[Any]", np.asarray(options, dtype=object)[idx])

    def _weighted(self, options: Sequence[str], weights: Sequence[float], size: int) -> NDArray[Any]:
        probabilities = np.asarray(weights, dtype=float)
        probabilities = probabilities / probabilities.sum()
        idx = self.rng.choice(len(options), size=size, p=probabilities)
        return cast("NDArray[Any]", np.asarray(options, dtype=object)[idx])

    def _render_person(
        self, script: str, swap_order: bool = False, honorific: bool = False
    ) -> tuple[str, NameForm, NameForm, int, int]:
        given_pool, family_pool = _SCRIPT_POOLS[script]
        given = given_pool[int(self.rng.integers(0, len(given_pool)))]
        family = family_pool[int(self.rng.integers(0, len(family_pool)))]
        given_variant = int(self.rng.integers(0, len(given.variants)))
        family_variant = int(self.rng.integers(0, len(family.variants)))
        middle, middle_variant = self._draw_middle(script, 0.30)
        name = self._assemble(
            given, family, given_variant, family_variant, script, swap_order, honorific,
            middle=middle, middle_variant=middle_variant,
        )
        return name, given, family, given_variant, family_variant

    def _draw_middle(self, script: str, probability: float) -> tuple[NameForm | None, int]:
        """Second given name, drawn for every script except Han.

        Han personal names are a one- or two-character given name with no
        middle-name slot, so inventing one would be a data-realism error that
        the transliterator would then faithfully reproduce.
        """
        if script == "HAN" or self.rng.random() >= probability:
            return None, 0
        pool = _SCRIPT_POOLS[script][0]
        form = pool[int(self.rng.integers(0, len(pool)))]
        return form, int(self.rng.integers(0, len(form.variants)))

    def _assemble(
        self,
        given: NameForm,
        family: NameForm,
        given_variant: int,
        family_variant: int,
        script: str,
        swap_order: bool,
        honorific: bool,
        native: bool = False,
        middle: NameForm | None = None,
        middle_variant: int = 0,
    ) -> str:
        if native and script != "LATIN":
            # Han personal names are written surname-first with no separator.
            if script == "HAN":
                return f"{family.native}{given.native}"
            parts = [given.native]
            if middle is not None:
                parts.append(middle.native)
            parts.append(family.native)
            return " ".join(parts)
        given_text = given.variants[given_variant % len(given.variants)]
        family_text = family.variants[family_variant % len(family.variants)]
        # A middle or second given name present on one side and absent on the
        # other is among the most common real mismatches between a KYC record
        # and a watchlist entry. It is also what keeps the (given, family) pair
        # space large enough not to manufacture collisions.
        personal = [given_text]
        if middle is not None:
            personal.append(middle.variants[middle_variant % len(middle.variants)])
        # Han romanization keeps surname first; everything else is given-first
        # unless the record is one of the deliberately order-swapped ones.
        surname_first = (script == "HAN") != swap_order
        core = (
            " ".join([family_text, *personal])
            if surname_first
            else " ".join([*personal, family_text])
        )
        if honorific:
            core = f"{nc.HONORIFICS[int(self.rng.integers(0, len(nc.HONORIFICS)))]} {core}"
        return core

    def _render_org(self) -> str:
        prefix = nc.ORG_PREFIX[int(self.rng.integers(0, len(nc.ORG_PREFIX)))]
        core = nc.ORG_CORE[int(self.rng.integers(0, len(nc.ORG_CORE)))]
        suffix = nc.ORG_SUFFIX[int(self.rng.integers(0, len(nc.ORG_SUFFIX)))]
        return f"{prefix} {core} {suffix}"

    def _random_dob(self, min_age: int = 22, max_age: int = 78) -> str:
        year = int(self.window_start.year - self.rng.integers(min_age, max_age))
        month = int(self.rng.integers(1, 13))
        day = int(self.rng.integers(1, 29))
        return f"{year:04d}-{month:02d}-{day:02d}"

    # ------------------------------------------------------------ watchlist
    def generate_watchlist(self) -> pd.DataFrame:
        cfg = self.config.watchlist
        n = self.config.n_watchlist
        mix = cfg["script_mix"]
        scripts = self._weighted(
            [_script_key(s) for s in mix.keys()], list(mix.values()), n
        )
        programs = tuple(cfg["programs"])
        alias_max = int(cfg["alias_count_max"])

        rows: list[dict[str, object]] = []
        for i in range(n):
            script = str(scripts[i])
            is_entity = bool(self.rng.random() < 0.18)
            swap = bool(self.rng.random() < float(cfg["name_order_swap_rate"]))
            honorific = bool(self.rng.random() < float(cfg["honorific_rate"]) and not is_entity)

            aliases: list[str] = []
            if is_entity:
                primary = self._render_org()
                if self.rng.random() < 0.5:
                    aliases.append(primary.rsplit(" ", 1)[0])
                given = family = middle = None
                given_variant = family_variant = middle_variant = 0
            else:
                given_pool, family_pool = _SCRIPT_POOLS[script]
                given = given_pool[int(self.rng.integers(0, len(given_pool)))]
                family = family_pool[int(self.rng.integers(0, len(family_pool)))]
                given_variant = int(self.rng.integers(0, len(given.variants)))
                family_variant = int(self.rng.integers(0, len(family.variants)))
                # A third of non-Latin entries are published in native script,
                # mirroring how OFAC carries the original-script form alongside
                # a romanization rather than instead of it.
                native_primary = script != "LATIN" and self.rng.random() < 0.32
                middle, middle_variant = self._draw_middle(script, 0.22)
                primary = self._assemble(
                    given, family, given_variant, family_variant,
                    script, swap, honorific, native=native_primary,
                    middle=middle, middle_variant=middle_variant,
                )
                n_alias = int(self.rng.integers(0, alias_max + 1))
                for _ in range(n_alias):
                    alt_given = int(self.rng.integers(0, len(given.variants)))
                    alt_family = int(self.rng.integers(0, len(family.variants)))
                    alias = self._assemble(
                        given, family, alt_given, alt_family, script,
                        swap_order=bool(self.rng.random() < 0.35),
                        honorific=False,
                        native=bool(
                            script != "LATIN" and not native_primary and self.rng.random() < 0.25
                        ),
                        # An a.k.a. routinely drops the middle name the primary
                        # entry carries; that asymmetry is the point of aliases.
                        middle=middle if self.rng.random() < 0.4 else None,
                        middle_variant=middle_variant,
                    )
                    if alias != primary and alias not in aliases:
                        aliases.append(alias)

            has_dob = not is_entity and self.rng.random() >= float(cfg["missing_dob_rate"])
            if has_dob:
                dob_full = self._random_dob()
                # Year-only dates are extremely common on the real list and are
                # the reason DOB corroboration has to handle partial precision.
                dob: str | None = dob_full[:4] if self.rng.random() < 0.30 else dob_full
            else:
                dob = None

            nationality = (
                None
                if self.rng.random() < float(cfg["missing_nationality_rate"])
                else str(self._choice(self.high_risk + self.medium_risk, 1)[0])
            )
            national_id = (
                f"ID{int(self.rng.integers(10_000_000, 99_999_999))}"
                if self.rng.random() < 0.42
                else None
            )
            partial = self.rng.random() < float(cfg["partial_address_rate"])
            address_country = (
                None if partial and self.rng.random() < 0.35
                else str(self._choice(self.high_risk + self.medium_risk, 1)[0])
            )
            address_line = (
                None if partial
                else f"{int(self.rng.integers(1, 400))} {nc.ORG_CORE[int(self.rng.integers(0, len(nc.ORG_CORE)))]} Street"
            )
            listed = self.window_start - timedelta(days=int(self.rng.integers(30, 4200)))

            rows.append(
                {
                    "sdn_uid": f"SDN-{i + 100000:06d}",
                    "primary_name": primary,
                    "name_script": script,
                    "entity_type": "ENTITY" if is_entity else "INDIVIDUAL",
                    "program": str(programs[int(self.rng.integers(0, len(programs)))]),
                    "aliases": "|".join(aliases),
                    "dob": dob,
                    "nationality": nationality,
                    "national_id": national_id,
                    "address_country": address_country,
                    "address_line": address_line,
                    "listed_date": listed,
                    "remarks": f"Designated under {programs[int(self.rng.integers(0, len(programs)))]}.",
                    # Retained for planting hits; dropped before the frame is returned.
                    "_given_idx": -1 if given is None else _SCRIPT_POOLS[script][0].index(given),
                    "_family_idx": -1 if family is None else _SCRIPT_POOLS[script][1].index(family),
                    "_given_variant": given_variant,
                    "_family_variant": family_variant,
                    "_middle_idx": (
                        -1
                        if given is None or middle is None
                        else _SCRIPT_POOLS[script][0].index(middle)
                    ),
                }
            )

        frame = pd.DataFrame(rows)
        frame["listed_date"] = pd.to_datetime(frame["listed_date"])
        logger.info(
            "watchlist generated",
            extra={
                "entities": len(frame),
                "individuals": int((frame["entity_type"] == "INDIVIDUAL").sum()),
                "with_aliases": int((frame["aliases"] != "").sum()),
                "missing_dob_pct": round(float(frame["dob"].isna().mean()) * 100, 2),
            },
        )
        return frame

    # ------------------------------------------------------------ customers
    def generate_customers(self, watchlist: pd.DataFrame) -> pd.DataFrame:
        n = self.config.n_customers
        segments = [seg.segment for seg in self.config.activity_mix]
        weights = [seg.weight for seg in self.config.activity_mix]
        segment_array = self._weighted(segments, weights, n)
        script_mix = self.config.watchlist["script_mix"]
        scripts = self._weighted(
            [_script_key(s) for s in script_mix.keys()], list(script_mix.values()), n
        )

        names: list[str] = []
        for i in range(n):
            script = str(scripts[i])
            name, _, _, _, _ = self._render_person(script)
            names.append(name)

        residence = self._weighted(
            list(self.low_risk + self.medium_risk + self.high_risk),
            [6.0] * len(self.low_risk) + [2.0] * len(self.medium_risk) + [0.7] * len(self.high_risk),
            n,
        )
        nationality = residence.copy()
        # A minority hold a nationality different from where they bank, which is
        # what makes nationality a genuinely noisy corroborator.
        swap_mask = self.rng.random(n) < 0.18
        nationality[swap_mask] = self._choice(
            self.low_risk + self.medium_risk + self.high_risk, int(swap_mask.sum())
        )

        open_offsets = self.rng.integers(-2600, 300, size=n)
        account_open = np.array(
            [self.window_start + timedelta(days=int(d)) for d in open_offsets], dtype="datetime64[ns]"
        )

        risk = np.where(
            np.isin(nationality, list(self.high_risk)), "HIGH",
            np.where(np.isin(nationality, list(self.medium_risk)), "MEDIUM", "LOW"),
        )
        # Segment nudges the rating: cash-intensive and trade businesses are
        # rated up regardless of geography under most institutions' methodology.
        risk = np.where(
            (np.isin(segment_array, ["CASH_INTENSIVE", "TRADE_CORRIDOR"])) & (risk == "LOW"),
            "MEDIUM", risk,
        )

        frame = pd.DataFrame(
            {
                "customer_id": [f"CUST-{i + 1:06d}" for i in range(n)],
                "full_name": names,
                "name_script": scripts,
                "dob": [self._random_dob(18, 85) for _ in range(n)],
                "nationality": nationality,
                "residence_country": residence,
                "national_id": [f"ID{int(v)}" for v in self.rng.integers(10_000_000, 99_999_999, n)],
                "occupation": self._choice(nc.OCCUPATIONS, n),
                "behaviour_segment": segment_array,
                "kyc_risk_rating": risk,
                "account_open_date": account_open,
                "is_pep": self.rng.random(n) < 0.012,
                "watchlist_true_match_uid": pd.Series([None] * n, dtype="object"),
                "is_screening_near_miss": np.zeros(n, dtype=bool),
            }
        )
        frame = self._plant_screening_ground_truth(frame, watchlist)
        logger.info(
            "customers generated",
            extra={
                "customers": len(frame),
                "true_watchlist_matches": int(frame["watchlist_true_match_uid"].notna().sum()),
                "planted_near_misses": int(frame["is_screening_near_miss"].sum()),
            },
        )
        return frame

    def _plant_screening_ground_truth(
        self, customers: pd.DataFrame, watchlist: pd.DataFrame
    ) -> pd.DataFrame:
        truth_cfg = self.config.screening_ground_truth
        n_true = int(truth_cfg["true_match_customers"])
        n_near = int(truth_cfg["near_miss_customers"])

        individuals = watchlist[watchlist["entity_type"] == "INDIVIDUAL"]
        if individuals.empty:
            return customers

        chosen_rows = self.rng.choice(len(customers), size=n_true + n_near, replace=False)
        true_rows = chosen_rows[:n_true]
        near_rows = chosen_rows[n_true:]
        wl_positions = self.rng.integers(0, len(individuals), size=n_true + n_near)

        uid_col = customers.columns.get_loc("watchlist_true_match_uid")
        name_col = customers.columns.get_loc("full_name")

        for offset, customer_row in enumerate(true_rows):
            entry = individuals.iloc[int(wl_positions[offset])]
            script = str(entry["name_script"])
            given_pool, family_pool = _SCRIPT_POOLS[script]
            given = given_pool[int(entry["_given_idx"])]
            family = family_pool[int(entry["_family_idx"])]
            # The customer record is the SAME person under different feed
            # conventions: another accepted romanization, possibly reordered,
            # possibly with an honorific the bank captured from a passport.
            middle_idx = int(entry["_middle_idx"])
            # Half the time the bank captured the middle name and half the time
            # it did not. That asymmetry is what a production matcher has to
            # tolerate without either missing the hit or over-scoring it.
            middle = (
                given_pool[middle_idx] if middle_idx >= 0 and self.rng.random() < 0.5 else None
            )
            variant_name = self._assemble(
                given, family,
                given_variant=int(self.rng.integers(0, len(given.variants))),
                family_variant=int(self.rng.integers(0, len(family.variants))),
                script=script,
                swap_order=bool(self.rng.random() < 0.30),
                honorific=bool(self.rng.random() < 0.22),
                native=bool(script != "LATIN" and self.rng.random() < 0.18),
                middle=middle,
                middle_variant=int(self.rng.integers(0, 4)),
            )
            customers.iat[int(customer_row), name_col] = variant_name
            customers.iat[int(customer_row), uid_col] = str(entry["sdn_uid"])
            customers.iat[int(customer_row), customers.columns.get_loc("name_script")] = script
            if entry["dob"] is not None and not pd.isna(entry["dob"]) and self.rng.random() < 0.65:
                customers.iat[int(customer_row), customers.columns.get_loc("dob")] = str(entry["dob"])
            if entry["nationality"] is not None and not pd.isna(entry["nationality"]):
                customers.iat[int(customer_row), customers.columns.get_loc("nationality")] = str(
                    entry["nationality"]
                )

        near_col = customers.columns.get_loc("is_screening_near_miss")
        for offset, customer_row in enumerate(near_rows):
            entry = individuals.iloc[int(wl_positions[n_true + offset])]
            script = str(entry["name_script"])
            given_pool, family_pool = _SCRIPT_POOLS[script]
            family = family_pool[int(entry["_family_idx"])]
            # Near miss: the surname is genuinely shared — common surnames are
            # common — but the given name is a different person entirely. Half
            # of them additionally share the listed date of birth, which is the
            # combination that survives naive corroboration rules.
            other_given = given_pool[int(self.rng.integers(0, len(given_pool)))]
            listed_given = given_pool[int(entry["_given_idx"])]
            if other_given.canonical == listed_given.canonical:
                other_given = given_pool[(int(entry["_given_idx"]) + 3) % len(given_pool)]
            near_name = self._assemble(
                other_given, family,
                given_variant=int(self.rng.integers(0, len(other_given.variants))),
                family_variant=int(self.rng.integers(0, len(family.variants))),
                script=script,
                swap_order=False,
                honorific=False,
            )
            customers.iat[int(customer_row), name_col] = near_name
            customers.iat[int(customer_row), near_col] = True
            customers.iat[int(customer_row), customers.columns.get_loc("name_script")] = script
            if entry["dob"] is not None and not pd.isna(entry["dob"]) and self.rng.random() < 0.5:
                customers.iat[int(customer_row), customers.columns.get_loc("dob")] = str(entry["dob"])
        return customers

    # --------------------------------------------------------- counterparties
    def build_counterparty_pools(self, watchlist: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        """Three pools: benign, watchlist-confusable, and genuine watchlist hits.

        The confusable pool exists so that counterparty screening has to earn
        its precision. It is built from real watchlist surnames paired with
        unrelated given names, which is exactly the population that floods an
        L1 queue in production.
        """
        benign: list[str] = []
        for _ in range(4000):
            if self.rng.random() < 0.42:
                benign.append(self._render_org())
            else:
                script = str(self._choice(("LATIN", "LATIN", "ARABIC", "CYRILLIC", "HAN"), 1)[0])
                name, _, _, _, _ = self._render_person(script)
                benign.append(name)

        individuals = watchlist[watchlist["entity_type"] == "INDIVIDUAL"]
        confusable: list[str] = []
        true_hits: list[str] = []
        for _ in range(900):
            entry = individuals.iloc[int(self.rng.integers(0, len(individuals)))]
            script = str(entry["name_script"])
            given_pool, family_pool = _SCRIPT_POOLS[script]
            family = family_pool[int(entry["_family_idx"])]
            other = given_pool[int(self.rng.integers(0, len(given_pool)))]
            confusable.append(
                self._assemble(
                    other, family,
                    int(self.rng.integers(0, len(other.variants))),
                    int(self.rng.integers(0, len(family.variants))),
                    script, swap_order=False, honorific=False,
                )
            )
        for _ in range(400):
            entry = individuals.iloc[int(self.rng.integers(0, len(individuals)))]
            script = str(entry["name_script"])
            given_pool, family_pool = _SCRIPT_POOLS[script]
            given = given_pool[int(entry["_given_idx"])]
            family = family_pool[int(entry["_family_idx"])]
            true_hits.append(
                self._assemble(
                    given, family,
                    int(self.rng.integers(0, len(given.variants))),
                    int(self.rng.integers(0, len(family.variants))),
                    script,
                    swap_order=bool(self.rng.random() < 0.3),
                    honorific=bool(self.rng.random() < 0.2),
                )
            )
        return {
            "benign": tuple(benign),
            "confusable": tuple(confusable),
            "true_hit": tuple(true_hits),
            "banks": tuple(
                [f"{prefix} Bank" for prefix in nc.ORG_PREFIX]
                + [f"Banque {prefix}" for prefix in nc.ORG_PREFIX[:10]]
            ),
        }

    # ------------------------------------------------------------- baseline
    def _draw_counts(self, customers: pd.DataFrame, total: int) -> NDArray[Any]:
        """Per-customer transaction counts summing to exactly `total`.

        Gamma-Poisson rather than Poisson alone: real portfolios are
        over-dispersed, and the zero-inflation it produces is what creates the
        inactive customer-months that keep the scored population honest.
        """
        mean_map = {seg.segment: seg.annual_txns_mean for seg in self.config.activity_mix}
        shape_map = {seg.segment: seg.annual_txns_shape for seg in self.config.activity_mix}
        segment = customers["behaviour_segment"]
        means = segment.map(mean_map).to_numpy(dtype=float)
        dispersion = segment.map(shape_map).to_numpy(dtype=float)

        gamma_shape = 1.0 / dispersion
        intensity = self.rng.gamma(shape=gamma_shape, scale=means * dispersion)
        counts = self.rng.poisson(intensity).astype(np.int64)

        if counts.sum() == 0:
            raise ValueError("degenerate activity draw produced no transactions")
        counts = np.floor(counts * (total / counts.sum())).astype(np.int64)

        # Largest-remainder style reconciliation so the feed hits the declared
        # volume exactly; a "roughly 500k" dataset makes determinism diffs noisy.
        deficit = int(total - counts.sum())
        if deficit > 0:
            weights = (counts + 1).astype(float)
            picks = self.rng.choice(len(counts), size=deficit, p=weights / weights.sum())
            np.add.at(counts, picks, 1)
        elif deficit < 0:
            eligible = np.flatnonzero(counts > 0)
            picks = self.rng.choice(eligible, size=-deficit, replace=True)
            np.add.at(counts, picks, -1)
            counts = np.maximum(counts, 0)
            residual = int(total - counts.sum())
            if residual > 0:
                extra = self.rng.choice(len(counts), size=residual)
                np.add.at(counts, extra, 1)
        return cast("NDArray[Any]", counts)

    def _generate_baseline(
        self, customers: pd.DataFrame, total: int, pools: Mapping[str, tuple[str, ...]]
    ) -> pd.DataFrame:
        counts = self._draw_counts(customers, total)
        cust_idx = np.repeat(np.arange(len(customers)), counts)
        n_txn = int(cust_idx.size)

        segments = customers["behaviour_segment"].to_numpy()
        segment_names = list(_CHANNEL_MIX.keys())
        lookup = {name: i for i, name in enumerate(segment_names)}
        segment_code = customers["behaviour_segment"].map(lookup).to_numpy(dtype=np.int64)

        months = self.config.months
        month_starts = pd.date_range(self.window_start, periods=months, freq="MS")
        month_start_ns = month_starts.values.astype("datetime64[ns]").astype(np.int64)
        days_in_month = month_starts.days_in_month.to_numpy().astype(np.int64)

        # Activity is clustered, not uniform across the year: a dormant retail
        # account transacts in one or two months, a salaried account every month.
        base_month = self.rng.integers(0, months, size=len(customers))
        spread = np.empty(len(customers), dtype=np.int64)
        for name, options in _MONTH_SPREAD.items():
            mask = segments == name
            if mask.any():
                spread[mask] = np.asarray(options)[
                    self.rng.integers(0, len(options), size=int(mask.sum()))
                ]

        # Accounts opened part-way through the window can only transact from the
        # month they were opened. Without this floor a customer onboarded in
        # October carries transactions back to January, which breaks the
        # account-age scenario and fails the DQ10 consistency control.
        open_month = np.maximum(
            (
                (customers["account_open_date"].dt.year.to_numpy() - self.window_start.year) * 12
                + (customers["account_open_date"].dt.month.to_numpy() - self.window_start.month)
            ),
            0,
        ).astype(np.int64)
        available = np.maximum(months - open_month, 1)
        month = open_month[cust_idx] + (
            base_month[cust_idx] + (self.rng.random(n_txn) * spread[cust_idx]).astype(np.int64)
        ) % available[cust_idx]
        day = (self.rng.random(n_txn) * days_in_month[month]).astype(np.int64)
        # Same constraint within the opening month itself. The test is whether
        # the account was opened inside the observation window at all, not
        # whether its opening month index is non-zero: an account opened on the
        # 15th of the first month still cannot transact on the 3rd.
        opened_in_window = (
            customers["account_open_date"].to_numpy() >= np.datetime64(self.window_start)
        )
        opened_in_month = (month == open_month[cust_idx]) & opened_in_window[cust_idx]
        open_day = customers["account_open_date"].dt.day.to_numpy().astype(np.int64) - 1
        day = np.where(
            opened_in_month,
            np.minimum(open_day[cust_idx] + day, days_in_month[month] - 1),
            day,
        )
        night = self.rng.random(n_txn) < 0.17
        hour = np.where(
            night,
            self.rng.integers(19, 30, size=n_txn) % 24,
            self.rng.integers(8, 19, size=n_txn),
        )
        minute = self.rng.integers(0, 60, size=n_txn)
        second = self.rng.integers(0, 60, size=n_txn)
        ts_ns = month_start_ns[month] + (
            day * 86_400 + hour * 3_600 + minute * 60 + second
        ) * 1_000_000_000

        channel_matrix = np.cumsum(
            np.array([_CHANNEL_MIX[name] for name in segment_names], dtype=float), axis=1
        )
        draw = self.rng.random(n_txn)
        channel_idx = (draw[:, None] > channel_matrix[segment_code[cust_idx]]).sum(axis=1)
        channel_idx = np.clip(channel_idx, 0, 3)
        channel_names = np.array(["WIRE", "ACH", "CARD", "CASH"], dtype=object)
        channel = channel_names[channel_idx]

        # ------------------------------------------------------------------
        # Customer-level habits.
        #
        # Without these the baseline is unrealistically clean: no ordinary month
        # ever *looks* like a typology, so "carries an injected pattern" becomes
        # separable from "ordinary" at ROC-AUC 0.98 and the model's headline
        # number measures the generator rather than the problem. In a real
        # portfolio a large minority of cash businesses genuinely bank several
        # deposits just under the reporting threshold every month, importers
        # genuinely settle most of their value into one corridor, and plenty of
        # firms genuinely pay in round thousands. Those customers are the
        # legitimate population a monitoring system actually has to sift.
        #
        # These are traits of the CUSTOMER, not of individual transactions, so
        # they produce whole months with a typology-like shape.
        cash_business = np.isin(segments, ["CASH_INTENSIVE", "SMALL_BUSINESS"])
        high_denomination_depositor = cash_business & (
            self.rng.random(len(customers)) < 0.42
        )
        corridor_focused = (segments == "TRADE_CORRIDOR") & (
            self.rng.random(len(customers)) < 0.50
        )
        round_payer = np.isin(segments, ["SMALL_BUSINESS", "TRADE_CORRIDOR"]) & (
            self.rng.random(len(customers)) < 0.28
        )
        # A focused importer sends to the same jurisdiction month after month.
        focus_country = self._choice(self.high_risk + self.medium_risk, len(customers))

        amounts_cfg = self.config.amounts["lognormal"]
        customer_scale = np.exp(self.rng.normal(0.0, 0.55, size=len(customers)))
        channel_scale = np.array(
            [_CHANNEL_AMOUNT_SCALE[c] for c in ("WIRE", "ACH", "CARD", "CASH")]
        )
        credit_p = np.array([0.45, 0.50, 0.06, 0.72])
        direction = np.where(
            self.rng.random(n_txn) < credit_p[channel_idx], "CREDIT", "DEBIT"
        ).astype(object)

        amount = (
            self.rng.lognormal(float(amounts_cfg["mu"]), float(amounts_cfg["sigma"]), size=n_txn)
            * customer_scale[cust_idx]
            * channel_scale[channel_idx]
        )
        # High-denomination cash takings: a legitimate retail or hospitality
        # business banking a day's float lands in the same band a structurer
        # aims for. This is the single largest source of organic overlap with
        # the structuring typology.
        band = self.config.amounts["structuring_band"]
        near_threshold = (
            high_denomination_depositor[cust_idx]
            & (channel == "CASH")
            & (direction == "CREDIT")
            & (self.rng.random(n_txn) < 0.62)
        )
        amount = np.where(
            near_threshold,
            self.rng.uniform(band[0] * 0.94, band[1], size=n_txn),
            amount,
        )

        # A baseline share of genuinely round payments (rent, salary, invoices
        # settled in round lots) so that the round-dollar scenario cannot simply
        # key on "any round amount at all". Round-paying customers do it most of
        # the time, which is what produces whole round-valued months.
        round_mask = (self.rng.random(n_txn) < 0.13) | (
            round_payer[cust_idx]
            & np.isin(channel, ["WIRE", "ACH"])
            & (self.rng.random(n_txn) < 0.62)
        )
        multiples = np.array(self.config.amounts["round_dollar_multiples"], dtype=float)
        chosen_multiple = multiples[self.rng.integers(0, len(multiples), size=n_txn)]
        amount = np.where(
            round_mask,
            np.maximum(chosen_multiple, np.round(amount / chosen_multiple) * chosen_multiple),
            amount,
        )
        amount = np.clip(np.round(amount, 2), 1.0, 4_900_000.0)

        residence = customers["residence_country"].to_numpy()
        profile = np.array([_CORRIDOR_PROFILE[name] for name in segment_names])
        p_cross = profile[segment_code[cust_idx], 0]
        p_medium = profile[segment_code[cust_idx], 1]
        p_high = profile[segment_code[cust_idx], 2]
        cross = self.rng.random(n_txn) < p_cross
        tier_draw = self.rng.random(n_txn)
        country = residence[cust_idx].astype(object)
        high_pick = self._choice(self.high_risk, n_txn)
        medium_pick = self._choice(self.medium_risk, n_txn)
        low_pick = self._choice(self.low_risk, n_txn)
        country = np.where(cross & (tier_draw < p_high), high_pick, country)
        country = np.where(
            cross & (tier_draw >= p_high) & (tier_draw < p_high + p_medium), medium_pick, country
        )
        country = np.where(cross & (tier_draw >= p_high + p_medium), low_pick, country)

        # A focused importer concentrates cross-border value on one jurisdiction,
        # which is exactly the shape the high-risk-corridor scenario looks for
        # and is entirely legitimate trade.
        country = np.where(
            cross
            & corridor_focused[cust_idx]
            & (self.rng.random(n_txn) < 0.65),
            focus_country[cust_idx],
            country,
        )

        benign = np.asarray(pools["benign"], dtype=object)
        confusable = np.asarray(pools["confusable"], dtype=object)
        counterparty = benign[self.rng.integers(0, len(benign), size=n_txn)]
        # Confusable counterparties are ordinary business partners whose names
        # collide with the watchlist. They are true negatives, and they are the
        # dominant cost centre of a real screening operation.
        confusable_mask = self.rng.random(n_txn) < 0.030
        counterparty = np.where(
            confusable_mask,
            confusable[self.rng.integers(0, len(confusable), size=n_txn)],
            counterparty,
        )
        banks = np.asarray(pools["banks"], dtype=object)

        return pd.DataFrame(
            {
                "customer_id": customers["customer_id"].to_numpy()[cust_idx],
                "txn_ts": ts_ns.astype("datetime64[ns]"),
                "amount": amount,
                "currency": "USD",
                "channel": channel,
                "direction": direction,
                "counterparty_name": counterparty,
                "counterparty_country": country,
                "counterparty_bank": banks[self.rng.integers(0, len(banks), size=n_txn)],
                "is_cross_border": country != residence[cust_idx],
                "_origin": "BASELINE",
            }
        )

    # -------------------------------------------------------------- periods
    def _month_index(self, timestamps: pd.Series) -> NDArray[Any]:
        start = pd.Timestamp(self.window_start)
        return cast(
            "NDArray[Any]",
            (
                (timestamps.dt.year - start.year) * 12 + (timestamps.dt.month - start.month)
            ).to_numpy(dtype=np.int64),
        )

    # ------------------------------------------------------------- assembly
    def _active_cells(self, transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "customer_id": transactions["customer_id"].to_numpy(),
                "month": self._month_index(transactions["txn_ts"]),
            }
        )
        counts = (
            frame.groupby(["customer_id", "month"], sort=False)
            .size()
            .reset_index(name="txn_count")
        )
        segments = customers.set_index("customer_id")["behaviour_segment"]
        counts["behaviour_segment"] = counts["customer_id"].map(segments).to_numpy()
        return counts.sort_values(["customer_id", "month"], kind="mergesort").reset_index(drop=True)

    def _suppress_pre_dormancy(
        self, baseline: pd.DataFrame, cells: Sequence[InjectionCell]
    ) -> pd.DataFrame:
        """Make dormant-reactivation cells genuinely dormant.

        Applied to both populations. If only the anomalous ones had their prior
        activity cleared, "was dormant" would become a label tell, which is the
        exact failure this module exists to prevent.
        """
        targets = [c for c in cells if c.pattern == "dormant_reactivation" and c.intensity > 0]
        if not targets:
            return baseline
        months = self._month_index(baseline["txn_ts"])
        customer_ids = baseline["customer_id"].to_numpy()
        drop = np.zeros(len(baseline), dtype=bool)
        for cell in targets:
            window = (months >= cell.month - 3) & (months < cell.month)
            drop |= (customer_ids == cell.customer_id) & window
        removed = int(drop.sum())
        if removed:
            logger.info("suppressed pre-dormancy activity", extra={"rows_removed": removed})
        return baseline.loc[~drop].reset_index(drop=True)

    def _substitute(
        self,
        baseline: pd.DataFrame,
        cells: Sequence[InjectionCell],
        injected_counts: Mapping[tuple[str, int], int],
    ) -> pd.DataFrame:
        """Displace ordinary activity in every injected cell.

        Combined with the size-preserving renderer, this keeps the injected
        cell's transaction count where it started. Without it the injection is
        purely additive and "anomalous" degrades into "unusually busy" — a
        distinction an unsupervised model picks up with no labels at all, which
        is how the first version of this generator produced a 0.98 ROC-AUC
        Isolation Forest.
        """
        ratio = float(self.config.typologies["substitution_ratio"])
        if ratio <= 0.0 or not injected_counts:
            return baseline
        months = self._month_index(baseline["txn_ts"])
        customer_ids = baseline["customer_id"].to_numpy()
        keys = pd.MultiIndex.from_arrays([customer_ids, months])
        positions_by_key: dict[tuple[str, int], list[int]] = {}
        for position, key in enumerate(keys):
            positions_by_key.setdefault((str(key[0]), int(key[1])), []).append(position)

        drop = np.zeros(len(baseline), dtype=bool)
        for cell in cells:
            available = positions_by_key.get(cell.key)
            if not available:
                continue
            wanted = int(round(injected_counts.get(cell.key, 0) * ratio))
            if wanted <= 0:
                continue
            # The whole injected batch may displace the cell's entire ordinary
            # traffic: the rendered rows keep the customer-month alive, so the
            # cell never disappears and its size stays where it started.
            take = min(wanted, len(available))
            if take <= 0:
                continue
            chosen = self.rng.choice(np.asarray(available), size=take, replace=False)
            drop[chosen] = True
        removed = int(drop.sum())
        logger.info("baseline displaced by injection", extra={"rows_removed": removed})
        return baseline.loc[~drop].reset_index(drop=True)

    def _rebalance_to_target(
        self, frame: pd.DataFrame, target: int, protected: set[str]
    ) -> pd.DataFrame:
        """Trim or top up so the feed contains exactly the configured volume.

        Only unprotected baseline rows are eligible for removal, so trimming can
        never delete an injected typology or near-miss transaction and silently
        change a label.
        """
        # Window-filter FIRST. Reconciling the count and then dropping
        # out-of-window rows would land just under the declared volume, which is
        # how this previously produced 499,990 transactions instead of 500,000.
        frame = frame.loc[self._inside_window(frame)].reset_index(drop=True)

        surplus = len(frame) - target
        if surplus == 0:
            return frame
        if surplus > 0:
            eligible = np.flatnonzero(
                (frame["_origin"].to_numpy() == "BASELINE")
                & ~frame["customer_id"].isin(protected).to_numpy()
            )
            if len(eligible) < surplus:
                raise ValueError("cannot trim to target without touching injected rows")
            drop_positions = self.rng.choice(eligible, size=surplus, replace=False)
            keep = np.ones(len(frame), dtype=bool)
            keep[drop_positions] = False
            return frame.loc[keep].reset_index(drop=True)
        deficit = -surplus
        baseline_rows = np.flatnonzero(frame["_origin"].to_numpy() == "BASELINE")
        duplicates = frame.iloc[self.rng.choice(baseline_rows, size=deficit, replace=True)].copy()
        # Re-jitter the copies so the top-up is not a set of exact duplicate rows.
        # Forward only: shifting a copy backwards can push it before the account
        # opening date, which fails the DQ10 consistency control.
        offsets = self.rng.integers(0, 72, size=deficit)
        shifted = duplicates["txn_ts"] + pd.to_timedelta(offsets, unit="h")
        # A forward shift can run past the end of the window. Reflect those back
        # rather than dropping them, so the top-up delivers exactly `deficit`
        # rows and the declared volume is hit on the nose.
        overshoot = shifted >= pd.Timestamp(self.window_end)
        duplicates["txn_ts"] = shifted.where(
            ~overshoot, shifted - pd.to_timedelta(144, unit="h")
        )
        duplicates["amount"] = np.round(
            duplicates["amount"].to_numpy() * self.rng.uniform(0.92, 1.08, size=deficit), 2
        )
        combined = pd.concat([frame, duplicates], ignore_index=True)
        return combined.loc[self._inside_window(combined)].reset_index(drop=True)

    def _inside_window(self, frame: pd.DataFrame) -> "pd.Series[bool]":
        return (frame["txn_ts"] >= pd.Timestamp(self.window_start)) & (
            frame["txn_ts"] < pd.Timestamp(self.window_end)
        )

    def _finalize_transactions(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Rebalancing already applied the window filter; this is a guard.
        if not bool(self._inside_window(frame).all()):
            raise ValueError("transactions escaped the observation window after rebalancing")
        frame = frame.reset_index(drop=True)
        frame["_seq"] = np.arange(len(frame), dtype=np.int64)
        # Sorting on the synthetic sequence as the final key makes the row order
        # a pure function of the seed, which is what `make reproduce` verifies.
        frame = frame.sort_values(
            ["txn_ts", "customer_id", "_seq"], kind="mergesort"
        ).reset_index(drop=True)
        frame.insert(0, "txn_id", [f"TXN-{i + 1:08d}" for i in range(len(frame))])
        return frame.drop(columns=["_seq"])

    def _build_labels(
        self, transactions: pd.DataFrame, cells: Sequence[InjectionCell]
    ) -> pd.DataFrame:
        months = self._month_index(transactions["txn_ts"])
        active = pd.DataFrame(
            {"customer_id": transactions["customer_id"].to_numpy(), "month": months}
        ).drop_duplicates()
        active = active.sort_values(["customer_id", "month"], kind="mergesort").reset_index(
            drop=True
        )

        by_key = {cell.key: cell for cell in cells}
        keys = [
            (str(c), int(m))
            for c, m in zip(active["customer_id"].to_numpy(), active["month"].to_numpy())
        ]
        matched = [by_key.get(key) for key in keys]
        start = pd.Timestamp(self.window_start)
        return pd.DataFrame(
            {
                "customer_id": active["customer_id"].to_numpy(),
                "period": [
                    (start + pd.DateOffset(months=int(m))).strftime("%Y-%m")
                    for m in active["month"].to_numpy()
                ],
                "is_true_anomaly": [c is not None and c.is_anomaly for c in matched],
                "typology": [c.pattern if c is not None and c.is_anomaly else None for c in matched],
                "intensity": [
                    c.intensity_band if c is not None and c.is_anomaly else None for c in matched
                ],
                "is_near_miss": [c is not None and not c.is_anomaly for c in matched],
            }
        )

    def generate(self) -> GeneratedDataset:
        watchlist = self.generate_watchlist()
        customers = self.generate_customers(watchlist)
        pools = self.build_counterparty_pools(watchlist)

        target = self.config.n_transactions
        baseline = self._generate_baseline(customers, target, pools)

        active = self._active_cells(baseline, customers)
        cells = plan_injections(self.rng, self.config, active)

        baseline = self._suppress_pre_dormancy(baseline, cells)

        renderer = PatternRenderer(self.rng, self.config, pd.Timestamp(self.window_start), pools)
        customer_lookup = customers.set_index("customer_id")
        injected: list[dict[str, object]] = []
        injected_counts: dict[tuple[str, int], int] = {}
        for cell in cells:
            rows = renderer.render(cell, customer_lookup.loc[cell.customer_id])
            if rows:
                injected.extend(rows)
                injected_counts[cell.key] = len(rows)

        baseline = self._substitute(baseline, cells, injected_counts)

        combined = (
            pd.concat([baseline, pd.DataFrame(injected)], ignore_index=True)
            if injected
            else baseline
        )
        protected = {cell.customer_id for cell in cells}
        combined = self._rebalance_to_target(combined, target, protected)
        transactions = self._finalize_transactions(combined)
        labels = self._build_labels(transactions, cells)

        manifest: dict[str, object] = {
            "seed": self.config.seed,
            "window_start": self.window_start.date().isoformat(),
            "window_end": self.window_end.date().isoformat(),
            "watchlist_entities": len(watchlist),
            "customers": len(customers),
            "transactions": len(transactions),
            "scored_customer_months": len(labels),
            "true_anomaly_cells": int(labels["is_true_anomaly"].sum()),
            "true_anomaly_rate": round(float(labels["is_true_anomaly"].mean()), 6),
            "near_miss_cells": int(labels["is_near_miss"].sum()),
            "near_miss_rate": round(float(labels["is_near_miss"].mean()), 6),
            "invisible_anomalies": int((labels["intensity"] == "INVISIBLE").sum()),
            "attenuated_anomalies": int((labels["intensity"] == "ATTENUATED").sum()),
            "typology_counts": {
                str(k): int(v)
                for k, v in labels["typology"].value_counts().sort_index().items()
            },
            "screening_true_matches": int(customers["watchlist_true_match_uid"].notna().sum()),
            "screening_near_misses": int(customers["is_screening_near_miss"].sum()),
            "injected_transactions": int((transactions["_origin"] != "BASELINE").sum()),
        }
        logger.info("dataset generated", extra=manifest)

        transactions = transactions.drop(columns=["_origin"])
        watchlist = watchlist.drop(
            columns=[
                "_given_idx", "_family_idx", "_given_variant",
                "_family_variant", "_middle_idx",
            ]
        )
        return GeneratedDataset(
            watchlist=watchlist,
            customers=customers,
            transactions=transactions,
            labels=labels,
            manifest=manifest,
        )


def generate_dataset(config: GenerationConfig) -> GeneratedDataset:
    return SyntheticDataGenerator(config).generate()
