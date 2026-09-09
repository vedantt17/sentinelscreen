"""Typed configuration objects.

Configs are frozen dataclasses rather than free-floating dicts so that a typo
in a YAML key fails at load time. During a threshold sweep, silently reading
`None` for a parameter would produce a plausible-looking but wrong tuning table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import yaml

from src.exceptions import ConfigurationError
from src.paths import CONFIG_DIR


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"malformed YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path} must contain a top-level mapping")
    return cast("dict[str, Any]", raw)


def _require(mapping: Mapping[str, Any], key: str, source: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"missing required key {key!r} in {source}")
    return mapping[key]


@dataclass(frozen=True)
class ActivitySegment:
    segment: str
    weight: float
    annual_txns_mean: float
    annual_txns_shape: float


@dataclass(frozen=True)
class GenerationConfig:
    seed: int
    start_date: date
    months: int
    n_watchlist: int
    n_customers: int
    n_transactions: int
    activity_mix: tuple[ActivitySegment, ...]
    watchlist: Mapping[str, Any]
    screening_ground_truth: Mapping[str, Any]
    typologies: Mapping[str, Any]
    corridors: Mapping[str, Sequence[str]]
    amounts: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], source: str = "generation.yaml"
    ) -> "GenerationConfig":
        volumes = _require(raw, "volumes", source)
        mix = tuple(
            ActivitySegment(
                segment=str(item["segment"]),
                weight=float(item["weight"]),
                annual_txns_mean=float(item["annual_txns_mean"]),
                annual_txns_shape=float(item["annual_txns_shape"]),
            )
            for item in _require(raw, "activity_mix", source)
        )
        weight_total = sum(seg.weight for seg in mix)
        if abs(weight_total - 1.0) > 1e-6:
            raise ConfigurationError(
                f"activity_mix weights must sum to 1.0, got {weight_total:.6f}"
            )
        start = _require(raw, "start_date", source)
        return cls(
            seed=int(_require(raw, "seed", source)),
            start_date=start if isinstance(start, date) else date.fromisoformat(str(start)),
            months=int(_require(raw, "months", source)),
            n_watchlist=int(volumes["watchlist_entities"]),
            n_customers=int(volumes["customers"]),
            n_transactions=int(volumes["transactions"]),
            activity_mix=mix,
            watchlist=dict(_require(raw, "watchlist", source)),
            screening_ground_truth=dict(_require(raw, "screening_ground_truth", source)),
            typologies=dict(_require(raw, "typologies", source)),
            corridors={k: list(v) for k, v in _require(raw, "corridors", source).items()},
            amounts=dict(_require(raw, "amounts", source)),
        )

    @property
    def high_risk_countries(self) -> frozenset[str]:
        return frozenset(self.corridors["high_risk"])

    @property
    def ctr_threshold(self) -> float:
        return float(self.amounts["ctr_threshold"])


@dataclass(frozen=True)
class ScoringWeights:
    jaro_winkler: float
    token_set: float
    phonetic: float
    initials: float

    def __post_init__(self) -> None:
        total = self.jaro_winkler + self.token_set + self.phonetic + self.initials
        if abs(total - 1.0) > 1e-9:
            raise ConfigurationError(f"scoring weights must sum to 1.0, got {total:.6f}")

    def as_dict(self) -> dict[str, float]:
        return {
            "jaro_winkler": self.jaro_winkler,
            "token_set": self.token_set,
            "phonetic": self.phonetic,
            "initials": self.initials,
        }


@dataclass(frozen=True)
class ScreeningConfig:
    rule_version: str
    honorifics: frozenset[str]
    corporate_suffixes: frozenset[str]
    strip_punctuation: bool
    collapse_whitespace: bool
    drop_single_char_tokens: bool
    blocking_strategies: tuple[str, ...]
    max_block_size: int
    min_token_length: int
    prefilter_jaro_winkler: float
    prefilter_max_candidates: int
    weights: ScoringWeights
    jaro_winkler_prefix_weight: float
    corroboration: Mapping[str, float]
    strong_name_floor: float
    strong_name_floor_score: float
    single_token_penalty: float
    auto_escalate: float
    review: float

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], source: str = "screening.yaml"
    ) -> "ScreeningConfig":
        norm = _require(raw, "normalization", source)
        blocking = _require(raw, "blocking", source)
        scoring = _require(raw, "scoring", source)
        thresholds = _require(raw, "thresholds", source)
        weights = ScoringWeights(**{k: float(v) for k, v in scoring["weights"].items()})
        auto_escalate = float(thresholds["auto_escalate"])
        review = float(thresholds["review"])
        if review > auto_escalate:
            raise ConfigurationError("review threshold cannot exceed auto_escalate threshold")
        return cls(
            rule_version=str(_require(raw, "rule_version", source)),
            honorifics=frozenset(str(h).lower() for h in norm["honorifics"]),
            corporate_suffixes=frozenset(str(s).lower() for s in norm["corporate_suffixes"]),
            strip_punctuation=bool(norm.get("strip_punctuation", True)),
            collapse_whitespace=bool(norm.get("collapse_whitespace", True)),
            drop_single_char_tokens=bool(norm.get("drop_single_char_tokens", False)),
            blocking_strategies=tuple(str(s) for s in blocking["strategies"]),
            max_block_size=int(blocking["max_block_size"]),
            min_token_length=int(blocking["min_token_length"]),
            prefilter_jaro_winkler=float(blocking.get("prefilter_jaro_winkler", 0.0)),
            prefilter_max_candidates=int(blocking.get("prefilter_max_candidates", 10_000)),
            weights=weights,
            jaro_winkler_prefix_weight=float(scoring["jaro_winkler_prefix_weight"]),
            corroboration={k: float(v) for k, v in scoring["corroboration"].items()},
            strong_name_floor=float(scoring["strong_name_floor"]),
            strong_name_floor_score=float(scoring["strong_name_floor_score"]),
            single_token_penalty=float(scoring["single_token_penalty"]),
            auto_escalate=auto_escalate,
            review=review,
        )


@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    name: str
    description: str
    params: Mapping[str, float]
    sweep_param: str
    sweep_values: tuple[float, ...]

    def with_param(self, key: str, value: float) -> "ScenarioConfig":
        if key not in self.params:
            raise ConfigurationError(f"scenario {self.id} has no parameter {key!r}")
        updated = dict(self.params)
        updated[key] = value
        return ScenarioConfig(
            id=self.id,
            name=self.name,
            description=self.description,
            params=updated,
            sweep_param=self.sweep_param,
            sweep_values=self.sweep_values,
        )


@dataclass(frozen=True)
class MonitoringConfig:
    rule_version: str
    scenarios: tuple[ScenarioConfig, ...]
    split: Mapping[str, Any]
    lightgbm: Mapping[str, Any]
    xgboost: Mapping[str, Any]
    isolation_forest: Mapping[str, Any]
    max_plausible_roc_auc: float
    min_plausible_roc_auc: float
    max_unsupervised_roc_auc: float
    target_recall: float

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], source: str = "monitoring_rules.yaml"
    ) -> "MonitoringConfig":
        scenarios = tuple(
            ScenarioConfig(
                id=str(item["id"]),
                name=str(item["name"]),
                description=str(item["description"]).strip(),
                params={k: float(v) for k, v in item["params"].items()},
                sweep_param=str(item["sweep_param"]),
                sweep_values=tuple(float(v) for v in item["sweep_values"]),
            )
            for item in _require(raw, "scenarios", source)
        )
        if len(scenarios) != 8:
            raise ConfigurationError(f"expected 8 monitoring scenarios, found {len(scenarios)}")
        for scenario in scenarios:
            if scenario.sweep_param not in scenario.params:
                raise ConfigurationError(
                    f"scenario {scenario.id} sweeps {scenario.sweep_param!r} "
                    "which is not one of its params"
                )
        model = _require(raw, "model", source)
        return cls(
            rule_version=str(_require(raw, "rule_version", source)),
            scenarios=scenarios,
            split=dict(model["split"]),
            lightgbm=dict(model["lightgbm"]),
            xgboost=dict(model["xgboost"]),
            isolation_forest=dict(model["isolation_forest"]),
            max_plausible_roc_auc=float(model["max_plausible_roc_auc"]),
            min_plausible_roc_auc=float(model["min_plausible_roc_auc"]),
            max_unsupervised_roc_auc=float(model.get("max_unsupervised_roc_auc", 0.95)),
            target_recall=float(model["target_recall"]),
        )

    def by_id(self, rule_id: str) -> ScenarioConfig:
        for scenario in self.scenarios:
            if scenario.id == rule_id:
                return scenario
        raise ConfigurationError(f"unknown scenario id {rule_id!r}")


@dataclass(frozen=True)
class DataQualityCheck:
    id: str
    dimension: str
    severity: str
    description: str
    sql: str

    def __post_init__(self) -> None:
        if self.severity not in {"BLOCKER", "CRITICAL", "WARNING"}:
            raise ConfigurationError(f"{self.id}: unknown severity {self.severity!r}")


@dataclass(frozen=True)
class AppConfig:
    generation: GenerationConfig
    screening: ScreeningConfig
    monitoring: MonitoringConfig
    dq_checks: tuple[DataQualityCheck, ...] = field(default_factory=tuple)

    @property
    def seed(self) -> int:
        return self.generation.seed


def load_app_config(config_dir: Path | None = None) -> AppConfig:
    directory = config_dir or CONFIG_DIR
    dq_raw = load_yaml(directory / "dq_checks.yaml")
    checks = tuple(
        DataQualityCheck(
            id=str(item["id"]),
            dimension=str(item["dimension"]),
            severity=str(item["severity"]),
            description=str(item["description"]).strip(),
            sql=str(item["sql"]).strip(),
        )
        for item in dq_raw["checks"]
    )
    if len(checks) != 10:
        raise ConfigurationError(f"expected 10 data-quality checks, found {len(checks)}")
    return AppConfig(
        generation=GenerationConfig.from_mapping(load_yaml(directory / "generation.yaml")),
        screening=ScreeningConfig.from_mapping(load_yaml(directory / "screening.yaml")),
        monitoring=MonitoringConfig.from_mapping(load_yaml(directory / "monitoring_rules.yaml")),
        dq_checks=checks,
    )
