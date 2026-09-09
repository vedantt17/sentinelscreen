"""Transaction monitoring: DuckDB scenarios, features, models, ATL/BTL, diagnostics."""

from src.monitoring.atl_btl import recommended_thresholds, run_atl_btl
from src.monitoring.diagnostics import separability_report
from src.monitoring.features import BaselineTransformer, build_base_features, prepare_matrices
from src.monitoring.models import ModelEvaluation, OperatingPoint, TrainedModels, train_all
from src.monitoring.rules import RuleEngine, ScenarioResult, combine_alerts

__all__ = [
    "BaselineTransformer",
    "ModelEvaluation",
    "OperatingPoint",
    "RuleEngine",
    "ScenarioResult",
    "TrainedModels",
    "build_base_features",
    "combine_alerts",
    "prepare_matrices",
    "recommended_thresholds",
    "run_atl_btl",
    "separability_report",
    "train_all",
]
