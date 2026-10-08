"""Modeling, evaluation, and registry package."""

from src.telco_churn.models.evaluate import compute_metrics, evaluate_predictions
from src.telco_churn.models.registry import (
    ModelRegistry,
    load_production_artifact,
    save_production_artifact,
)
from src.telco_churn.models.train import build_pipeline, run_model_comparison, train_model
from src.telco_churn.models.tune import tune_hyperparameters

__all__ = [
    "build_pipeline",
    "train_model",
    "run_model_comparison",
    "evaluate_predictions",
    "compute_metrics",
    "tune_hyperparameters",
    "ModelRegistry",
    "save_production_artifact",
    "load_production_artifact",
]
