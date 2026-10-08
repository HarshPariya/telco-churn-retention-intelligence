"""Inference service package."""

from src.telco_churn.inference.predictor import (
    BatchPredictionResponse,
    CustomerPredictionRequest,
    CustomerPredictionResponse,
    TelcoChurnPredictor,
    get_predictor,
)

__all__ = [
    "CustomerPredictionRequest",
    "CustomerPredictionResponse",
    "BatchPredictionResponse",
    "TelcoChurnPredictor",
    "get_predictor",
]
