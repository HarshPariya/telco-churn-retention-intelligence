"""FastAPI dependency injection module."""

from src.telco_churn.config import AppConfig, load_config
from src.telco_churn.inference.predictor import TelcoChurnPredictor, get_predictor


def get_app_config() -> AppConfig:
    """Dependency provider for application configuration."""
    return load_config()


def get_model_predictor() -> TelcoChurnPredictor:
    """Dependency provider for singleton model predictor."""
    return get_predictor()
