"""FastAPI logging configuration."""

from src.telco_churn.logging_config import configure_logger

api_logger = configure_logger("telco_api")
