"""Structured logging configuration for the Telco Churn platform."""

import logging
import os
import sys
from typing import Optional


def configure_logger(
    name: str = "telco_churn",
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a structured logger."""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)

        if log_format is None:
            log_format = (
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            )

        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
