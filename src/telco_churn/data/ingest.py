"""Data ingestion module for Telco Customer Churn."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("data_ingest")


def load_raw_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load raw Telco dataset from immutable storage.

    Ensures that raw data is never mutated in place.
    """
    if file_path is None:
        config = load_config()
        path = PROJECT_ROOT / config.data.raw_path
    else:
        path = Path(file_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            f"Please run 'python scripts/download_data.py' to acquire the dataset."
        )

    logger.info(f"Loading raw dataset from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded raw dataset with shape: {df.shape}")
    return df
