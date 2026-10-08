"""Execute data cleaning and splitting pipeline."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.telco_churn.config import load_config
from src.telco_churn.data.preprocess import clean_and_preprocess_data
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("prepare_data")


def main() -> None:
    config = load_config()
    logger.info("Starting data preparation pipeline...")
    train_df, val_df, test_df = clean_and_preprocess_data(config)
    logger.info(
        f"Data preparation complete! "
        f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )


if __name__ == "__main__":
    main()
