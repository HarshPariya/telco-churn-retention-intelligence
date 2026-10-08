"""Download and verify the raw Telco Customer Churn dataset."""

import sys
from pathlib import Path

import pandas as pd
import requests

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.telco_churn.config import load_config
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("download_data")

EXPECTED_ROW_COUNT = 7043
EXPECTED_COL_COUNT = 21
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

DATASET_URLS = [
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
]


def download_and_verify() -> Path:
    config = load_config()
    target_path = PROJECT_ROOT / config.data.raw_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        logger.info(f"Raw dataset already present at {target_path}. Verifying integrity...")
    else:
        logger.info(f"Raw dataset not found at {target_path}. Attempting automated download...")
        download_success = False
        for url in DATASET_URLS:
            try:
                logger.info(f"Fetching from {url}...")
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and len(response.content) > 100000:
                    with open(target_path, "wb") as f:
                        f.write(response.content)
                    download_success = True
                    logger.info("Download completed successfully.")
                    break
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")

        if not download_success:
            raise RuntimeError(
                f"Could not download dataset automatically.\n"
                f"Please manually place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' into: {target_path}\n"
                f"Source: Kaggle blastchar/telco-customer-churn or IBM Cognos sample dataset."
            )

    # Validate dataset schema & size
    df = pd.read_csv(target_path)
    logger.info(f"Dataset shape: {df.shape}")

    if df.shape[0] != EXPECTED_ROW_COUNT:
        logger.warning(f"Row count {df.shape[0]} differs from standard {EXPECTED_ROW_COUNT}.")
    if df.shape[1] != EXPECTED_COL_COUNT:
        raise ValueError(
            f"Column count {df.shape[1]} does not match expected {EXPECTED_COL_COUNT}."
        )

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    logger.info(
        f"Verification successful: {df.shape[0]} rows, {df.shape[1]} columns. Raw data verified immutable."
    )
    return target_path


if __name__ == "__main__":
    download_and_verify()
