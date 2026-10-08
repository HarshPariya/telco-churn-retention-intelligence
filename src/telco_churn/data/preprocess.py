"""Data cleaning, normalization, and splitting module."""

from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.telco_churn.config import PROJECT_ROOT, AppConfig, load_config
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("data_preprocess")


def clean_raw_dataframe(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """Clean raw dataframe applying domain rules and assignment requirements.

    1. Coerce TotalCharges to numeric (replacing blank strings with NaN).
    2. Impute TotalCharges NaNs with 0.0 (verified to be tenure == 0 accounts).
    3. Collapse "No internet service" and "No phone service" pseudo-categories into "No".
    4. Encode Churn target (Yes -> 1, No -> 0) if present.
    """
    df_clean = df.copy()

    # TotalCharges coercion
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        # For new customers with tenure == 0, TotalCharges is 0.0
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Collapse pseudo-categories for internet services
    internet_sub_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in internet_sub_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace({"No internet service": "No"})

    # Collapse pseudo-categories for phone services
    if "MultipleLines" in df_clean.columns:
        df_clean["MultipleLines"] = df_clean["MultipleLines"].replace({"No phone service": "No"})

    # Target variable encoding
    if "Churn" in df_clean.columns and not is_inference:
        if df_clean["Churn"].dtype == object:
            df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

    return df_clean


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
    stratify_col: Optional[str] = "Churn",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified split into Train, Validation, and Holdout Test sets.

    Split strategy:
    - Holdout test set = test_size (e.g. 20%)
    - Remaining 80% is split into Train (1 - val_size) and Validation (val_size).
    """
    logger.info(
        f"Splitting dataset of shape {df.shape} (test_size={test_size}, val_size={val_size}, seed={random_state})..."
    )

    stratify_target = df[stratify_col] if stratify_col in df.columns else None

    # Step 1: Split off final holdout test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Step 2: Split train_val into train and validation sets
    stratify_train_val = (
        train_val_df[stratify_col] if stratify_col in train_val_df.columns else None
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    logger.info(f"Split complete: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")
    if stratify_col in df.columns:
        logger.info(f"Train Churn Rate: {train_df[stratify_col].mean():.2%}")
        logger.info(f"Val Churn Rate:   {val_df[stratify_col].mean():.2%}")
        logger.info(f"Test Churn Rate:  {test_df[stratify_col].mean():.2%}")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def clean_and_preprocess_data(
    config: Optional[AppConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end data preparation and save splits."""
    if config is None:
        config = load_config()

    raw_path = PROJECT_ROOT / config.data.raw_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    raw_df = pd.read_csv(raw_path)
    clean_df = clean_raw_dataframe(raw_df)

    # Save cleaned interim data
    interim_path = PROJECT_ROOT / config.data.interim_path
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(interim_path, index=False)
    logger.info(f"Saved interim cleaned data to {interim_path}")

    # Split into train, val, test
    train_df, val_df, test_df = split_data(
        clean_df,
        test_size=config.split.test_size,
        val_size=config.split.val_size,
        random_state=config.split.random_state,
        stratify_col=config.data.target_column,
    )

    # Save processed splits
    train_path = PROJECT_ROOT / config.data.train_path
    val_path = PROJECT_ROOT / config.data.val_path
    test_path = PROJECT_ROOT / config.data.test_path
    processed_path = PROJECT_ROOT / config.data.processed_path

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    clean_df.to_parquet(processed_path, index=False)

    logger.info(f"Saved processed splits to {train_path.parent}")
    return train_df, val_df, test_df
