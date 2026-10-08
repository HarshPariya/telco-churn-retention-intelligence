"""Feature engineering transformers and pipeline builder."""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.telco_churn.config import load_config
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("feature_engineering")


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain feature engineering transformer for Telco Customer Churn.

    Creates:
    - tenure_bucket: Segmented into [0-12m, 12-24m, 24-48m, 48-60m, 60-72m]
    - avg_monthly_spend: TotalCharges / tenure (or MonthlyCharges when tenure == 0)
    - services_count: Total count of subscribed services
    - monthly_to_total_ratio: Ratio of MonthlyCharges to TotalCharges (early tenure risk)
    - contract_monthly_risk: 1 if Month-to-month and MonthlyCharges > 65, else 0
    """

    def __init__(
        self,
        tenure_bins: Optional[List[int]] = None,
        tenure_labels: Optional[List[str]] = None,
    ) -> None:
        self.tenure_bins = tenure_bins or [0, 12, 24, 48, 60, 72]
        self.tenure_labels = tenure_labels or ["0-12m", "12-24m", "24-48m", "48-60m", "60-72m"]
        self.service_cols = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TelcoFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. tenure_bucket
        if "tenure" in df.columns:
            # Clip tenure to 0-72
            clipped_tenure = df["tenure"].clip(lower=0, upper=72)
            df["tenure_bucket"] = pd.cut(
                clipped_tenure,
                bins=[-1, 12, 24, 48, 60, 72],
                labels=self.tenure_labels,
                include_lowest=True,
            ).astype(str)

        # 2. avg_monthly_spend
        if "TotalCharges" in df.columns and "tenure" in df.columns:
            safe_tenure = np.where(df["tenure"] <= 0, 1.0, df["tenure"].astype(float))
            df["avg_monthly_spend"] = np.where(
                df["tenure"] <= 0,
                df.get("MonthlyCharges", 0.0),
                df["TotalCharges"].astype(float) / safe_tenure,
            )
            df["avg_monthly_spend"] = df["avg_monthly_spend"].round(2)

        # 3. services_count
        service_counts = np.zeros(len(df), dtype=int)
        for col in self.service_cols:
            if col in df.columns:
                service_counts += (df[col] == "Yes").astype(int)
        if "InternetService" in df.columns:
            service_counts += (df["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)
        df["services_count"] = service_counts

        # 4. monthly_to_total_ratio
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            df["monthly_to_total_ratio"] = (
                df["MonthlyCharges"].astype(float) / (df["TotalCharges"].astype(float) + 1.0)
            ).round(4)

        # 5. contract_monthly_risk
        if "Contract" in df.columns and "MonthlyCharges" in df.columns:
            df["contract_monthly_risk"] = (
                (df["Contract"] == "Month-to-month") & (df["MonthlyCharges"].astype(float) > 65.0)
            ).astype(int)

        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper to apply TelcoFeatureEngineer on a DataFrame."""
    engineer = TelcoFeatureEngineer()
    return engineer.transform(df)


def create_preprocessor(
    categorical_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Create scikit-learn ColumnTransformer for categorical encoding and numeric scaling."""
    config = load_config()

    if categorical_features is None:
        categorical_features = config.features.categorical_features

    if numeric_features is None:
        numeric_features = config.features.numeric_features

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor
