"""Configuration management module for the Telco Churn Platform."""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataConfig(BaseModel):
    raw_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    interim_path: str = "data/interim/telco_churn_cleaned.parquet"
    processed_path: str = "data/processed/telco_churn_processed.parquet"
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"
    test_path: str = "data/processed/test.parquet"
    id_column: str = "customerID"
    target_column: str = "Churn"
    positive_class: str = "Yes"
    negative_class: str = "No"
    numeric_coercion_columns: List[str] = Field(default_factory=lambda: ["TotalCharges"])
    internet_service_sub_features: List[str] = Field(
        default_factory=lambda: [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
    )
    phone_service_sub_features: List[str] = Field(default_factory=lambda: ["MultipleLines"])
    all_service_features: List[str] = Field(
        default_factory=lambda: [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
    )


class SplitConfig(BaseModel):
    test_size: float = 0.20
    val_size: float = 0.20
    stratify: bool = True
    random_state: int = 42


class FeaturesConfig(BaseModel):
    tenure_bucket_bins: List[int] = Field(default_factory=lambda: [0, 12, 24, 48, 60, 72])
    tenure_bucket_labels: List[str] = Field(
        default_factory=lambda: ["0-12m", "12-24m", "24-48m", "48-60m", "60-72m"]
    )
    categorical_features: List[str] = Field(
        default_factory=lambda: [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
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
            "tenure_bucket",
        ]
    )
    numeric_features: List[str] = Field(
        default_factory=lambda: [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "avg_monthly_spend",
            "services_count",
            "monthly_to_total_ratio",
            "contract_monthly_risk",
        ]
    )


class RiskBandsConfig(BaseModel):
    critical: float = 0.70
    high: float = 0.50
    medium: float = 0.30


class BusinessConfig(BaseModel):
    retention_offer_cost: float = float(os.getenv("RETENTION_OFFER_COST", "35.0"))
    churn_loss_factor: float = float(os.getenv("CHURN_LOSS_FACTOR", "1.0"))
    usd_to_inr_rate: float = float(os.getenv("USD_TO_INR_RATE", "83.5"))
    risk_bands: RiskBandsConfig = Field(default_factory=RiskBandsConfig)


class ArtifactsConfig(BaseModel):
    model_dir: str = "models"
    model_file: str = "models/production_pipeline.joblib"
    metadata_file: str = "models/metadata.json"
    shap_explainer_file: str = "models/shap_explainer.joblib"
    figures_dir: str = "reports/figures"
    tables_dir: str = "reports/tables"


class MLflowConfig(BaseModel):
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-customer-churn")


class ApiConfig(BaseModel):
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    title: str = "Telco Churn & Retention Optimization Platform API"
    version: str = "1.0.0"


class DashboardConfig(BaseModel):
    port: int = int(os.getenv("DASHBOARD_PORT", "8501"))
    title: str = "Telco Retention Intelligence Dashboard"
    api_url: str = os.getenv("API_URL", "http://localhost:8000")


class ModelingConfig(BaseModel):
    models_to_evaluate: List[str] = Field(
        default_factory=lambda: ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    )
    cv_folds: int = 5
    primary_metric: str = "roc_auc"
    decision_metric: str = "f1"


class AppConfig(BaseModel):
    project_name: str = "telco-customer-churn"
    version: str = "1.0.0"
    random_seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    business: BusinessConfig = Field(default_factory=BusinessConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load and parse application configuration from YAML with fallback to defaults."""
    target_path: Path
    if config_path:
        target_path = Path(config_path)
    else:
        env = os.getenv("APP_ENV", "development").lower()
        candidate = PROJECT_ROOT / "configs" / f"{env}.yaml"
        if candidate.exists():
            target_path = candidate
        else:
            target_path = PROJECT_ROOT / "configs" / "base.yaml"

    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        return AppConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    # Handle base inheritance if present
    if "_base_" in raw_dict:
        base_file = PROJECT_ROOT / raw_dict.pop("_base_")
        if base_file.exists():
            with open(base_file, "r", encoding="utf-8") as bf:
                base_dict = yaml.safe_load(bf) or {}
                # Deep merge base with raw_dict
                for k, v in raw_dict.items():
                    if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                        base_dict[k].update(v)
                    else:
                        base_dict[k] = v
                raw_dict = base_dict

    return AppConfig.model_validate(raw_dict)
