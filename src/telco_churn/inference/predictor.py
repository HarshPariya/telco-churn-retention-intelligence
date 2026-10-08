"""Unified inference and prediction service."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.telco_churn.business.prioritization import (
    assign_risk_tier,
    calculate_clv,
    calculate_retention_priority,
    convert_currency,
)
from src.telco_churn.config import load_config
from src.telco_churn.data.preprocess import clean_raw_dataframe
from src.telco_churn.explainability.shap_explainer import TelcoShapExplainer
from src.telco_churn.logging_config import configure_logger
from src.telco_churn.models.registry import ModelRegistry

logger = configure_logger("inference_predictor")


class CustomerPredictionRequest(BaseModel):
    customer_id: Optional[str] = Field(
        default="CUST-PREDICT", description="Optional customer account ID"
    )
    gender: str = Field(default="Female", description="Demographic gender: Male, Female")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Senior Citizen status (0 or 1)")
    Partner: str = Field(default="No", description="Partner status: Yes, No")
    Dependents: str = Field(default="No", description="Dependents status: Yes, No")
    tenure: int = Field(default=1, ge=0, le=120, description="Tenure in months (0-120)")
    PhoneService: str = Field(default="No", description="Phone service: Yes, No")
    MultipleLines: str = Field(
        default="No phone service", description="Multiple lines: Yes, No, No phone service"
    )
    InternetService: str = Field(
        default="DSL", description="Internet service: DSL, Fiber optic, No"
    )
    OnlineSecurity: str = Field(
        default="No", description="Online security: Yes, No, No internet service"
    )
    OnlineBackup: str = Field(
        default="Yes", description="Online backup: Yes, No, No internet service"
    )
    DeviceProtection: str = Field(
        default="No", description="Device protection: Yes, No, No internet service"
    )
    TechSupport: str = Field(default="No", description="Tech support: Yes, No, No internet service")
    StreamingTV: str = Field(default="No", description="Streaming TV: Yes, No, No internet service")
    StreamingMovies: str = Field(
        default="No", description="Streaming Movies: Yes, No, No internet service"
    )
    Contract: str = Field(
        default="Month-to-month", description="Contract: Month-to-month, One year, Two year"
    )
    PaperlessBilling: str = Field(default="Yes", description="Paperless billing: Yes, No")
    PaymentMethod: str = Field(
        default="Electronic check",
        description="Payment method: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)",
    )
    MonthlyCharges: float = Field(default=29.85, ge=0.0, description="Monthly billing charges ($)")
    TotalCharges: Optional[Union[float, str]] = Field(
        default=29.85, description="Total lifetime charges ($ or string)"
    )


class CustomerDriver(BaseModel):
    feature: str
    direction: str
    impact: float


class CustomerPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_level: str
    threshold: float
    clv: float
    retention_priority_score: float
    clv_inr: float
    retention_priority_inr: float
    top_drivers: List[CustomerDriver]
    model_version: str


class BatchPredictionResponse(BaseModel):
    total_records: int
    high_risk_count: int
    total_at_risk_clv: float
    predictions: List[CustomerPredictionResponse]


class TelcoChurnPredictor:
    """Production inference engine encapsulating model, preprocessor, and explainability."""

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        self.registry = ModelRegistry(registry_dir)
        self.pipeline, self.metadata = self.registry.load_production_model()
        self.explainer = TelcoShapExplainer(self.pipeline)
        self.config = load_config()
        logger.info(
            f"Initialized TelcoChurnPredictor: Model={self.metadata.algorithm}, "
            f"Threshold={self.metadata.optimal_threshold}, Version={self.metadata.model_version}"
        )

    def predict_single(self, request: CustomerPredictionRequest) -> CustomerPredictionResponse:
        """Process and predict churn for a single customer profile."""
        req_dict = request.model_dump()
        cid = req_dict.pop("customer_id", "CUST-0001") or "CUST-0001"

        # Convert to single-row dataframe
        df_raw = pd.DataFrame([req_dict])
        df_clean = clean_raw_dataframe(df_raw, is_inference=True)

        # Predict probability
        y_prob = float(self.pipeline.predict_proba(df_clean)[0, 1])
        threshold = self.metadata.optimal_threshold
        prediction = 1 if y_prob >= threshold else 0

        # Calculate business indicators
        tenure_val = float(df_clean["tenure"].iloc[0])
        monthly_val = float(df_clean["MonthlyCharges"].iloc[0])
        clv_val = float(calculate_clv(monthly_val, tenure_val))
        priority_val = float(calculate_retention_priority(y_prob, clv_val))
        risk_level = assign_risk_tier(y_prob)

        # Financial conversions
        clv_inr = convert_currency(clv_val)
        priority_inr = convert_currency(priority_val)

        # SHAP explainability
        drivers = self.explainer.explain_instance(df_clean, top_k=3)
        formatted_drivers = [
            CustomerDriver(feature=d.feature, direction=d.direction, impact=d.impact)
            for d in drivers
        ]

        return CustomerPredictionResponse(
            customer_id=cid,
            churn_probability=round(y_prob, 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            threshold=round(threshold, 4),
            clv=round(clv_val, 2),
            retention_priority_score=round(priority_val, 2),
            clv_inr=clv_inr,
            retention_priority_inr=priority_inr,
            top_drivers=formatted_drivers,
            model_version=self.metadata.model_version,
        )

    def predict_dataframe(
        self, df: pd.DataFrame, batch_explain: bool = True
    ) -> BatchPredictionResponse:
        """Process and predict churn for a batch DataFrame of customer records."""
        df_work = df.copy()
        customer_ids = (
            df_work["customerID"].tolist()
            if "customerID" in df_work.columns
            else [f"CUST-{i:04d}" for i in range(len(df_work))]
        )

        # Drop ID and target if present
        cols_to_drop = [c for c in ["customerID", "Churn"] if c in df_work.columns]
        df_features = df_work.drop(columns=cols_to_drop, errors="ignore")

        df_clean = clean_raw_dataframe(df_features, is_inference=True)
        y_probs = self.pipeline.predict_proba(df_clean)[:, 1]
        threshold = self.metadata.optimal_threshold

        # Compute CLV and Priority
        tenures = df_clean["tenure"].values
        monthly = df_clean["MonthlyCharges"].values
        clvs_arr = np.asarray(calculate_clv(monthly, tenures))
        priorities_arr = np.asarray(calculate_retention_priority(y_probs, clvs_arr))

        # SHAP explanations
        if batch_explain:
            all_drivers = self.explainer.explain_batch(df_clean, top_k=3)
        else:
            all_drivers = [[] for _ in range(len(df_clean))]

        responses = []
        high_risk_count = 0
        total_at_risk_clv = 0.0

        for i in range(len(df_clean)):
            prob = float(y_probs[i])
            pred = 1 if prob >= threshold else 0
            risk_tier = assign_risk_tier(prob)
            clv = float(clvs_arr[i])
            priority = float(priorities_arr[i])

            if risk_tier in ("CRITICAL", "HIGH"):
                high_risk_count += 1
                total_at_risk_clv += clv

            drivers_formatted = (
                [
                    CustomerDriver(feature=d.feature, direction=d.direction, impact=d.impact)
                    for d in all_drivers[i]
                ]
                if all_drivers[i]
                else []
            )

            responses.append(
                CustomerPredictionResponse(
                    customer_id=str(customer_ids[i]),
                    churn_probability=round(prob, 4),
                    churn_prediction=pred,
                    risk_level=risk_tier,
                    threshold=round(threshold, 4),
                    clv=round(clv, 2),
                    retention_priority_score=round(priority, 2),
                    clv_inr=convert_currency(clv),
                    retention_priority_inr=convert_currency(priority),
                    top_drivers=drivers_formatted,
                    model_version=self.metadata.model_version,
                )
            )

        return BatchPredictionResponse(
            total_records=len(df_clean),
            high_risk_count=high_risk_count,
            total_at_risk_clv=round(total_at_risk_clv, 2),
            predictions=responses,
        )


# Global singleton instance for high-performance API serving
_PREDICTOR_INSTANCE: Optional[TelcoChurnPredictor] = None


def get_predictor() -> TelcoChurnPredictor:
    """Singleton getter to prevent re-instantiating model pipeline per request."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = TelcoChurnPredictor()
    return _PREDICTOR_INSTANCE
