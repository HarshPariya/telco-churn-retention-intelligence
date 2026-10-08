"""SHAP Explainability Engine for Telco Churn Predictions."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from src.telco_churn.config import PROJECT_ROOT
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("shap_explainer")

# Friendly labels for business users
FEATURE_NAME_MAP = {
    "num__tenure": "Account Tenure (months)",
    "num__MonthlyCharges": "Monthly Charges ($)",
    "num__TotalCharges": "Total Lifetime Charges ($)",
    "num__avg_monthly_spend": "Average Monthly Spend ($)",
    "num__services_count": "Subscribed Services Count",
    "num__monthly_to_total_ratio": "Monthly to Total Charges Ratio",
    "num__contract_monthly_risk": "Month-to-month High Spend Risk",
    "cat__Contract_Month-to-month": "Month-to-Month Contract",
    "cat__Contract_One year": "One-Year Contract",
    "cat__Contract_Two year": "Two-Year Contract",
    "cat__InternetService_Fiber optic": "Fiber Optic Internet",
    "cat__InternetService_DSL": "DSL Internet",
    "cat__InternetService_No": "No Internet Service",
    "cat__PaymentMethod_Electronic check": "Electronic Check Payment",
    "cat__PaymentMethod_Mailed check": "Mailed Check Payment",
    "cat__PaymentMethod_Bank transfer (automatic)": "Automatic Bank Transfer",
    "cat__PaymentMethod_Credit card (automatic)": "Automatic Credit Card",
    "cat__OnlineSecurity_No": "No Online Security",
    "cat__OnlineSecurity_Yes": "Online Security Subscribed",
    "cat__TechSupport_No": "No Tech Support",
    "cat__TechSupport_Yes": "Tech Support Subscribed",
    "cat__PaperlessBilling_Yes": "Paperless Billing Enabled",
    "cat__PaperlessBilling_No": "Paperless Billing Disabled",
    "cat__tenure_bucket_0-12m": "Tenure Bucket: 0-12 months",
    "cat__tenure_bucket_12-24m": "Tenure Bucket: 12-24 months",
    "cat__tenure_bucket_24-48m": "Tenure Bucket: 24-48 months",
    "cat__tenure_bucket_48-60m": "Tenure Bucket: 48-60 months",
    "cat__tenure_bucket_60-72m": "Tenure Bucket: 60-72 months",
}


class ExplanationDriver(BaseModel):
    feature: str
    technical_name: str
    direction: str  # "INCREASES_CHURN" or "DECREASES_CHURN"
    impact: float
    shap_value: float


class TelcoShapExplainer:
    """SHAP explainer wrapper for production inference and global audits."""

    def __init__(self, pipeline: Pipeline, background_data: Optional[pd.DataFrame] = None) -> None:
        self.pipeline = pipeline
        self.feature_engineer = pipeline.named_steps["feature_engineer"]
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]

        # Feature names after preprocessing
        self.feature_names = list(self.preprocessor.get_feature_names_out())

        # Transform background data if provided
        self.background_transformed = None
        if background_data is not None:
            bg_eng = self.feature_engineer.transform(background_data)
            self.background_transformed = self.preprocessor.transform(bg_eng)

        # Initialize explainer
        self.explainer = self._init_explainer()

    def _init_explainer(self) -> Any:
        classifier_name = type(self.classifier).__name__.lower()
        if "forest" in classifier_name or "xgb" in classifier_name or "lgbm" in classifier_name:
            logger.info(f"Initializing SHAP TreeExplainer for {classifier_name}...")
            try:
                return shap.TreeExplainer(self.classifier)
            except Exception as e:
                logger.warning(
                    f"TreeExplainer initialization failed: {e}. Falling back to shap.Explainer..."
                )
                if self.background_transformed is not None:
                    return shap.Explainer(
                        self.classifier.predict_proba, self.background_transformed
                    )
                return shap.Explainer(self.classifier)
        else:
            logger.info(f"Initializing SHAP LinearExplainer for {classifier_name}...")
            if self.background_transformed is not None:
                return shap.LinearExplainer(self.classifier, self.background_transformed)
            else:
                return shap.Explainer(self.classifier)

    def _transform_input(self, df: pd.DataFrame) -> np.ndarray:
        df_eng = self.feature_engineer.transform(df)
        return self.preprocessor.transform(df_eng)

    def explain_instance(self, df_single: pd.DataFrame, top_k: int = 3) -> List[ExplanationDriver]:
        """Explain a single customer prediction and return top-k drivers."""
        X_trans = self._transform_input(df_single)

        raw_shap_vals = self.explainer.shap_values(X_trans)

        # Handle various SHAP return structures across tree and linear models
        if isinstance(raw_shap_vals, list):
            # Binary classification list [class_0, class_1]
            shap_vec = raw_shap_vals[1][0]
        elif isinstance(raw_shap_vals, np.ndarray):
            if raw_shap_vals.ndim == 3:  # (samples, features, classes)
                shap_vec = raw_shap_vals[0, :, 1]
            elif raw_shap_vals.ndim == 2:  # (samples, features)
                shap_vec = raw_shap_vals[0]
            else:
                shap_vec = raw_shap_vals
        else:
            shap_vec = np.asarray(raw_shap_vals.values)[0]

        # Rank features by absolute impact
        indices = np.argsort(np.abs(shap_vec))[::-1][:top_k]

        drivers = []
        for idx in indices:
            tech_name = self.feature_names[idx]
            friendly_name = FEATURE_NAME_MAP.get(
                tech_name, tech_name.replace("num__", "").replace("cat__", "").replace("_", " ")
            )
            s_val = float(shap_vec[idx])
            direction = "INCREASES_CHURN" if s_val > 0 else "DECREASES_CHURN"

            drivers.append(
                ExplanationDriver(
                    feature=friendly_name,
                    technical_name=tech_name,
                    direction=direction,
                    impact=round(abs(s_val), 4),
                    shap_value=round(s_val, 4),
                )
            )

        return drivers

    def explain_batch(
        self, df_batch: pd.DataFrame, top_k: int = 3
    ) -> List[List[ExplanationDriver]]:
        """Explain a batch of customer records and return top-k drivers for each."""
        X_trans = self._transform_input(df_batch)
        raw_shap_vals = self.explainer.shap_values(X_trans)

        if isinstance(raw_shap_vals, list):
            shap_matrix = raw_shap_vals[1]
        elif isinstance(raw_shap_vals, np.ndarray):
            if raw_shap_vals.ndim == 3:
                shap_matrix = raw_shap_vals[:, :, 1]
            else:
                shap_matrix = raw_shap_vals
        else:
            shap_matrix = np.asarray(raw_shap_vals.values)

        batch_drivers = []
        for row_idx in range(len(df_batch)):
            shap_vec = shap_matrix[row_idx]
            indices = np.argsort(np.abs(shap_vec))[::-1][:top_k]

            row_drivers = []
            for idx in indices:
                tech_name = self.feature_names[idx]
                friendly_name = FEATURE_NAME_MAP.get(
                    tech_name, tech_name.replace("num__", "").replace("cat__", "").replace("_", " ")
                )
                s_val = float(shap_vec[idx])
                direction = "INCREASES_CHURN" if s_val > 0 else "DECREASES_CHURN"

                row_drivers.append(
                    ExplanationDriver(
                        feature=friendly_name,
                        technical_name=tech_name,
                        direction=direction,
                        impact=round(abs(s_val), 4),
                        shap_value=round(s_val, 4),
                    )
                )
            batch_drivers.append(row_drivers)

        return batch_drivers

    def generate_global_plots(
        self,
        df_sample: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        """Generate and save global SHAP summary plot and feature importance bar plot."""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        X_trans = self._transform_input(df_sample)
        raw_shap_vals = self.explainer.shap_values(X_trans)

        if isinstance(raw_shap_vals, list):
            shap_matrix = raw_shap_vals[1]
        elif isinstance(raw_shap_vals, np.ndarray) and raw_shap_vals.ndim == 3:
            shap_matrix = raw_shap_vals[:, :, 1]
        else:
            shap_matrix = raw_shap_vals

        # Readable feature names for plot
        readable_names = [
            FEATURE_NAME_MAP.get(name, name.replace("num__", "").replace("cat__", ""))
            for name in self.feature_names
        ]

        # 1. Summary plot (Beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_matrix,
            X_trans,
            feature_names=readable_names,
            show=False,
            max_display=15,
        )
        plt.title("SHAP Feature Impact Summary on Customer Churn", fontsize=14, pad=15)
        plt.tight_layout()
        summary_path = output_dir / "shap_summary_plot.png"
        plt.savefig(summary_path, dpi=200, bbox_inches="tight")
        plt.close()

        # 2. Bar plot (Global Importance)
        plt.figure(figsize=(10, 6))
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:12]

        top_names = [readable_names[i] for i in sorted_indices]
        top_vals = [mean_abs_shap[i] for i in sorted_indices]

        plt.barh(top_names[::-1], top_vals[::-1], color="#1f77b4")
        plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
        plt.title("Top 12 Features Driving Customer Churn", fontsize=14)
        plt.tight_layout()
        bar_path = output_dir / "shap_feature_importance.png"
        plt.savefig(bar_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP plots generated at: {summary_path} and {bar_path}")
        return {
            "summary_plot": str(summary_path),
            "bar_plot": str(bar_path),
        }
