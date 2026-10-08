"""Drift detection and data distribution monitoring."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import ks_2samp

from src.telco_churn.logging_config import configure_logger

logger = configure_logger("monitoring_drift")


class FeatureDriftMetric(BaseModel):
    feature_name: str
    feature_type: str
    psi: float
    drift_detected: bool
    status: str  # "STABLE", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"
    details: Dict[str, Any]


class DriftReport(BaseModel):
    timestamp: str
    total_baseline_records: int
    total_current_records: int
    overall_status: str
    features_drifted_count: int
    metrics: List[FeatureDriftMetric]


def calculate_psi_numeric(
    baseline: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) for continuous numerical features."""
    base_clean = baseline[~np.isnan(baseline)]
    curr_clean = current[~np.isnan(current)]

    if len(base_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine quantile bins on baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(base_clean, quantiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5

    # Handle duplicate bin edges
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    base_counts, _ = np.histogram(base_clean, bins=bins)
    curr_counts, _ = np.histogram(curr_clean, bins=bins)

    base_pct = np.maximum(base_counts / len(base_clean), eps)
    curr_pct = np.maximum(curr_counts / len(curr_clean), eps)

    psi_val = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return round(float(psi_val), 4)


def calculate_psi_categorical(
    baseline: pd.Series,
    current: pd.Series,
    eps: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) for discrete categorical features."""
    all_categories = sorted(set(baseline.dropna().unique()) | set(current.dropna().unique()))
    if not all_categories:
        return 0.0

    base_counts = baseline.value_counts(normalize=True)
    curr_counts = current.value_counts(normalize=True)

    psi_val = 0.0
    for cat in all_categories:
        b_pct = max(base_counts.get(cat, 0.0), eps)
        c_pct = max(curr_counts.get(cat, 0.0), eps)
        psi_val += (c_pct - b_pct) * np.log(c_pct / b_pct)

    return round(float(psi_val), 4)


def detect_dataset_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features_to_monitor: Optional[List[str]] = None,
    psi_warning_threshold: float = 0.10,
    psi_critical_threshold: float = 0.20,
) -> DriftReport:
    """Audit baseline vs inference dataset for distribution drift."""
    from datetime import datetime, timezone

    if features_to_monitor is None:
        features_to_monitor = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "PaymentMethod",
            "InternetService",
            "OnlineSecurity",
            "TechSupport",
        ]

    metrics = []
    drifted_count = 0

    for feat in features_to_monitor:
        if feat not in baseline_df.columns or feat not in current_df.columns:
            continue

        base_col = baseline_df[feat]
        curr_col = current_df[feat]

        if pd.api.types.is_numeric_dtype(base_col):
            feat_type = "numeric"
            psi = calculate_psi_numeric(base_col.values, curr_col.values)
            # Kolmogorov-Smirnov 2-sample test
            ks_res = ks_2samp(base_col.dropna(), curr_col.dropna())
            details = {
                "ks_statistic": round(float(ks_res.statistic), 4),
                "ks_pvalue": round(float(ks_res.pvalue), 4),
            }
        else:
            feat_type = "categorical"
            psi = calculate_psi_categorical(base_col, curr_col)
            details = {}

        if psi >= psi_critical_threshold:
            status = "SIGNIFICANT_DRIFT"
            drift_detected = True
            drifted_count += 1
        elif psi >= psi_warning_threshold:
            status = "MODERATE_DRIFT"
            drift_detected = True
            drifted_count += 1
        else:
            status = "STABLE"
            drift_detected = False

        metrics.append(
            FeatureDriftMetric(
                feature_name=feat,
                feature_type=feat_type,
                psi=psi,
                drift_detected=drift_detected,
                status=status,
                details=details,
            )
        )

    overall_status = (
        "CRITICAL"
        if any(m.status == "SIGNIFICANT_DRIFT" for m in metrics)
        else ("WARNING" if drifted_count > 0 else "HEALTHY")
    )

    report = DriftReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_baseline_records=len(baseline_df),
        total_current_records=len(current_df),
        overall_status=overall_status,
        features_drifted_count=drifted_count,
        metrics=metrics,
    )

    logger.info(
        f"Drift Audit complete: Status={overall_status}, Drifted Features={drifted_count}/{len(metrics)}"
    )
    return report
