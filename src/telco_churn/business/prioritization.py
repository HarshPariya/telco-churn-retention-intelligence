"""Business logic module for Customer Lifetime Value (CLV) and Retention Prioritization."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.telco_churn.config import load_config
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("business_prioritization")


def calculate_clv(
    monthly_charges: Union[float, np.ndarray, pd.Series],
    tenure: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate Customer Lifetime Value (CLV).

    Formula specified by project requirements:
        CLV = MonthlyCharges * tenure

    For customers with tenure == 0 (new signups), minimum baseline CLV is
    clamped to 1 * MonthlyCharges to represent current first-month commitment.
    """
    if isinstance(monthly_charges, (pd.Series, np.ndarray)) or isinstance(
        tenure, (pd.Series, np.ndarray)
    ):
        mc = np.maximum(np.asarray(monthly_charges, dtype=float), 0.0)
        t = np.maximum(np.asarray(tenure, dtype=float), 0.0)
        # If tenure is 0, give 1 month of spend as current commitment
        effective_tenure = np.where(t == 0, 1.0, t)
        clv = mc * effective_tenure
        if isinstance(monthly_charges, pd.Series):
            return pd.Series(clv, index=monthly_charges.index)
        return clv

    mc_val = max(float(monthly_charges), 0.0)
    t_val = max(float(tenure), 0.0)
    effective_t = 1.0 if t_val == 0 else t_val
    return round(mc_val * effective_t, 2)


def calculate_retention_priority(
    churn_probability: Union[float, np.ndarray, pd.Series],
    clv: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate Retention Priority Score.

    Formula:
        retention_priority_score = churn_probability * CLV
    """
    if isinstance(churn_probability, (pd.Series, np.ndarray)) or isinstance(
        clv, (pd.Series, np.ndarray)
    ):
        prob = np.clip(np.asarray(churn_probability, dtype=float), 0.0, 1.0)
        val = np.maximum(np.asarray(clv, dtype=float), 0.0)
        score = prob * val
        if isinstance(churn_probability, pd.Series):
            return pd.Series(score, index=churn_probability.index)
        return score

    prob_val = min(max(float(churn_probability), 0.0), 1.0)
    clv_val = max(float(clv), 0.0)
    return round(prob_val * clv_val, 2)


def assign_risk_tier(
    churn_probability: float,
    risk_bands: Optional[Dict[str, float]] = None,
) -> str:
    """Assign human-readable risk category based on churn probability."""
    if risk_bands is None:
        cfg = load_config()
        risk_bands = {
            "critical": cfg.business.risk_bands.critical,
            "high": cfg.business.risk_bands.high,
            "medium": cfg.business.risk_bands.medium,
        }

    prob = float(churn_probability)
    if prob >= risk_bands.get("critical", 0.70):
        return "CRITICAL"
    elif prob >= risk_bands.get("high", 0.50):
        return "HIGH"
    elif prob >= risk_bands.get("medium", 0.30):
        return "MEDIUM"
    else:
        return "LOW"


def convert_currency(amount_usd: float, rate: Optional[float] = None) -> float:
    """Convert amount in USD to INR using configurable conversion rate."""
    if rate is None:
        cfg = load_config()
        rate = cfg.business.usd_to_inr_rate
    return round(amount_usd * rate, 2)


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clv_values: np.ndarray,
    retention_offer_cost: float = 35.0,
    churn_loss_factor: float = 1.0,
    intervention_success_rate: float = 0.35,
) -> Tuple[float, Dict[str, Any]]:
    """Determine the optimal classification threshold maximizing net business value.

    Business Financial Model:
    - True Positive (Contacted Churner):
        Cost = retention_offer_cost + (1 - intervention_success_rate) * CLV * churn_loss_factor
    - False Positive (Contacted Non-Churner):
        Cost = retention_offer_cost
    - False Negative (Uncontacted Churner):
        Cost = CLV * churn_loss_factor
    - True Negative (Uncontacted Non-Churner):
        Cost = 0.0
    """
    thresholds = np.linspace(0.15, 0.85, 71)
    best_threshold = 0.50
    min_cost = float("inf")
    metrics_at_best: Dict[str, Any] = {}

    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    clv = np.asarray(clv_values).astype(float)

    for thresh in thresholds:
        preds = (y_p >= thresh).astype(int)

        fn_mask = (y_t == 1) & (preds == 0)
        fp_mask = (y_t == 0) & (preds == 1)
        tp_mask = (y_t == 1) & (preds == 1)

        cost_fn = float(np.sum(clv[fn_mask] * churn_loss_factor))
        cost_fp = float(np.sum(fp_mask) * retention_offer_cost)
        cost_tp = float(
            np.sum(
                retention_offer_cost
                + (1.0 - intervention_success_rate) * clv[tp_mask] * churn_loss_factor
            )
        )

        total_cost = cost_fn + cost_fp + cost_tp

        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = round(float(thresh), 2)

            # Record performance metrics at this threshold
            tp = int(tp_mask.sum())
            fp = int(fp_mask.sum())
            fn = int(fn_mask.sum())
            tn = int(((y_t == 0) & (preds == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics_at_best = {
                "threshold": best_threshold,
                "total_business_cost": round(total_cost, 2),
                "cost_fn_lost_clv": round(cost_fn, 2),
                "cost_campaign_offers": round(cost_fp + cost_tp, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

    logger.info(
        f"Cost-sensitive threshold optimization: optimal threshold={best_threshold}, "
        f"min business cost=${min_cost:,.2f}, recall={metrics_at_best['recall']:.2%}"
    )
    return best_threshold, metrics_at_best
