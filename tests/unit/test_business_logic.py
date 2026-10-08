"""Unit tests for business logic, CLV, and retention prioritization."""

import numpy as np

from src.telco_churn.business.prioritization import (
    assign_risk_tier,
    calculate_clv,
    calculate_retention_priority,
    convert_currency,
    optimize_threshold,
)


def test_calculate_clv_scalar():
    """Verify CLV calculation = MonthlyCharges * tenure."""
    assert calculate_clv(50.0, 10.0) == 500.0
    assert calculate_clv(75.5, 24.0) == 1812.0
    # Boundary: tenure 0 should default to 1 month spend
    assert calculate_clv(100.0, 0.0) == 100.0


def test_calculate_clv_vector():
    """Verify vector calculation of CLV."""
    mc = np.array([20.0, 50.0, 80.0])
    tenure = np.array([0.0, 12.0, 24.0])
    expected = np.array([20.0, 600.0, 1920.0])
    result = calculate_clv(mc, tenure)
    np.testing.assert_array_almost_equal(result, expected)


def test_calculate_retention_priority():
    """Verify retention priority = probability * CLV."""
    # 80% prob on $1,000 CLV -> 800 priority score
    assert calculate_retention_priority(0.80, 1000.0) == 800.0
    # 20% prob on $5,000 CLV -> 1000 priority score
    assert calculate_retention_priority(0.20, 5000.0) == 1000.0
    # Out-of-bounds probability clipping
    assert calculate_retention_priority(1.5, 100.0) == 100.0
    assert calculate_retention_priority(-0.2, 100.0) == 0.0


def test_assign_risk_tier():
    """Verify risk tier assignment matches thresholds."""
    bands = {"critical": 0.70, "high": 0.50, "medium": 0.30}
    assert assign_risk_tier(0.85, bands) == "CRITICAL"
    assert assign_risk_tier(0.55, bands) == "HIGH"
    assert assign_risk_tier(0.35, bands) == "MEDIUM"
    assert assign_risk_tier(0.15, bands) == "LOW"


def test_convert_currency():
    """Verify currency conversion to INR."""
    # 100 USD at 83.50 rate
    assert convert_currency(100.0, 83.50) == 8350.0
    assert convert_currency(0.0, 83.50) == 0.0


def test_optimize_threshold_minimizes_cost():
    """Verify cost-sensitive threshold optimizer."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    clv = np.array([1000.0, 800.0, 1200.0, 500.0, 400.0, 600.0])

    thresh, metrics = optimize_threshold(y_true, y_prob, clv, retention_offer_cost=35.0)
    assert 0.15 <= thresh <= 0.85
    assert "recall" in metrics
    assert "total_business_cost" in metrics
    assert metrics["recall"] > 0.50
