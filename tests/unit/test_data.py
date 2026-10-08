"""Unit tests for data validation, cleaning, and splitting."""

import pandas as pd

from src.telco_churn.data.preprocess import clean_raw_dataframe, split_data
from src.telco_churn.data.validate import validate_raw_data


def test_clean_raw_dataframe_totalcharges_quirk():
    """Verify that blank string TotalCharges is coerced to 0.0 for tenure 0."""
    raw_data = pd.DataFrame(
        {
            "customerID": ["CUST-1", "CUST-2"],
            "tenure": [0, 5],
            "MonthlyCharges": [25.0, 50.0],
            "TotalCharges": [" ", "250.0"],
            "MultipleLines": ["No phone service", "Yes"],
            "OnlineSecurity": ["No internet service", "Yes"],
            "Churn": ["No", "Yes"],
        }
    )

    cleaned = clean_raw_dataframe(raw_data)

    # TotalCharges should be float
    assert cleaned["TotalCharges"].dtype in [float, "float64"]
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 250.0

    # Pseudo-categories collapsed
    assert cleaned["MultipleLines"].iloc[0] == "No"
    assert cleaned["OnlineSecurity"].iloc[0] == "No"

    # Target mapped to binary int
    assert cleaned["Churn"].iloc[0] == 0
    assert cleaned["Churn"].iloc[1] == 1


def test_validate_raw_data_detects_issues():
    """Verify programmatic validation flags anomalies."""
    invalid_data = pd.DataFrame(
        {
            "customerID": ["CUST-1", "CUST-1"],  # Duplicate ID
            "gender": ["Alien", "Female"],  # Invalid category
            "SeniorCitizen": [0, 0],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [10, 20],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "No"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": [" ", "1200.0"],  # Blank string
            "Churn": ["Maybe", "No"],  # Invalid target
        }
    )

    report = validate_raw_data(invalid_data)
    assert not report.validation_passed
    assert any("Duplicate" in iss or "duplicate" in iss for iss in report.issues_detected)
    assert any("invalid category" in iss for iss in report.issues_detected)


def test_split_data_preserves_stratification():
    """Verify that split_data maintains class proportion across splits."""
    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i}" for i in range(100)],
            "tenure": [i % 72 for i in range(100)],
            "MonthlyCharges": [50.0] * 100,
            "TotalCharges": [50.0 * (i % 72) for i in range(100)],
            "Churn": [1 if i < 26 else 0 for i in range(100)],  # ~26% churn
        }
    )

    train_df, val_df, test_df = split_data(df, test_size=0.20, val_size=0.20, random_state=42)

    assert len(train_df) == 64
    assert len(val_df) == 16
    assert len(test_df) == 20

    # Churn rates should be approximately equal
    assert abs(train_df["Churn"].mean() - 0.26) < 0.05
    assert abs(test_df["Churn"].mean() - 0.26) < 0.05
