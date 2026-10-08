"""Unit tests for feature engineering transformations."""

import pandas as pd

from src.telco_churn.features.build import TelcoFeatureEngineer, create_preprocessor


def test_feature_engineer_transformer():
    """Verify that engineered features match domain business logic."""
    df = pd.DataFrame(
        {
            "tenure": [0, 8, 36, 65],
            "MonthlyCharges": [30.0, 70.0, 85.0, 110.0],
            "TotalCharges": [0.0, 560.0, 3060.0, 7150.0],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "PhoneService": ["Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "Yes", "Yes"],
            "InternetService": ["No", "Fiber optic", "DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "No", "Yes", "Yes"],
            "OnlineBackup": ["No", "No", "Yes", "Yes"],
            "DeviceProtection": ["No", "No", "No", "Yes"],
            "TechSupport": ["No", "No", "Yes", "Yes"],
            "StreamingTV": ["No", "Yes", "Yes", "Yes"],
            "StreamingMovies": ["No", "Yes", "Yes", "Yes"],
        }
    )

    engineer = TelcoFeatureEngineer()
    df_transformed = engineer.transform(df)

    # 1. tenure_bucket
    assert "tenure_bucket" in df_transformed.columns
    assert df_transformed["tenure_bucket"].iloc[0] == "0-12m"
    assert df_transformed["tenure_bucket"].iloc[1] == "0-12m"
    assert df_transformed["tenure_bucket"].iloc[2] == "24-48m"
    assert df_transformed["tenure_bucket"].iloc[3] == "60-72m"

    # 2. avg_monthly_spend
    assert "avg_monthly_spend" in df_transformed.columns
    assert df_transformed["avg_monthly_spend"].iloc[0] == 30.0
    assert df_transformed["avg_monthly_spend"].iloc[1] == 70.0  # 560 / 8

    # 3. services_count
    assert "services_count" in df_transformed.columns
    # Row 0: PhoneService only = 1
    assert df_transformed["services_count"].iloc[0] == 1
    # Row 3: Phone(1) + Multiple(1) + Fiber(1) + Security(1) + Backup(1) + Protect(1) + Support(1) + TV(1) + Movies(1) = 9
    assert df_transformed["services_count"].iloc[3] == 9

    # 4. contract_monthly_risk
    assert "contract_monthly_risk" in df_transformed.columns
    # Row 1 is Month-to-month with MonthlyCharges > 65 -> 1
    assert df_transformed["contract_monthly_risk"].iloc[1] == 1
    # Row 2 is One year -> 0
    assert df_transformed["contract_monthly_risk"].iloc[2] == 0


def test_column_transformer_fit_transform():
    """Verify that create_preprocessor cleanly transforms features."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "tenure": [12, 48],
            "MonthlyCharges": [45.0, 95.0],
            "TotalCharges": [540.0, 4560.0],
            "tenure_bucket": ["0-12m", "48-60m"],
            "avg_monthly_spend": [45.0, 95.0],
            "services_count": [2, 6],
            "monthly_to_total_ratio": [0.08, 0.02],
            "contract_monthly_risk": [0, 0],
        }
    )

    preprocessor = create_preprocessor()
    X_out = preprocessor.fit_transform(df)

    assert X_out.shape[0] == 2
    assert X_out.shape[1] > 20  # Numeric + one-hot categories
