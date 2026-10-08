"""Unit tests for production inference predictor and SHAP explanations."""

from src.telco_churn.inference.predictor import (
    CustomerPredictionRequest,
    get_predictor,
)


def test_predictor_single_prediction():
    """Verify that predictor returns valid prediction with SHAP drivers."""
    predictor = get_predictor()

    req = CustomerPredictionRequest(
        customer_id="TEST-001",
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=2,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=89.5,
        TotalCharges=179.0,
    )

    res = predictor.predict_single(req)

    assert res.customer_id == "TEST-001"
    assert 0.0 <= res.churn_probability <= 1.0
    assert res.churn_prediction in (0, 1)
    assert res.risk_level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    assert res.clv > 0
    assert res.retention_priority_score >= 0
    assert len(res.top_drivers) == 3

    # Check driver format
    driver = res.top_drivers[0]
    assert driver.feature != ""
    assert driver.direction in ("INCREASES_CHURN", "DECREASES_CHURN")
    assert driver.impact >= 0.0


def test_predictor_batch_prediction():
    """Verify batch prediction on multiple customers."""
    import pandas as pd

    predictor = get_predictor()

    df = pd.DataFrame(
        [
            {
                "customerID": "BATCH-1",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 70.0,
            },
            {
                "customerID": "BATCH-2",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 65.0,
                "TotalCharges": 3900.0,
            },
        ]
    )

    batch_res = predictor.predict_dataframe(df, batch_explain=True)

    assert batch_res.total_records == 2
    assert len(batch_res.predictions) == 2
    assert batch_res.predictions[0].customer_id == "BATCH-1"
    assert batch_res.predictions[1].customer_id == "BATCH-2"
    # Month-to-month customer should have higher risk than Two-year loyal customer
    assert batch_res.predictions[0].churn_probability > batch_res.predictions[1].churn_probability
