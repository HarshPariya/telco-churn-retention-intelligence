"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """Verify GET /health returns 200 with model info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_name" in data
    assert "optimal_threshold" in data


def test_api_metadata_endpoint():
    """Verify GET /metadata returns comprehensive model information."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm" in data
    assert "metrics" in data
    assert "feature_names" in data
    assert len(data["feature_names"]) > 10


def test_api_predict_single_endpoint():
    """Verify POST /predict returns prediction and top 3 drivers."""
    payload = {
        "customer_id": "API-TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 239.55,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "API-TEST-001"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert "clv" in data
    assert "retention_priority_score" in data
    assert "clv_inr" in data
    assert len(data["top_drivers"]) == 3
    for d in data["top_drivers"]:
        assert "feature" in d
        assert "direction" in d
        assert "impact" in d


def test_api_predict_batch_endpoint():
    """Verify POST /predict/batch handles multiple records."""
    payload = {
        "customers": [
            {
                "customer_id": "BATCH-API-1",
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
                "customer_id": "BATCH-API-2",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 50,
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
                "TotalCharges": 3250.0,
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["customer_id"] == "BATCH-API-1"
    assert data["predictions"][1]["customer_id"] == "BATCH-API-2"


def test_api_predict_validation_error():
    """Verify that invalid payloads return structured 422 error."""
    payload = {
        "tenure": -5,  # Invalid negative tenure
        "MonthlyCharges": -100.0,  # Invalid negative charges
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "details" in data
