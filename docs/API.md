# REST API Documentation

## Telco Customer Churn Prediction & Retention Optimization API

**Base URL (Local):** `http://localhost:8000`  
**Swagger Interactive Documentation:** `http://localhost:8000/docs`  
**ReDoc Documentation:** `http://localhost:8000/redoc`  

---

## 1. Endpoints Overview

| Method | Endpoint | Description | Auth Required |
| --- | --- | --- | --- |
| `GET` | `/health` | Service health status, model version, and optimal threshold | None |
| `GET` | `/metadata` | Detailed production model metadata, features, and test metrics | None |
| `POST` | `/predict` | Predict churn probability, risk level, CLV, priority, and SHAP drivers for a single customer | None |
| `POST` | `/predict/batch` | Batch predict multiple customer profiles with ranking summary | None |

---

## 2. Endpoint Details

### 2.1 GET `/health`

Check API service readiness and verify model artifact is active in memory.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "telco-churn-retention-model",
  "model_version": "v1.0.0",
  "optimal_threshold": 0.23,
  "timestamp": "2026-10-08T07:40:00Z"
}
```

---

### 2.2 GET `/metadata`

Retrieve active model algorithm details, feature list, and actual holdout evaluation metrics.

**Response (200 OK):**

```json
{
  "model_name": "telco-churn-retention-model",
  "model_version": "v1.0.0",
  "algorithm": "Tuned XGBoost Classifier (scale_pos_weight=2.77)",
  "trained_at": "2026-10-08T07:39:20.123456+00:00",
  "optimal_threshold": 0.23,
  "feature_names": [
    "num__tenure",
    "num__MonthlyCharges",
    "num__TotalCharges",
    "num__avg_monthly_spend",
    "num__services_count",
    "cat__Contract_Month-to-month",
    "cat__PaymentMethod_Electronic check"
  ],
  "metrics": {
    "roc_auc": 0.8439,
    "pr_auc": 0.6582,
    "recall": 0.9385,
    "precision": 0.4105,
    "f1": 0.5712,
    "brier_score": 0.1631
  },
  "business_assumptions": {
    "retention_offer_cost": 35.0,
    "churn_loss_factor": 1.0,
    "usd_to_inr_rate": 83.5
  }
}
```

---

### 2.3 POST `/predict`

Score a single customer account and receive explainability drivers.

**Request Body:**

```json
{
  "customer_id": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

**Response (200 OK):**

```json
{
  "customer_id": "7590-VHVEG",
  "churn_probability": 0.7421,
  "churn_prediction": 1,
  "risk_level": "CRITICAL",
  "threshold": 0.23,
  "clv": 29.85,
  "retention_priority_score": 22.15,
  "clv_inr": 2492.48,
  "retention_priority_inr": 1849.53,
  "top_drivers": [
    {
      "feature": "Month-to-Month Contract",
      "direction": "INCREASES_CHURN",
      "impact": 0.8421
    },
    {
      "feature": "Account Tenure (months)",
      "direction": "INCREASES_CHURN",
      "impact": 0.6125
    },
    {
      "feature": "Electronic Check Payment",
      "direction": "INCREASES_CHURN",
      "impact": 0.3842
    }
  ],
  "model_version": "v1.0.0"
}
```

---

### 2.4 POST `/predict/batch`

Process a batch list of customer profiles.

**Request Body:**

```json
{
  "customers": [
    {
      "customer_id": "CUST-001",
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 2,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "Yes",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 89.5,
      "TotalCharges": 179.0
    }
  ]
}
```

**Response (200 OK):**

```json
{
  "total_records": 1,
  "high_risk_count": 1,
  "total_at_risk_clv": 179.0,
  "predictions": [
    {
      "customer_id": "CUST-001",
      "churn_probability": 0.8254,
      "churn_prediction": 1,
      "risk_level": "CRITICAL",
      "threshold": 0.23,
      "clv": 179.0,
      "retention_priority_score": 147.75,
      "clv_inr": 14946.5,
      "retention_priority_inr": 12337.13,
      "top_drivers": [
        {
          "feature": "Month-to-Month Contract",
          "direction": "INCREASES_CHURN",
          "impact": 0.8654
        }
      ],
      "model_version": "v1.0.0"
    }
  ]
}
```

---

## 3. Error Handling Specifications

All errors return structured JSON with an HTTP status code:

- `400 Bad Request`: Malformed or empty request payload.
- `422 Unprocessable Entity`: Request validation failure (e.g. negative tenure, unknown category).
- `500 Internal Server Error`: Server exception (details logged to console without leaking raw stack trace to caller).
