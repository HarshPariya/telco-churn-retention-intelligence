# Troubleshooting & Diagnostic Guide

## Telco Customer Churn Platform

---

## 1. Common Issues & Quick Fixes

| Issue | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError: WA_Fn-UseC...csv` | Dataset not downloaded yet | Run `python scripts/download_data.py` |
| `ImportError: cannot import name 'FallbackAsyncAdaptedQueuePool'` | Incompatible SQLAlchemy $\ge 2.1$ with MLflow | Run `uv pip install "sqlalchemy<2.1.0"` |
| `ValueError: could not convert string to float '[5.001E-1]'` | XGBoost 3.x string base score with SHAP TreeExplainer | Run `uv pip install "xgboost<3.0.0"` |
| `ModuleNotFoundError: No module named 'src'` | Working directory not in PYTHONPATH | Ensure `pythonpath = ["."]` in `pyproject.toml` or set `PYTHONPATH=.` |
| `Docker Compose build failure` | Docker daemon inactive | Ensure Docker Desktop is running |
| Streamlit dashboard cannot connect to API | Incorrect API URL | Set `API_URL=http://localhost:8000` in `.env` |

---

## 2. Diagnostic Commands

```bash
# Check test suite
pytest tests/ -v

# Check linter
ruff check .

# Inspect MLflow database
mlflow ui --backend-store-uri sqlite:///mlruns.db

# Test API curl
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"customer_id\": \"TEST\", \"gender\": \"Male\", \"SeniorCitizen\": 0, \"Partner\": \"No\", \"Dependents\": \"No\", \"tenure\": 5, \"PhoneService\": \"Yes\", \"MultipleLines\": \"No\", \"InternetService\": \"DSL\", \"OnlineSecurity\": \"No\", \"OnlineBackup\": \"No\", \"DeviceProtection\": \"No\", \"TechSupport\": \"No\", \"StreamingTV\": \"No\", \"StreamingMovies\": \"No\", \"Contract\": \"Month-to-month\", \"PaperlessBilling\": \"Yes\", \"PaymentMethod\": \"Electronic check\", \"MonthlyCharges\": 45.0, \"TotalCharges\": 225.0}"
```
