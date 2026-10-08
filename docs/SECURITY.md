# Security Policy & Security Audit

## Telco Customer Churn & Retention Optimization Platform

---

## 1. Secrets & Credentials Policy

- **Zero Committed Secrets:** No API keys, passwords, database credentials, or private certificates are stored in Git history.
- **Environment Separation:** All deployment secrets and configuration variables are loaded exclusively via environment variables using `.env` (development) or orchestration secrets (production).
- **Template Guidance:** `.env.example` provides explicit documentation of configuration keys without embedding actual secrets.

---

## 2. Input Validation & Injection Defense

- **Strong Typing via Pydantic:** Every incoming API payload is parsed, validated, and sanitized through strict Pydantic schemas.
- **Enum Category Restrictions:** Feature inputs (`Contract`, `InternetService`, `PaymentMethod`) are restricted to allowed sets. Any unapproved category is rejected with HTTP 422.
- **Numerical Bounds:** Constraints enforce logical bounds (e.g. `tenure` $\in [0, 120]$, `MonthlyCharges` $\ge 0$).
- **No Unsafe Deserialization:** Models are loaded strictly from controlled internal registry paths (`models/production_pipeline.joblib`) via `joblib`. Arbitrary user-uploaded pickled models are rejected.
- **CSV Upload Safeguards:** Streamlit CSV ingestion uses pure `pandas.read_csv()` parsing with column schema validation, preventing arbitrary code execution.

---

## 3. Container Security

- **Non-Root Execution:** Docker containers create and run as an unprivileged user (`appuser` with UID 1000).
- **Minimal Attack Surface:** Container images utilize `python:3.11-slim` with build dependencies removed post-compilation.
- **Read-Only / Principle of Least Privilege:** Production containers only require read access to `models/` and `configs/`.

---

## 4. API Error Shielding

- Generic server exception handlers catch and log internal errors with full stack traces in server logs while returning clean, uninformative error messages to clients (e.g., `{"error": "Internal server error occurred"}`) to prevent information leakage.
