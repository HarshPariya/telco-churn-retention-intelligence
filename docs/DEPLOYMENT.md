# Deployment Guide

## Telco Customer Churn & Retention Optimization Platform

This document describes how to deploy the platform locally, within Docker containers, and to cloud hosting providers.

---

## 1. Local Development Deployment

### Prerequisites

- Python 3.11+
- Git
- `uv` package manager (or standard `pip`)

### Step-by-Step Local Setup

1. **Clone & Initialize Repository:**

   ```bash
   git clone <repo-url>
   cd "Internship Project"
   ```

2. **Create Virtual Environment & Install Dependencies:**

   ```bash
   uv venv .venv --python 3.11
   .venv\Scripts\activate   # On Windows
   # source .venv/bin/activate  # On Linux/macOS
   uv pip install -r requirements.txt
   ```

3. **Data Acquisition & Model Training:**

   ```bash
   python scripts/download_data.py
   python scripts/validate_data.py
   python scripts/prepare_data.py
   python scripts/train_model.py
   python scripts/evaluate_model.py
   python scripts/generate_reports.py
   ```

4. **Launch Backend API:**

   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   Verify: Visit `http://localhost:8000/health` or `http://localhost:8000/docs`.

5. **Launch Streamlit Dashboard:**

   ```bash
   streamlit run dashboard/app.py --server.port 8501
   ```

   Verify: Visit `http://localhost:8501`.

---

## 2. Docker & Docker Compose Deployment

### Build and Launch Multi-Container Stack

1. **Build Container Images:**

   ```bash
   docker compose build
   ```

2. **Start Services in Background:**

   ```bash
   docker compose up -d
   ```

3. **Inspect Running Containers & Health Checks:**

   ```bash
   docker compose ps
   ```

4. **Verify Health Endpoints:**

   ```bash
   curl -f http://localhost:8000/health
   curl -f http://localhost:8501/_stcore/health
   ```

5. **Stop Stack:**

   ```bash
   docker compose down
   ```

---

## 3. Cloud Production Deployment Architecture

### Deployment Status
>
> **Deployment Status:** Deployment-ready (Containerized architecture with health checks verified locally; cloud secrets required for live SaaS hosting).

### Recommended Cloud Targets

1. **Render / Railway (PaaS):**
   - Deploy `Dockerfile.api` as a Web Service on Port 8000.
   - Deploy `Dockerfile.dashboard` as a Web Service on Port 8501 with environment variable `API_URL=https://<your-api-service>.onrender.com`.
2. **Hugging Face Spaces:**
   - Deploy Streamlit frontend directly via Docker SDK space.

### Environment Variables Checklist for Cloud

- `APP_ENV=production`
- `API_HOST=0.0.0.0`
- `API_PORT=8000`
- `LOG_LEVEL=INFO`
- `RETENTION_OFFER_COST=35.0`
- `CHURN_LOSS_FACTOR=1.0`
- `USD_TO_INR_RATE=83.5`
- `MLFLOW_TRACKING_URI=sqlite:///mlruns.db`
