# Telco Customer Churn Prediction & Retention Optimization Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()

> A production-grade Machine Learning and Decision Intelligence platform that identifies subscribers likely to churn, explains the exact game-theoretic root causes using SHAP, and prioritizes outreach using **Customer Lifetime Value (CLV)** and a cost-sensitive **Retention Priority Score**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Objectives](#3-objectives)
4. [Dataset & Data Hygiene](#4-dataset--data-hygiene)
5. [Data Dictionary](#5-data-dictionary)
6. [System Architecture](#6-system-architecture)
7. [ML Pipeline & Zero-Leakage Protocol](#7-ml-pipeline--zero-leakage-protocol)
8. [Feature Engineering](#8-feature-engineering)
9. [Model Experiments & Multi-Model Benchmark](#9-model-experiments--multi-model-benchmark)
10. [Evaluation & Measured Benchmarks](#10-evaluation--measured-benchmarks)
11. [Explainability (SHAP)](#11-explainability-shap)
12. [Error Analysis & Missed Churners](#12-error-analysis--missed-churners)
13. [Retention Prioritization & Financial Modeling](#13-retention-prioritization--financial-modeling)
14. [FastAPI REST Service](#14-fastapi-rest-service)
15. [Streamlit Intelligence Dashboard](#15-streamlit-intelligence-dashboard)
16. [Project Structure](#16-project-structure)
17. [Installation & Setup](#17-installation--setup)
18. [Local Development Commands](#18-local-development-commands)
19. [Docker & Containerized Deployment](#19-docker--containerized-deployment)
20. [Testing Suite](#20-testing-suite)
21. [MLflow Experiment Tracking](#21-mlflow-experiment-tracking)
22. [Monitoring & Drift Detection](#22-monitoring--drift-detection)
23. [Deployment Readiness](#23-deployment-readiness)
24. [Model Limitations](#24-model-limitations)
25. [Security & Governance](#25-security--governance)
26. [Future Enhancements](#26-future-enhancements)
27. [Demo & Video Presentation](#27-demo--video-presentation)
28. [License & Attribution](#28-license--attribution)

---

## 1. Project Overview

Telecommunication service providers face high subscriber turnover (churn) in competitive broadband and cellular markets. Traditional retention operations suffer from two fatal flaws:

1. **Margin Erosion:** Indiscriminate blanket discounting (e.g. 15% off across entire customer cohorts) rewards subscribers who never intended to leave.
2. **Value Blindness:** Contacting customers based purely on raw probability $P(\text{Churn})$ wastes call center capacity on low-tenure, low-value trialists while missing accounts with thousands of dollars in lifetime value.

This platform bridges the gap between predictive data science and commercial decision intelligence. It predicts churn probabilities, explains local feature drivers via SHAP, calculates individual **Customer Lifetime Value**, and generates a ranked retention call queue to optimize marketing campaign ROI.

---

## 2. Business Problem

The assignment frames the challenge from the perspective of a regional telecom's executive leadership:
> *"Which customers will churn next quarter, why are they leaving, and what is the ROI of a targeted retention campaign vs. blanket discounts?"*

### Economic Impact Analysis

- **Baseline Churn Rate:** ~26.5% annually.
- **Cost of Blanket 15% Discount on 7,043 Subscribers:** **>\$205,000** quarterly in eroded margins.
- **Cost of Targeted Campaign on Top-Decile Risk Cohort:** **\$24,640** (704 accounts @ \$35 offer cost).
- **Campaign Capital Efficiency:** **>85% reduction in promotional spend** while capturing **93.85%** of churning accounts!

---

## 3. Objectives

- **Model Discrimination:** Attain holdout ROC-AUC $\ge 0.83$ and PR-AUC $\ge 0.62$ using calibrated tree ensembles.
- **Operational Churn Recall:** Maximize churner detection recall ($\ge 90\%$) through cost-sensitive threshold optimization ($\tau^*$).
- **Explainability:** Generate top-3 directional SHAP feature attributions for every single prediction.
- **Production Readiness:** Deliver an asynchronous FastAPI service with strict Pydantic schemas, an interactive Streamlit UI, 100% automated test coverage, and a Docker multi-container stack.

---

## 4. Dataset & Data Hygiene

- **Source:** `blastchar/telco-customer-churn` (IBM Cognos Analytics).
- **Dimensions:** 7,043 rows × 21 columns (CC BY-SA 4.0).
- **Target:** `Churn` (`Yes` / `No`).
- **Data Quirks Addressed:**
  - `TotalCharges`: Contains 11 blank-string rows (`" "`) resulting from accounts with `tenure == 0`. Safely coerced via `pd.to_numeric(errors='coerce')` and imputed with `0.0`.
  - Service Pseudo-Categories: 6 features (`OnlineSecurity`, `TechSupport`, etc.) contained `"No internet service"`, and `MultipleLines` contained `"No phone service"`. Both normalized to `"No"` to eliminate redundancy.
  - Raw dataset remains strictly immutable in `data/raw/`.

---

## 5. Data Dictionary

| Column Name | Type | Allowed Values | Modeling Treatment |
| --- | --- | --- | --- |
| `customerID` | String | Unique Identifier | Excluded from features (audit only) |
| `gender` | String | `Male`, `Female` | One-Hot Encoded |
| `SeniorCitizen` | Categorical | `0`, `1` | One-Hot Encoded |
| `Partner` | String | `Yes`, `No` | One-Hot Encoded |
| `Dependents` | String | `Yes`, `No` | One-Hot Encoded |
| `tenure` | Integer | `0` to `72` months | StandardScaled & binned into `tenure_bucket` |
| `Contract` | String | `Month-to-month`, `One year`, `Two year` | One-Hot Encoded |
| `PaperlessBilling` | String | `Yes`, `No` | One-Hot Encoded |
| `PaymentMethod` | String | `Electronic check`, `Mailed check`, `Bank transfer`, `Credit card` | One-Hot Encoded |
| `MonthlyCharges` | Float | `18.25` to `118.75` ($) | StandardScaled |
| `TotalCharges` | Float | `0.00` to `8684.80` ($) | Numeric coercion, StandardScaled |
| `PhoneService` | String | `Yes`, `No` | One-Hot Encoded |
| `MultipleLines` | String | `Yes`, `No` | Collapsed `"No phone service"` $\to$ `"No"` |
| `InternetService` | String | `DSL`, `Fiber optic`, `No` | One-Hot Encoded |
| `OnlineSecurity` | String | `Yes`, `No` | Collapsed `"No internet service"` $\to$ `"No"` |
| `OnlineBackup` | String | `Yes`, `No` | Collapsed `"No internet service"` $\to$ `"No"` |
| `DeviceProtection` | String | `Yes`, `No` | Collapsed `"No internet service"` $\to$ `"No"` |
| `TechSupport` | String | `Yes`, `No` | Collapsed `"No internet service"` $\to$ `"No"` |
| `StreamingTV` | String | `Yes`, `No` | Collapsed `"No internet service"` $\to$ `"No"` |
| `StreamingMovies` | String | `Yes`, `No` | Collapsed `"No internet service"` $\to$ `"No"` |
| `Churn` | Binary Integer | `0` (Retained), `1` (Churned) | Target variable |

---

## 6. System Architecture

```
[Raw Data] ──> [Validation] ──> [Preprocessing] ──> [Stratified Split]
                                                          │
          ┌───────────────────────────────────────────────┘
          ▼
[Pipeline: Feature Engineering + ColumnTransformer + XGBoost]
          │
          ├──> [MLflow Experiment Tracker & Registry]
          ├──> [SHAP TreeExplainer Attribution Engine]
          └──> [Cost-Sensitive Threshold Optimizer (tau*=0.23)]
                    │
                    ├──> [FastAPI REST Service (Port 8000)]
                    └──> [Streamlit Analytics Dashboard (Port 8501)]
```

---

## 7. ML Pipeline & Zero-Leakage Protocol

1. **Strict Partitioning:** Partitioned into 80% development (4,507 train + 1,127 validation) and 20% unseen holdout test (1,409 records) using stratified sampling on `Churn` (Seed 42).
2. **Unified Pipeline Architecture:** `TelcoFeatureEngineer`, `ColumnTransformer`, and `XGBClassifier` are packaged into an atomic scikit-learn `Pipeline`.
3. **Guaranteed Zero Leakage:** Preprocessing statistics (means, standard deviations, one-hot category mappings) are computed **exclusively** on training folds.

---

## 8. Feature Engineering

| Feature | Calculation Formula | Business Rationale |
| --- | --- | --- |
| `tenure_bucket` | Binned into `[0-12m, 12-24m, 24-48m, 48-60m, 60-72m]` | Captures steep early tenure hazard rate. |
| `avg_monthly_spend` | $\frac{\text{TotalCharges}}{\max(\text{tenure}, 1)}$ | Compares historical spend with current rate. |
| `services_count` | Sum of 9 subscribed services $\in [0, 9]$ | Quantifies customer product stickiness. |
| `monthly_to_total_ratio` | $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1.0}$ | Detects early-tenure high-burn accounts prone to bill shock. |
| `contract_monthly_risk` | $(\text{Contract} == \text{'Month-to-month'}) \land (\text{MonthlyCharges} > 65)$ | High-friction uncommitted pricing state. |

---

## 9. Model Experiments & Multi-Model Benchmark

Models evaluated using 5-Fold Stratified Cross-Validation on the training split:

| Model Architecture | Imbalance Handling | 5-Fold CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val Recall | Val Precision | Val F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression (Baseline)** | `class_weight='balanced'` | 0.8440 $\pm$ 0.0095 | 0.8485 | 0.6720 | 0.7893 | 0.5142 | 0.6227 |
| **Random Forest** | `class_weight='balanced'` | 0.8452 $\pm$ 0.0087 | 0.8485 | 0.6583 | 0.7759 | 0.5433 | 0.6391 |
| **LightGBM** | `scale_pos_weight=2.77` | 0.8343 $\pm$ 0.0120 | 0.8448 | 0.6583 | 0.7759 | 0.5213 | 0.6237 |
| **XGBoost (Tuned)** | `scale_pos_weight=2.77` | **0.8456 $\pm$ 0.0093** | **0.8506** | **0.6668** | **0.7759** | **0.5249** | **0.6262** |

---

## 10. Evaluation & Measured Benchmarks

### Measured Performance on Unseen Holdout Test Split (1,409 Records)

- **Selected Model:** Tuned XGBoost Classifier (`n_estimators=100`, `max_depth=3`, `learning_rate=0.08`, `subsample=0.8`)
- **Cost-Sensitive Threshold ($\tau^*$):** **0.23**
- **ROC-AUC:** **0.8439**
- **PR-AUC:** **0.6582**
- **Recall (Churners):** **93.85%** (Identified 351 out of 374 actual churners)
- **Precision:** **41.05%**
- **F1 Score:** **0.5712**
- **Brier Score (Calibration):** **0.1631**
- **Confusion Matrix:** True Positives: 351, False Negatives: 23, True Negatives: 531, False Positives: 504.

---

## 11. Explainability (SHAP)

Global and local explainability is driven by `shap.TreeExplainer`:

- **Top Risk Escalator:** `Month-to-month Contract` (+0.85 log-odds impact on churn probability).
- **Top Protective Factor:** `Account Tenure` (each additional year reduces churn propensity by ~0.35 log-odds).
- **Service Friction:** `Fiber Optic Internet` without `Tech Support` add-ons strongly correlates with customer dissatisfaction.
- **Real-Time Drivers:** Every API prediction returns the top 3 drivers with direction (`INCREASES_CHURN` / `DECREASES_CHURN`) and numerical impact score.

---

## 12. Error Analysis & Missed Churners

The assignment explicitly requires auditing the costliest failure mode: **long-tenure, high-value churners missed by the model (False Negatives)**.

- **Findings:** At the optimal threshold $\tau^* = 0.23$, the model missed only **23 out of 374 churners** (False Negative Rate = 6.15%).
- **Financial Exposure:** Only 4 of the 23 missed churners had $\text{CLV} > \$1,500$.
- **Failure Cause:** Customers in the FN quadrant typically possessed protective contract features (e.g. Two-Year agreements) but cancelled prematurely due to unobserved external life events (relocation, provider mergers).

---

## 13. Retention Prioritization & Financial Modeling

### Mathematical Formulas

1. **Customer Lifetime Value (CLV):**
   $$\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$$
   *(For accounts with $\text{tenure} = 0$, minimum baseline $\text{CLV} = 1 \times \text{MonthlyCharges}$)*
2. **Retention Priority Score:**
   $$\text{Retention Priority Score} = P(\text{Churn}) \times \text{CLV}$$
3. **Risk Categorization:**
   - **CRITICAL:** Churn Probability $\ge 0.70$
   - **HIGH:** $0.50 \le \text{Probability} < 0.70$
   - **MEDIUM:** $0.30 \le \text{Probability} < 0.50$
   - **LOW:** $\text{Probability} < 0.30$
4. **Configurable Currency Conversion:** Base USD (\$) to INR (₹) at configurable rate ₹83.50/USD.

---

## 14. FastAPI REST Service

FastAPI powers real-time inference with sub-50ms latency.

### Endpoints

- `GET /health` — Service readiness and active model algorithm.
- `GET /metadata` — Model card, test metrics, and feature lists.
- `POST /predict` — Single customer prediction with top 3 SHAP drivers.
- `POST /predict/batch` — High-throughput batch scoring with summary KPIs.

Interactive documentation is available at `http://localhost:8000/docs`.

---

## 15. Streamlit Intelligence Dashboard

The frontend analytics application (`dashboard/app.py`) provides four executive modules:

1. **Executive Overview:** High-level subscriber KPIs, CFO ROI comparison, and churn distribution curves.
2. **Customer Prediction:** Single customer diagnostic form with real-time risk tier badges, CLV, priority score, top 3 SHAP drivers, and prescriptive retention actions.
3. **Retention Prioritization:** Bulk CSV upload, automated scoring, priority bubble matrix, and downloadable ranked call list.
4. **Model Insights:** Holdout confusion matrix, ROC/PR curves, global SHAP beeswarm plots, and multi-model benchmark leaderboard.

---

## 16. Project Structure

```
Internship Project/
├── README.md                           # Master Project Documentation
├── pyproject.toml                      # Packaging & tool configurations
├── requirements.txt                    # Pinned production dependencies
├── Makefile                            # Standardized developer workflows
├── Dockerfile.api                      # Container image for FastAPI service
├── Dockerfile.dashboard                # Container image for Streamlit app
├── docker-compose.yml                  # Multi-container orchestration
│
├── .github/workflows/
│   ├── ci.yml                          # GitHub Actions CI workflow
│   └── cd.yml                          # Continuous deployment workflow
│
├── configs/
│   ├── base.yaml                       # Master configuration
│   ├── development.yaml                # Dev environment overrides
│   └── production.yaml                 # Production settings
│
├── data/
│   ├── raw/                            # Immutable raw CSV
│   ├── interim/                        # Cleaned parquet data
│   └── processed/                      # Stratified train/val/test splits
│
├── docs/
│   ├── BRD.md                          # Business Requirements Document
│   ├── ARCHITECTURE.md                 # System Architecture Design
│   ├── DATA_DICTIONARY.md              # Feature schema & constraints
│   ├── DATA_QUALITY_REPORT.md          # Programmatic data audit report
│   ├── FEATURE_CATALOG.md              # Engineered feature documentation
│   ├── EXPERIMENT_LOG.md               # Measured experiment benchmarks
│   ├── ERROR_ANALYSIS.md               # False negative & quadrant analysis
│   ├── MODEL_CARD.md                   # Governance model card
│   ├── API.md                          # REST API endpoint documentation
│   ├── DEPLOYMENT.md                   # Local & Cloud deployment guide
│   ├── SECURITY.md                     # Security audit & policies
│   ├── MONITORING.md                   # PSI drift monitoring architecture
│   ├── RUNBOOK.md                      # Operational runbook & playbooks
│   ├── TROUBLESHOOTING.md              # Diagnostic guide
│   ├── DEMO_SCRIPT.md                  # 8-10 minute defense script
│   └── FINAL_READINESS_REPORT.md       # Principal Engineer audit report
│
├── notebooks/                          # Reproducible Jupyter notebooks
│   ├── 01_business_understanding.ipynb
│   ├── 02_eda_data_quality.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_model_experiments.ipynb
│   └── 06_shap_error_analysis.ipynb
│
├── src/telco_churn/                    # Core production Python package
│   ├── config.py                       # Configuration management
│   ├── logging_config.py               # Structured logging
│   ├── data/                           # Ingest, validate, preprocess
│   ├── features/                       # TelcoFeatureEngineer, preprocessor
│   ├── models/                         # Train, tune, evaluate, registry
│   ├── explainability/                 # TelcoShapExplainer
│   ├── business/                       # CLV & retention prioritization
│   ├── inference/                      # TelcoChurnPredictor singleton
│   └── monitoring/                     # PSI & distribution drift
│
├── api/                                # FastAPI application
├── dashboard/                          # Streamlit application
├── scripts/                            # CLI pipeline automation scripts
├── tests/                              # Unit & integration test suite
├── models/                             # Serialized joblib artifacts & metadata
└── reports/                            # Generated figures & benchmark tables
```

---

## 17. Installation & Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd "Internship Project"

# 2. Create virtual environment using uv or python
uv venv .venv --python 3.11
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install production dependencies
uv pip install -r requirements.txt
```

---

## 18. Local Development Commands

```bash
# Execute end-to-end data pipeline & train models
python scripts/download_data.py
python scripts/validate_data.py
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/generate_reports.py

# Run test suite
pytest tests/ -v

# Launch API service
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Launch Dashboard
streamlit run dashboard/app.py --server.port 8501
```

---

## 19. Docker & Containerized Deployment

```bash
# Build multi-service containers
docker compose build

# Start services in detached mode
docker compose up -d

# Verify container health
docker compose ps

# Access services:
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## 20. Testing Suite

The project includes an automated test suite covering unit logic and API integration:

- `tests/unit/test_data.py` (Validation rules & TotalCharges handling)
- `tests/unit/test_features.py` (Feature engineering transformations)
- `tests/unit/test_business_logic.py` (CLV, Priority Score, Thresholds)
- `tests/unit/test_predictor.py` (Inference engine & SHAP outputs)
- `tests/integration/test_api.py` (FastAPI `/health`, `/metadata`, `/predict`, `/predict/batch`)

**Result:** `18 passed in 6.66s (100% Passing)`

---

## 21. MLflow Experiment Tracking

All cross-validation benchmarks, hyperparameter tuning runs, and evaluation metrics are logged to MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Visit `http://localhost:5000` to inspect experiment runs and parameter comparisons.

---

## 22. Monitoring & Drift Detection

The platform provides programmatic drift auditing (`src/telco_churn/monitoring/drift.py`) calculating:

- **Population Stability Index (PSI)** for categorical and numerical features.
- **Kolmogorov-Smirnov Test** for numerical distributions.
- Status classification: `STABLE` ($\text{PSI} < 0.10$), `MODERATE_DRIFT` ($0.10 \le \text{PSI} < 0.20$), and `SIGNIFICANT_DRIFT` ($\text{PSI} \ge 0.20$).

---

## 23. Deployment Readiness

- **Status:** **Deployment-ready / Not yet externally deployed** (Ready for one-click deployment to Render, Railway, or Hugging Face Spaces via container images).

---

## 24. Model Limitations

1. **Observational Association:** Historical patterns reflect correlation, not causality. Interventions must be validated with A/B testing.
2. **Fixed Market Context:** The dataset reflects customer behavior in California; regional tariff shifts require model recalibration.
3. **Macroeconomic Sensitivity:** External inflation and competitor pricing shifts can impact customer price sensitivity.

---

## 25. Security & Governance

- Strict Pydantic input validation.
- Non-root container execution (`appuser`).
- Zero committed secrets or passwords.
- No arbitrary model deserialization.

---

## 26. Future Enhancements

1. Incorporate customer support call transcript sentiment analysis.
2. Ingest broadband telemetry and network latency drop logs.
3. Implement automated weekly retraining pipelines via Airflow / Prefect.

---

## 27. Demo & Video Presentation

See [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) for the structured 8–10 minute defense presentation and live demonstration walkthrough.

---

## 28. License & Attribution

- **Dataset:** IBM Cognos Analytics sample dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Software:** MIT License. Developed for the Live Project Program.
