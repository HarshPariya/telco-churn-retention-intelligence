# Development Plan & Requirement Traceability Matrix

**Project:** Telco Customer Churn Prediction & Retention Optimization Platform  
**Track:** Track A — Binary Classification  
**Dataset:** `blastchar/telco-customer-churn` (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
**Standard:** Production-Grade End-to-End MLOps  

---

## 1. Master Requirement Traceability Matrix

| Ref # | Assignment Requirement | Implementation Component | File / Artifact Location | Verification / Test | Status |
| --- | --- | --- | --- | --- | --- |
| **REQ-01** | Track A Telco Customer Churn Selection | Binary classification for telecom churn | `configs/base.yaml`, `src/telco_churn/` | Architecture verification | **IN PROGRESS** |
| **REQ-02** | Raw Data Acquisition (7,043 x 21) | Ingestion pipeline & validation | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`, `scripts/download_data.py` | Line count (7,044) & checksum | **COMPLETED** |
| **REQ-03** | TotalCharges blank-string quirk fix | Numeric coercion with audit | `src/telco_churn/data/preprocess.py` | `tests/unit/test_data.py` | **PLANNED** |
| **REQ-04** | "No internet/phone service" pseudo-categories | Service feature category collapsing | `src/telco_churn/data/preprocess.py` | `tests/unit/test_data.py` | **PLANNED** |
| **REQ-05** | BRD tying Churn to CLV & Targeted Retention | Business Requirements Document | `docs/BRD.md` | Stakeholder review | **PLANNED** |
| **REQ-06** | Data Quality Report on actual dataset | Comprehensive quality audit report | `docs/DATA_QUALITY_REPORT.md` | `scripts/validate_data.py` | **PLANNED** |
| **REQ-07** | Reproducible EDA on actual Telco data | Jupyter notebook & visualizations | `notebooks/02_eda_data_quality.ipynb`, `reports/figures/` | Saved cell outputs & plots | **PLANNED** |
| **REQ-08** | Feature Engineering (`tenure_bucket`, `avg_monthly_spend`, `services_count`) | Modular feature pipeline without leakage | `src/telco_churn/features/build.py` | `tests/unit/test_features.py` | **PLANNED** |
| **REQ-09** | Feature Catalog | Comprehensive feature metadata | `docs/FEATURE_CATALOG.md` | Documentation audit | **PLANNED** |
| **REQ-10** | Stratified Train/Val/Test Split | No leakage, fixed seed splitting | `src/telco_churn/data/preprocess.py` | `tests/unit/test_data.py` | **PLANNED** |
| **REQ-11** | Logistic Regression Baseline | Baseline pipeline with F1, ROC-AUC, PR-AUC | `src/telco_churn/models/train.py` | `tests/unit/test_models.py` | **PLANNED** |
| **REQ-12** | Model Experimentation (RF, XGBoost, LightGBM) | Systematic multi-model comparison | `src/telco_churn/models/train.py`, `scripts/train_model.py` | MLflow experiment tracking | **PLANNED** |
| **REQ-13** | Class Imbalance Handling (~26.5% churn) | Class weighting vs SMOTE evaluation | `src/telco_churn/models/train.py` | Cross-validation metric comparison | **PLANNED** |
| **REQ-14** | Hyperparameter Tuning | Stratified CV parameter search | `src/telco_churn/models/tune.py` | Cross-validation results | **PLANNED** |
| **REQ-15** | Experiment Log | Complete history of runs and metrics | `docs/EXPERIMENT_LOG.md` | MLflow run matching | **PLANNED** |
| **REQ-16** | Cost-Sensitive Threshold Optimization | False Negative (CLV) vs False Positive (Offer Cost) | `src/telco_churn/business/prioritization.py` | `tests/unit/test_business_logic.py` | **PLANNED** |
| **REQ-17** | SHAP Explainability (Global + Local Top 3 Drivers) | TreeExplainer / LinearExplainer with feature mapping | `src/telco_churn/explainability/shap_explainer.py` | `tests/unit/test_predictor.py` | **PLANNED** |
| **REQ-18** | Error Analysis (High-CLV False Negatives) | Deep segment error review | `docs/ERROR_ANALYSIS.md` | Confusion matrix segments | **PLANNED** |
| **REQ-19** | Model Card | Comprehensive model metadata card | `docs/MODEL_CARD.md` | Documentation review | **PLANNED** |
| **REQ-20** | CLV Calculation (`MonthlyCharges * tenure`) | Reusable business calculation | `src/telco_churn/business/prioritization.py` | `tests/unit/test_business_logic.py` | **PLANNED** |
| **REQ-21** | Retention Priority Score (`churn_probability * CLV`) | Configurable risk bands & scoring | `src/telco_churn/business/prioritization.py` | `tests/unit/test_business_logic.py` | **PLANNED** |
| **REQ-22** | Inference Predictor Service | Unified predictor for API and dashboard | `src/telco_churn/inference/predictor.py` | `tests/unit/test_predictor.py` | **PLANNED** |
| **REQ-23** | FastAPI Endpoints (`/health`, `/metadata`, `/predict`, `/predict/batch`) | Production REST API with Pydantic validation | `api/main.py`, `api/schemas.py` | `tests/integration/test_api.py` | **PLANNED** |
| **REQ-24** | Streamlit Business Dashboard | Executive Overview, Predictor, Retention Prioritization, Insights | `dashboard/app.py` | Local browser verification | **PLANNED** |
| **REQ-25** | Monitoring & Drift Detection | Population Stability Index & Data Quality | `src/telco_churn/monitoring/drift.py` | `tests/unit/test_drift.py` | **PLANNED** |
| **REQ-26** | Automated Test Suite | Unit, integration, and edge-case tests | `tests/unit/`, `tests/integration/` | `pytest --cov=src` passing | **PLANNED** |
| **REQ-27** | Docker & Docker Compose | Containerized API, Dashboard, and MLflow | `Dockerfile.api`, `Dockerfile.dashboard`, `docker-compose.yml` | Container build and startup test | **PLANNED** |
| **REQ-28** | CI/CD Workflows | GitHub Actions CI and CD configs | `.github/workflows/ci.yml`, `cd.yml` | Lint + Test workflow run | **PLANNED** |
| **REQ-29** | Complete Documentation & Runbooks | BRD, Architecture, Runbook, Demo Script, etc. | `docs/*.md`, `README.md` | Comprehensive review | **PLANNED** |
| **REQ-30** | Final Engineering Readiness Audit | Principal Engineer audit report (PASS/PARTIAL/FAIL) | `docs/FINAL_READINESS_REPORT.md` | Verification of all criteria | **PLANNED** |

---

## 2. Implementation Phases

- **Phase 1:** Assignment Analysis & Development Plan (Current)
- **Phase 2:** Repository Foundation & Configuration
- **Phase 3:** Data Ingestion, Cleaning & Validation Pipeline
- **Phase 4:** Exploratory Data Analysis & Visualizations
- **Phase 5:** Feature Engineering & Leakage Prevention Pipeline
- **Phase 6:** Baseline Model (Logistic Regression) & Evaluation Benchmarks
- **Phase 7:** Model Experimentation (RF, XGBoost, LightGBM) + Tuning + MLflow
- **Phase 8:** SHAP Explainability & Cost-Sensitive Threshold Optimization
- **Phase 9:** Business Prioritization & CLV Engine
- **Phase 10:** Inference Predictor Engine & Model Artifact Persistence
- **Phase 11:** FastAPI REST Service
- **Phase 12:** Streamlit Business Analytics & Retention Dashboard
- **Phase 13:** Automated Test Suite (Unit & Integration)
- **Phase 14:** Dockerization & Docker Compose Verification
- **Phase 15:** Monitoring, Security, Runbooks, Demo Script & Final Documentation
- **Phase 16:** Principal Engineer Readiness Audit & Verification
