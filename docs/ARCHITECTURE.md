# System Architecture & Technical Design Document

**Project:** Telco Customer Churn Prediction & Retention Optimization Platform  
**Architecture Style:** Modular Microservice & Pipeline Architecture  
**Document Version:** 1.0.0  

---

## 1. High-Level Architectural Diagram

```
                              ┌─────────────────────────────────────────┐
                              │            RAW DATA LAYER               │
                              │  data/raw/WA_Fn-UseC_...csv (Immutable)  │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────┐
                              │        INGESTION & DATA QUALITY         │
                              │  - Safe coercion of TotalCharges blanks │
                              │  - Pseudo-category collapsing ("No")    │
                              │  - Programmatic Schema Validation       │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────┐
                              │      STRATIFIED DATA PARTITIONING       │
                              │  Train (80% dev) | Test (20% holdout)   │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────┐
                              │      FEATURE ENGINEERING & PIPELINE     │
                              │  - tenure_bucket (Binned)               │
                              │  - avg_monthly_spend                    │
                              │  - services_count (Product Stickiness)  │
                              │  - ColumnTransformer (Fit on Train only)│
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────┐
                              │       MODEL BENCHMARKING & TUNING       │
                              │  - Logistic Regression (Baseline)       │
                              │  - Random Forest (Weighted)             │
                              │  - XGBoost (Tuned & scale_pos_weight)   │
                              │  - LightGBM (Gradient Boosting)         │
                              │  - MLflow Experiment Tracking           │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────┐
                              │    COST-SENSITIVE THRESHOLD OPTIMIZER   │
                              │  Optimal Threshold tau* = 0.23          │
                              │  Minimizes Lost CLV vs Campaign Costs   │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────┐
                              │       MODEL REGISTRY & ARTIFACTS        │
                              │  - models/production_pipeline.joblib    │
                              │  - models/metadata.json                 │
                              └─────────┬─────────────────────┬─────────┘
                                        │                     │
                ┌───────────────────────┘                     └───────────────────────┐
                ▼                                                                     ▼
┌───────────────────────────────┐                                     ┌───────────────────────────────┐
│          FASTAPI API          │                                     │      STREAMLIT DASHBOARD      │
│  - GET /health                │                                     │  - Executive Overview         │
│  - GET /metadata              │                                     │  - Single Customer Prediction │
│  - POST /predict              │                                     │  - Retention Prioritization   │
│  - POST /predict/batch        │                                     │  - Model Insights             │
└───────────────┬───────────────┘                                     └───────────────┬───────────────┘
                │                                                                     │
                └───────────────────────────────┬─────────────────────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │       CONTAINERIZATION LAYER    │
                               │  - Dockerfile.api               │
                               │  - Dockerfile.dashboard         │
                               │  - docker-compose.yml           │
                               └─────────────────────────────────┘
```

---

## 2. Component Design & Responsibilities

### 2.1 Core Package (`src/telco_churn/`)

- `data/`: Ingestion, validation, TotalCharges blank coercion, pseudo-category normalization, and stratified partitioning.
- `features/`: Scikit-learn compatible transformer `TelcoFeatureEngineer` and `ColumnTransformer` builder guaranteeing zero leakage between training and inference.
- `models/`: Unified `Pipeline` construction, stratified cross-validation, hyperparameter tuning (`GridSearchCV`), and model serialization (`ModelRegistry`).
- `explainability/`: `TelcoShapExplainer` computing game-theoretic Shapley attributions (TreeExplainer) with business feature mapping.
- `business/`: Pure business logic functions for CLV ($\text{MonthlyCharges} \times \text{tenure}$), Retention Priority Score ($P(\text{Churn}) \times \text{CLV}$), and cost-sensitive threshold optimization.
- `inference/`: `TelcoChurnPredictor` singleton for low-latency batch and single-record predictions.
- `monitoring/`: Population Stability Index (PSI) and Kolmogorov-Smirnov distribution drift monitors.

### 2.2 Backend Service (`api/`)

- Built on FastAPI with Pydantic schema validation.
- Singleton model lifecycle (`lifespan`) to load weights once into memory.
- Standardized REST endpoints: `/health`, `/metadata`, `/predict`, `/predict/batch`.

### 2.3 Analytics Frontend (`dashboard/`)

- Streamlit application tailored for business executives and frontline retention teams.
- Modular architecture with clean component separation (`components/`, `views/`) and a warm light enterprise visual system (`#F5F0E7` ivory background, `#FFFDF8` cream surfaces, `#2D2924` espresso typography, `#5E6B4A` olive brand, `#A56B4F` terracotta accents).
- Strictly single entrypoint routing via `dashboard/app.py` with zero competing page discovery routes or dark theme toggles.

---

## 3. Design Decisions & Rationale

1. **Why XGBoost with `scale_pos_weight=2.77`?**  
   XGBoost achieved the highest validation ROC-AUC (0.8506) and PR-AUC (0.6582) while supporting native exact TreeExplainer SHAP calculations in $<5\text{ ms}$.
2. **Why Cost-Sensitive Thresholding ($\tau^* = 0.23$) instead of 0.50?**  
   Missing a high-CLV customer costs the business thousands of dollars in lost lifetime value, whereas a preventative outreach offer costs only \$35. Lowering the threshold to 0.23 increases churner recall to **93.85%** on unseen holdout test data.
3. **Zero Data Leakage Pipeline:**  
   By embedding `TelcoFeatureEngineer` and `ColumnTransformer` into a single `Pipeline`, transformations during inference precisely mirror those fit during training.
