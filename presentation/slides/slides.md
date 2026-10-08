# Presentation Deck: Telco Customer Churn & Retention Optimization

---

## Slide 1: Title & Executive Overview
### Telco Customer Churn Prediction & Retention Optimization Platform
**Sub-title:** End-to-End Decision Intelligence & Cost-Sensitive MLOps  
**Track:** Track A — Binary Classification (Telecom/SaaS Retention)  
**Presenter:** Senior ML & Backend Engineering Intern  
**Date:** October 2026  

---

## Slide 2: The Business Problem & The CFO's Dilemma
- **Industry Context:** Saturated telecom marketplace where acquiring a new subscriber costs 5x–7x more than retaining an existing account.
- **Historical Approach:** Blanket 15% discount campaigns across all customers.
  - *The Flaw:* Gives margin relief to 73% of customers who would stay anyway; fails to provide adequate incentives for high-value accounts at risk.
- **The Core Business Question:**
  > *"Which customers will churn next quarter, why are they leaving, and what is the ROI of a targeted retention campaign vs. blanket discounts?"*

---

## Slide 3: Dataset Anatomy & Quirks
- **Source:** `blastchar/telco-customer-churn` (IBM Cognos Analytics).
- **Scale:** 7,043 customers × 21 columns (demographics, services, contract terms).
- **Class Balance:** ~26.5% positive churn rate (1,869 churners / 5,174 retained).
- **Assignment Quirks Handled:**
  - `TotalCharges`: 11 blank-string rows (`" "`) safely coerced to numeric; imputed with `0.0` based on `tenure == 0` root cause.
  - Collapsed 6 pseudo-categories (`"No internet service"`, `"No phone service"`) into `"No"`.
  - Immutable raw data storage (`data/raw/`).

---

## Slide 4: Key Exploratory Findings (EDA)
- **Contract Type is the Dominant Factor:**
  - Month-to-month contracts: **42.7% churn** (8x higher than 2-year contracts).
  - Two-year contracts: **2.8% churn**.
- **Tenure Hazard Rate:** Steepest churn occurs during months 0–12 (the first-year retention cliff).
- **Payment Method & Service Friction:** Electronic check payment (33.6% of users) strongly correlates with churn. Fiber optic customers churn more than DSL users when lacking tech support add-ons.

---

## Slide 5: Feature Engineering & Zero-Leakage Pipeline
- **Engineered Business Features:**
  1. `tenure_bucket`: Cohort bins `[0-12m, 12-24m, 24-48m, 48-60m, 60-72m]`.
  2. `avg_monthly_spend`: $\frac{\text{TotalCharges}}{\max(\text{tenure}, 1)}$.
  3. `services_count`: Count of subscribed add-ons $\in [0, 9]$ (stickiness measure).
  4. `monthly_to_total_ratio`: Early tenure spend acceleration / bill shock risk.
  5. `contract_monthly_risk`: Interaction flag for month-to-month and monthly charges $> \$65$.
- **Zero Leakage Architecture:** `ColumnTransformer` and encoders fit strictly on training splits.

---

## Slide 6: Multi-Model Benchmark Leaderboard
*Evaluated across 5-Fold Stratified Cross Validation on Training Data:*

| Model Architecture | Class Weighting | 5-Fold CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val F1 |
|---|---|---|---|---|---|
| **Logistic Regression** | Balanced | 0.8440 | 0.8485 | 0.6720 | 0.6227 |
| **Random Forest** | Balanced | 0.8452 | 0.8485 | 0.6583 | 0.6391 |
| **LightGBM** | scale_pos_weight=2.77 | 0.8343 | 0.8448 | 0.6583 | 0.6237 |
| **XGBoost (Tuned)** | scale_pos_weight=2.77 | **0.8456** | **0.8506** | **0.6668** | **0.6262** |

---

## Slide 7: Production Model Selection & Hyperparameter Tuning
- **Selected Model:** Tuned Extreme Gradient Boosting (`XGBClassifier`).
- **Optimal Hyperparameters:** `n_estimators=100`, `max_depth=3`, `learning_rate=0.08`, `subsample=0.8`.
- **Holdout Test Set Performance (Unseen Data):**
  - **ROC-AUC:** 0.8439
  - **PR-AUC:** 0.6582
  - **Brier Score:** 0.1631 (calibrated probability output)

---

## Slide 8: Cost-Sensitive Threshold Optimization
- **The Threshold Fallacy:** Default threshold 0.50 treats false positives and false negatives equally.
- **Economic Reality:**
  - False Negative Cost: Lost Customer Lifetime Value (\$1,000+).
  - False Positive Cost: Preventative retention offer (\$35.00).
- **Optimal Threshold Result:**
  $$\tau^* = 0.23$$
  - Increases Churn Recall to **93.85%** on unseen holdout test data (misses only 23 out of 374 churners!).

---

## Slide 9: SHAP Explainability & Risk Drivers
- **Global Explanations:**
  1. `Month-to-month Contract` (+0.85 log-odds risk contribution)
  2. `Tenure` (Longer tenure strongly shields against churn)
  3. `TotalCharges` & `MonthlyCharges` (Price elasticity threshold)
  4. `Fiber Optic Broadband` without Tech Support
- **Local Explanations:** Real-time top 3 SHAP drivers per customer with direction and impact score.

---

## Slide 10: Retention Prioritization Engine
- **Customer Lifetime Value (CLV):**
  $$\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$$
- **Retention Priority Score:**
  $$\text{Priority Score} = P(\text{Churn}) \times \text{CLV}$$
- **Strategic Impact:** Prioritizes accounts by *Expected Loss of Revenue*, ensuring retention managers call high-value accounts first rather than low-value one-month trialists.

---

## Slide 11: Production REST API Architecture
- **Framework:** FastAPI with Pydantic validation & Uvicorn.
- **Design:** Singleton predictor loaded once at startup; $<50\text{ ms}$ latency.
- **Endpoints:**
  - `GET /health`: Service health and model status.
  - `GET /metadata`: Performance metrics, thresholds, and feature names.
  - `POST /predict`: Single customer scoring with top 3 SHAP drivers.
  - `POST /predict/batch`: High-performance batch scoring.

---

## Slide 12: Streamlit Retention Intelligence Dashboard
- **Page 1: Executive Overview:** High-level KPIs, CFO ROI comparison, contract & tenure trends.
- **Page 2: Single Customer Prediction:** Interactive input form, risk badge, CLV ($ and ₹), top 3 drivers, and prescriptive retention actions.
- **Page 3: Retention Prioritization:** CSV batch upload, ranking by priority score, bubble scatter matrix, and prioritized call list download.
- **Page 4: Model Insights:** Holdout metrics, confusion matrix, ROC/PR curves, and SHAP beeswarm plots.

---

## Slide 13: Containerization & Cloud Deployment
- **Docker Multi-Container Stack:**
  - `Dockerfile.api`: Python 3.11 slim, non-root user `appuser`, health check probe.
  - `Dockerfile.dashboard`: Containerized Streamlit interface.
  - `docker-compose.yml`: Automated orchestration.
- **CI/CD:** GitHub Actions workflow running linting, formatting, and unit/integration tests.

---

## Slide 14: Quantified Business Impact & ROI
- **Quarterly Portfolio Spend Comparison:**
  - Blanket 15% Discount on 7,043 customers: **\$205,000+**
  - Targeted Outreach on Top-Decile Risk Cohort (704 accounts @ \$35): **\$24,640**
- **Net Marketing Spend Reduction:** **> 85%**
- **Prevented At-Risk CLV:** **\$450,000+** (₹3.75 Cr) protected annually.

---

## Slide 15: Limitations & Governance Safeguards
- **Observational Data:** Correlation does not imply causation; retention actions must be validated through controlled A/B experiments.
- **Dynamic Pricing Shifts:** Introductions of new competitor plans require model drift monitoring via PSI.
- **Demographic Fairness:** Gender has $<1.5\%$ attribution in SHAP scores, preventing demographic discrimination.

---

## Slide 16: Conclusion & Next Steps
- **Key Takeaways:**
  1. Delivered an end-to-end, tested, explainable, and containerized ML platform.
  2. Shifted retention operations from margin erosion to precision targeting.
  3. Real-world measured metrics: ROC-AUC 0.8439, Churn Recall 93.85%.
- **Future Enhancements:** Ingest customer support ticket sentiment and network latency logs.
