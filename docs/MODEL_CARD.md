# Model Card: Telco Customer Churn & Retention Optimization

**Model ID:** `telco-churn-retention-model`  
**Model Version:** `v1.0.0`  
**Algorithm:** `Tuned XGBoost Classifier (scale_pos_weight=2.77)`  
**Release Date:** 2026-10-08T07:41:53.723860+00:00  
**Lead Engineer:** Senior MLOps & Machine Learning Engineering Team  

---

## 1. Intended Use & Scope
- **Intended Purpose:** Predict individual customer churn probability $P(\text{Churn}=1)$ and calculate retention prioritization scores for telecom and SaaS subscription accounts.
- **Intended Users:** Retention Marketing Managers, Customer Success Operations, Account Executives, and automated CRM notification webhooks.
- **Out-of-Scope Uses:**
  - Automated credit scoring or financial loan decisioning.
  - Punitive customer treatment, cancellation throttling, or service degradation.
  - Evaluation of enterprise B2B bespoke multi-million corporate contracts without recalibration.

---

## 2. Training Data & Dataset Profile
- **Source:** `blastchar/telco-customer-churn` (IBM Cognos Analytics).
- **Cohort:** 7,043 customer records in California.
- **Dataset Partition:**
  - Training Split: 4,507 rows (80% development subset)
  - Validation Split: 1,127 rows (tuning & threshold selection)
  - Holdout Test Split: 1,409 rows (unseen final evaluation)
  - Split Strategy: Stratified by Churn (Seed 42)
  - Historical Baseline Churn Rate: ~26.5%

---

## 3. Model Architecture & Pipeline
1. **Feature Engineering Stage:** Custom `TelcoFeatureEngineer` transformer creating `tenure_bucket`, `avg_monthly_spend`, `services_count`, `monthly_to_total_ratio`, and `contract_monthly_risk`.
2. **Preprocessing Stage:** `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown='ignore')` for categorical features.
3. **Classifier:** Extreme Gradient Boosting (`XGBClassifier`) with tuned tree depth ($3$), learning rate ($0.08$), subsample ($0.8$), and positive class re-weighting (`scale_pos_weight=2.77`).
4. **Decision Boundary:** Cost-sensitive threshold $\tau^* = 0.23$ balancing false negatives (lost customer CLV) against false positives (retention offer costs).

---

## 4. Performance Metrics (Holdout Test Set)

| Metric | Measured Score | Business Interpretation |
|---|---|---|
| **ROC-AUC** | `0.8439` | Strong ranking discrimination across thresholds |
| **PR-AUC** | `0.6582` | High precision-recall area under moderate class imbalance |
| **Recall (Churners)** | `0.9385` (93.85%) | Successfully identifies ~94% of churning customers |
| **Precision** | `0.4105` | Positive predictive value at the optimal retention threshold |
| **F1 Score** | `0.5712` | Harmonic balance of precision and recall |
| **Brier Score** | `0.1631` | Probability calibration accuracy |

---

## 5. Explainability & Governance
- **Local Explanations:** Real-time exact SHAP attribution scores calculated via `shap.TreeExplainer` returning top 3 primary drivers and direction of effect per prediction.
- **Global Feature Ranking:** Primary churn drivers identified:
  1. Month-to-month Contract (strongest risk escalator)
  2. Account Tenure (strongest protective barrier against churn)
  3. Monthly Charges & Total Charges
  4. Fiber Optic Internet Service & Lack of Tech Support

---

## 6. Fairness & Ethical Considerations
- **Demographic Parity:** Model inputs include `gender`, `SeniorCitizen`, `Partner`, and `Dependents`. Audits demonstrate minimal predictive reliance on `gender` (SHAP contribution $< 1.5\%$ of total attribution), preventing discriminatory treatment.
- **Senior Citizen Review:** Older adults with fixed incomes show slight sensitivity to higher fiber optic charges; marketing recommendations emphasize customer support onboarding rather than aggressive upsells.

---

## 7. Retraining & Monitoring Protocol
- **Trigger Conditions:**
  - Population Stability Index (PSI) $> 0.20$ on primary features (`Contract`, `tenure`, `MonthlyCharges`).
  - Observed quarterly churn prediction drift $> 15\%$.
  - Service portfolio updates (introduction of new broadband tiers or streaming bundles).
