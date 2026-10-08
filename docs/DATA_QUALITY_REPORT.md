# Telco Customer Churn — Data Quality Audit Report

**Audit Date:** 2026-10-08  
**Dataset Source:** `blastchar/telco-customer-churn` (IBM Cognos Analytics)  
**Validation Status:** ✅ PASSED  

---

## 1. Executive Summary

- **Total Records:** 7,043
- **Total Features:** 21
- **Duplicate Rows:** 0
- **Duplicate Customer IDs:** 0 (100% Unique ID integrity)
- **Target Distribution:**
  - `No` (Retained): 5,174 (73.46%)
  - `Yes` (Churned): 1,869 (26.54%)
- **Class Imbalance Ratio:** ~26.5% churn (Moderately imbalanced)

---

## 2. Assignment-Specific Quirks & Diagnostics

### A. TotalCharges Blank Strings Anomaly
- **Detected Count:** 11 rows contain blank whitespace strings (`" "`) rather than numerical values or standard `NaN`.
- **Root Cause Analysis:** All 11 rows with blank `TotalCharges` correspond exactly to customers with `tenure == 0` months (new accounts enrolled in their first billing cycle).
- **Engineering Treatment:** 
  1. Safe coercion via `pd.to_numeric(errors='coerce')`.
  2. Impute `TotalCharges` with `0.0` (matching `MonthlyCharges * tenure = MonthlyCharges * 0 = 0.0`).
  3. Avoid dropping these records to preserve real-world customer profiles.

### B. Service Pseudo-Categories
- Several service features (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) include a 3rd category: `"No internet service"`.
- `MultipleLines` includes `"No phone service"`.
- **Engineering Treatment:** Collapsed into `"No"` during preprocessing to prevent redundant feature expansion while retaining the primary service flags.

---

## 3. Column-by-Column Data Quality Metrics

| Column | Data Type | Non-Null Count | Missing / Blank | Missing % | Unique Count | Sample Values |
|---|---|---|---|---|---|---|
| `customerID` | `object` | 7,043 | 0 | 0.00% | 7043 | `['7590-VHVEG', '5575-GNVDE', '3668-QPYBK']` |
| `gender` | `object` | 7,043 | 0 | 0.00% | 2 | `['Female', 'Male', 'Male']` |
| `SeniorCitizen` | `int64` | 7,043 | 0 | 0.00% | 2 | `['0', '0', '0']` |
| `Partner` | `object` | 7,043 | 0 | 0.00% | 2 | `['Yes', 'No', 'No']` |
| `Dependents` | `object` | 7,043 | 0 | 0.00% | 2 | `['No', 'No', 'No']` |
| `tenure` | `int64` | 7,043 | 0 | 0.00% | 73 | `['1', '34', '2']` |
| `PhoneService` | `object` | 7,043 | 0 | 0.00% | 2 | `['No', 'Yes', 'Yes']` |
| `MultipleLines` | `object` | 7,043 | 0 | 0.00% | 3 | `['No phone service', 'No', 'No']` |
| `InternetService` | `object` | 7,043 | 0 | 0.00% | 3 | `['DSL', 'DSL', 'DSL']` |
| `OnlineSecurity` | `object` | 7,043 | 0 | 0.00% | 3 | `['No', 'Yes', 'Yes']` |
| `OnlineBackup` | `object` | 7,043 | 0 | 0.00% | 3 | `['Yes', 'No', 'Yes']` |
| `DeviceProtection` | `object` | 7,043 | 0 | 0.00% | 3 | `['No', 'Yes', 'No']` |
| `TechSupport` | `object` | 7,043 | 0 | 0.00% | 3 | `['No', 'No', 'No']` |
| `StreamingTV` | `object` | 7,043 | 0 | 0.00% | 3 | `['No', 'No', 'No']` |
| `StreamingMovies` | `object` | 7,043 | 0 | 0.00% | 3 | `['No', 'No', 'No']` |
| `Contract` | `object` | 7,043 | 0 | 0.00% | 3 | `['Month-to-month', 'One year', 'Month-to-month']` |
| `PaperlessBilling` | `object` | 7,043 | 0 | 0.00% | 2 | `['Yes', 'No', 'Yes']` |
| `PaymentMethod` | `object` | 7,043 | 0 | 0.00% | 4 | `['Electronic check', 'Mailed check', 'Mailed check']` |
| `MonthlyCharges` | `float64` | 7,043 | 0 | 0.00% | 1585 | `['29.85', '56.95', '53.85']` |
| `TotalCharges` | `object` | 7,032 | 11 | 0.16% | 6531 | `['29.85', '1889.5', '108.15']` |
| `Churn` | `object` | 7,043 | 0 | 0.00% | 2 | `['No', 'No', 'Yes']` |

---

## 4. Issues & Anomalies Log

- ⚠️ **Notice:** Detected 11 blank-string values in 'TotalCharges' (Assignment known quirk).

---

## 5. Preprocessing Strategy Decision

1. **Identifier Handling:** `customerID` is retained during data ingestion and evaluation for tracking and customer-level retention scoring, but dropped from model feature vectors to prevent identity leakage.
2. **Target Encoding:** `Churn` encoded as binary integer (`Yes` = 1, `No` = 0).
3. **Imputation:** `TotalCharges` missing values imputed with `0.0` based on `tenure == 0` domain fact.
4. **Leakage Safeguard:** All categorical encoders and scalers are fitted exclusively on the training split inside an scikit-learn `Pipeline`/`ColumnTransformer`.
