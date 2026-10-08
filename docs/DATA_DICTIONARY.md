# Data Dictionary

## Telco Customer Churn Feature Specifications

**Source:** `blastchar/telco-customer-churn` (IBM Cognos Analytics)  
**Total Records:** 7,043  
**Target Variable:** `Churn` (Yes / No)  

---

## 1. Demographic Features

| Column Name | Type | Allowed Values | Missing Values | Description | Modeling Role | Preprocessing Treatment |
| --- | --- | --- | --- | --- | --- | --- |
| `customerID` | String | Format: `####-AAAAA` | 0 (0.0%) | Unique customer account identifier | ID / Audit | Dropped from model feature matrix |
| `gender` | String | `Male`, `Female` | 0 (0.0%) | Biological sex / gender identity | Demographic | One-hot encoded |
| `SeniorCitizen` | Integer | `0`, `1` | 0 (0.0%) | Indicator whether subscriber is $\ge 65$ years | Demographic | One-hot encoded |
| `Partner` | String | `Yes`, `No` | 0 (0.0%) | Whether customer has a domestic partner | Demographic | One-hot encoded |
| `Dependents` | String | `Yes`, `No` | 0 (0.0%) | Whether customer lives with children/dependents | Demographic | One-hot encoded |

---

## 2. Account & Contract Features

| Column Name | Type | Allowed Values | Missing Values | Description | Modeling Role | Preprocessing Treatment |
| --- | --- | --- | --- | --- | --- | --- |
| `tenure` | Integer | `0` to `72` months | 0 (0.0%) | Duration in months customer has stayed with telco | Core Account | StandardScaled & binned into `tenure_bucket` |
| `Contract` | String | `Month-to-month`, `One year`, `Two year` | 0 (0.0%) | Current subscription contract duration | Key Predictor | One-hot encoded |
| `PaperlessBilling` | String | `Yes`, `No` | 0 (0.0%) | Customer receives electronic bills | Account Billing | One-hot encoded |
| `PaymentMethod` | String | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | 0 (0.0%) | Payment channel | Account Billing | One-hot encoded |
| `MonthlyCharges` | Float | `18.25` to `118.75` ($) | 0 (0.0%) | Current recurring monthly charge | Financial | StandardScaled |
| `TotalCharges` | Float | `0.00` to `8684.80` ($) | 11 blanks (0.16%) | Cumulative lifetime charges billed to customer | Financial | Coerced to numeric; 0.0 for tenure 0; StandardScaled |

---

## 3. Subscribed Services

| Column Name | Type | Raw Allowed Values | Modeling Treatment | Description |
| --- | --- | --- | --- | --- |
| `PhoneService` | String | `Yes`, `No` | One-hot encoded | Landline telephone service subscription |
| `MultipleLines` | String | `Yes`, `No`, `No phone service` | Collapsed `"No phone service"` $\to$ `"No"` | Multiple landline telephone lines |
| `InternetService` | String | `DSL`, `Fiber optic`, `No` | One-hot encoded | Core broadband internet delivery mechanism |
| `OnlineSecurity` | String | `Yes`, `No`, `No internet service` | Collapsed `"No internet service"` $\to$ `"No"` | Anti-malware and security add-on |
| `OnlineBackup` | String | `Yes`, `No`, `No internet service` | Collapsed `"No internet service"` $\to$ `"No"` | Cloud storage backup add-on |
| `DeviceProtection` | String | `Yes`, `No`, `No internet service` | Collapsed `"No internet service"` $\to$ `"No"` | Hardware warranty protection add-on |
| `TechSupport` | String | `Yes`, `No`, `No internet service` | Collapsed `"No internet service"` $\to$ `"No"` | Dedicated customer technical support |
| `StreamingTV` | String | `Yes`, `No`, `No internet service` | Collapsed `"No internet service"` $\to$ `"No"` | Streaming television subscription |
| `StreamingMovies` | String | `Yes`, `No`, `No internet service` | Collapsed `"No internet service"` $\to$ `"No"` | Streaming cinema subscription |

---

## 4. Target Variable

| Column Name | Type | Allowed Values | Class Balance | Description |
|---|---|---|---|---|
| `Churn` | Binary Integer | `0` (Retained), `1` (Churned) | `0`: 73.46% (5,174)<br/>`1`: 26.54% (1,869) | Whether the customer cancelled service during the period |

---

## 5. Engineered Features

| Feature | Type | Source Columns | Formula | Description |
| --- | --- | --- | --- | --- |
| `tenure_bucket` | Categorical | `tenure` | Binned into `[0-12m, 12-24m, 24-48m, 48-60m, 60-72m]` | Tenure hazard segmentation |
| `avg_monthly_spend` | Float | `TotalCharges`, `tenure` | $\frac{\text{TotalCharges}}{\max(\text{tenure}, 1)}$ | Historical spend rate |
| `services_count` | Integer | 9 service features | Count of active services $\in [0, 9]$ | Product stickiness index |
| `monthly_to_total_ratio` | Float | `MonthlyCharges`, `TotalCharges` | $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1.0}$ | Early tenure burn / bill shock risk |
| `contract_monthly_risk` | Binary | `Contract`, `MonthlyCharges` | $(\text{Contract} == \text{'Month-to-month'}) \land (\text{MonthlyCharges} > 65)$ | High-friction uncommitted contract |
| `clv` | Float | `MonthlyCharges`, `tenure` | $\text{MonthlyCharges} \times \max(\text{tenure}, 1)$ | Customer Lifetime Value |
| `retention_priority_score` | Float | Model Probability, `clv` | $P(\text{Churn}) \times \text{CLV}$ | Expected revenue at risk |
