# Feature Catalog
## Telco Customer Churn Prediction & Retention Optimization Platform

This catalog details all input and engineered features utilized within the Telco Churn model pipelines.

---

## 1. Raw Input Features

| Feature Name | Source Column | Data Type | Business Meaning | Modeling Role | Leakage Risk | Preprocessing Treatment |
|---|---|---|---|---|---|---|
| `customerID` | `customerID` | String | Unique alpha-numeric account identifier | Tracking / Auditing | **High (if used in model)** | Dropped from model inputs |
| `gender` | `gender` | String | Demographic gender ('Male', 'Female') | Demographic | None | One-hot encoded |
| `SeniorCitizen` | `SeniorCitizen` | Integer/Categorical | Flag whether customer is $\ge 65$ years (0 or 1) | Demographic | None | Categorical encoding |
| `Partner` | `Partner` | String | Whether customer has a partner ('Yes', 'No') | Demographic | None | One-hot encoded |
| `Dependents` | `Dependents` | String | Whether customer has dependents ('Yes', 'No') | Demographic | None | One-hot encoded |
| `tenure` | `tenure` | Integer | Total months customer has stayed with telecom | Account info | None | StandardScaled & binned |
| `PhoneService` | `PhoneService` | String | Whether customer has phone service ('Yes', 'No') | Service | None | One-hot encoded |
| `MultipleLines` | `MultipleLines` | String | Multiple phone lines ('Yes', 'No') | Service | None | Collapsed "No phone service" $\to$ "No"; One-hot |
| `InternetService` | `InternetService` | String | Internet connection type ('DSL', 'Fiber optic', 'No') | Core Service | None | One-hot encoded |
| `OnlineSecurity` | `OnlineSecurity` | String | Cyber-security add-on ('Yes', 'No') | Add-on Service | None | Collapsed "No internet service" $\to$ "No"; One-hot |
| `OnlineBackup` | `OnlineBackup` | String | Cloud backup add-on ('Yes', 'No') | Add-on Service | None | Collapsed "No internet service" $\to$ "No"; One-hot |
| `DeviceProtection` | `DeviceProtection` | String | Hardware protection add-on ('Yes', 'No') | Add-on Service | None | Collapsed "No internet service" $\to$ "No"; One-hot |
| `TechSupport` | `TechSupport` | String | Dedicated tech support ('Yes', 'No') | Add-on Service | None | Collapsed "No internet service" $\to$ "No"; One-hot |
| `StreamingTV` | `StreamingTV` | String | TV streaming subscription ('Yes', 'No') | Add-on Service | None | Collapsed "No internet service" $\to$ "No"; One-hot |
| `StreamingMovies` | `StreamingMovies` | String | Movies streaming subscription ('Yes', 'No') | Add-on Service | None | Collapsed "No internet service" $\to$ "No"; One-hot |
| `Contract` | `Contract` | String | Contract term ('Month-to-month', 'One year', 'Two year') | Account info | None | One-hot encoded |
| `PaperlessBilling` | `PaperlessBilling` | String | Digital electronic billing ('Yes', 'No') | Account info | None | One-hot encoded |
| `PaymentMethod` | `PaymentMethod` | String | Payment channel (Electronic check, Mailed check, etc.) | Account info | None | One-hot encoded |
| `MonthlyCharges` | `MonthlyCharges` | Float | Current monthly billing rate ($) | Account info | None | StandardScaled |
| `TotalCharges` | `TotalCharges` | Float | Cumulative total charges billed ($) | Account info | None | Coerced from string; 0.0 for tenure 0; StandardScaled |

---

## 2. Engineered Features

| Feature Name | Source Columns | Calculation Formula | Business Rationale | Data Type | Leakage Risk | Preprocessing Treatment |
|---|---|---|---|---|---|---|
| `tenure_bucket` | `tenure` | Binned into `[0-12m, 12-24m, 24-48m, 48-60m, 60-72m]` | Captures non-linear hazard rates; first-year retention cliff is steepest. | Categorical | None | One-hot encoded |
| `avg_monthly_spend` | `TotalCharges`, `tenure`, `MonthlyCharges` | $\frac{\text{TotalCharges}}{\max(\text{tenure}, 1)}$ | Detects historical average billing vs current rate (plan escalation / add-on accretion). | Float | None | StandardScaled |
| `services_count` | 9 service flags | Sum of active services ($\in [0, 9]$) | Represents product stickiness; customers with $\ge 4$ services churn significantly less. | Integer | None | StandardScaled |
| `monthly_to_total_ratio` | `MonthlyCharges`, `TotalCharges` | $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1.0}$ | Identifies early-tenure high-burn accounts susceptible to bill shock. | Float | None | StandardScaled |
| `contract_monthly_risk` | `Contract`, `MonthlyCharges` | $(\text{Contract} == \text{'Month-to-month'}) \land (\text{MonthlyCharges} > 65)$ | High-friction intersection: uncommitted contract with premium price point. | Binary (0/1) | None | StandardScaled |

---

## 3. Leakage Prevention Protocol

1. **Strict Train-Fit Isolation:** Preprocessing transformers (`StandardScaler`, `OneHotEncoder`) are fit **only** on the training subset ($\mathcal{D}_{\text{train}}$) inside `sklearn.compose.ColumnTransformer`.
2. **Zero In-Sample Fitting:** Test and validation datasets are transformed via the fitted pipeline.
3. **No Target Leakage:** Neither `Churn` nor post-churn operational records are accessible in the feature pipeline.
