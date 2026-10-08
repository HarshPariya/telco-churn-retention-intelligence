# Monitoring & Drift Detection Architecture

## Telco Customer Churn & Retention Optimization Platform

---

## 1. Monitoring Scope

To guarantee model reliability over long operational lifecycles, four primary dimensions are monitored:

1. **Prediction Volume & Risk Distribution:** Tracking ratio of `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` predictions.
2. **Feature Distribution Drift (Data Drift):** Detecting shifts in input demographics, tenure cohorts, and billing structures.
3. **Concept Drift & Performance Degradation:** Measuring ground-truth churn rates against model probability calibration once actual account outcomes materialize.
4. **Service Health & System Latency:** Monitoring API response latency ($p50, p95, p99$), request error rates, and system uptime.

---

## 2. Statistical Drift Metrics

### 2.1 Population Stability Index (PSI)

The platform utilizes **Population Stability Index (PSI)** to compare feature distributions between the baseline training distribution ($\mathcal{D}_{\text{train}}$) and inference batches ($\mathcal{D}_{\text{curr}}$):

$$\text{PSI} = \sum_{k=1}^{K} (P_{\text{curr}, k} - P_{\text{base}, k}) \times \ln\left( \frac{P_{\text{curr}, k}}{P_{\text{base}, k}} \right)$$

#### Alert Thresholds

- **$\text{PSI} < 0.10$:** Feature distribution is **STABLE**. No operational action required.
- **$0.10 \le \text{PSI} < 0.20$:** **MODERATE DRIFT** detected. Flag for monitoring; investigate marketing campaign changes.
- **$\text{PSI} \ge 0.20$:** **SIGNIFICANT DRIFT** detected. Triggers retraining alert and feature distribution diagnostic.

### 2.2 Kolmogorov-Smirnov (KS) Test

For continuous features (`MonthlyCharges`, `tenure`, `TotalCharges`), the two-sample Kolmogorov-Smirnov statistic detects shifts in cumulative empirical distribution functions.

---

## 3. Operational Implementation

The platform includes an automated drift module in `src/telco_churn/monitoring/drift.py`:

```python
from src.telco_churn.monitoring.drift import detect_dataset_drift

# Audit incoming monthly cohort against training baseline
report = detect_dataset_drift(baseline_df=train_df, current_df=monthly_cohort_df)
print(f"Overall Platform Status: {report.overall_status}")
```
