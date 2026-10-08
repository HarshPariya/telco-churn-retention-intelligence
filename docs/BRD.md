# Business Requirements Document (BRD)

## Telco Customer Churn Prediction & Retention Optimization Platform

**Document Version:** 1.0.0  
**Date:** 2026-10-08  
**Project Track:** Track A — Binary Classification (Telecom/SaaS Retention)  
**Sponsor:** Chief Financial Officer (CFO) & VP of Customer Success  

---

## 1. Executive Summary & Business Problem

Telecom service providers operate in saturated, highly competitive markets where customer acquisition cost (CAC) typically exceeds customer retention cost (CRC) by 5x to 7x.

In this regional telecom setting:

- **Baseline Churn Rate:** ~26.5% annually.
- **Problem Statement:** Historical retention initiatives relied on indiscriminate blanket discounting across entire cohorts. This resulted in:
  1. Massive margin erosion on satisfied customers who never intended to leave.
  2. Sub-optimal incentives for high-value customers at genuine risk of churn.
  3. Failure to intervene early enough for high-tenure, high-spend accounts where churn incurs irreparable customer lifetime value (CLV) loss.

### The CFO's Core Business Question
>
> *"Which customers will churn next quarter, why are they likely to churn, and what is the ROI of a targeted retention campaign vs. blanket discounts?"*

---

## 2. Project Objectives

### 2.1 Business Objectives

1. **Reduce High-Value Churn:** Identify customers exhibiting early churn propensity signals (specifically on month-to-month contracts, fiber optic connections, and electronic check billing).
2. **Maximize Retention ROI:** Replace blanket discounts with a tiered, cost-sensitive retention strategy prioritizing accounts by **Retention Priority Score** ($\text{Churn Probability} \times \text{CLV}$).
3. **Operationalize Proactive Interventions:** Deliver daily/weekly retention lists with the top 3 drivers per customer, enabling frontline retention reps to offer tailored, non-cash incentives (e.g., tech support add-ons, contract extensions, payment method upgrades).

### 2.2 Machine Learning Objectives

1. **Binary Classification System:** Accurately estimate customer churn probability $P(\text{Churn} = 1 \mid X)$.
2. **Performance Benchmarks:**
   - Outperform baseline Logistic Regression on ROC-AUC, PR-AUC, and F1 score.
   - Achieve ROC-AUC $\ge 0.82$ and PR-AUC $\ge 0.60$ with calibrated tree-based models (Random Forest, XGBoost, LightGBM).
   - Prioritize high **Recall** at optimized cost-sensitive decision thresholds to minimize costly False Negatives (unintervened high-CLV churners).
3. **Local & Global Explainability:** Provide exact SHAP values for each individual prediction to satisfy governance and operational transparency.

---

## 3. Stakeholders & Persona Workflows

| Stakeholder Persona | Role & Responsibilities | Core System Interaction |
| --- | --- | --- |
| **Chief Financial Officer (CFO)** | Executive Sponsor; capital allocation; margin preservation | Reviews Executive Overview dashboard, ROI metrics, CLV at risk, and savings vs. blanket discount. |
| **VP of Customer Retention** | Operational strategy; retention campaign budgeting | Configures offer cost assumptions, reviews risk tiers (Critical, High, Medium, Low), and tracks cohort stability. |
| **Retention Campaign Specialist** | Frontline execution; outreach to high-risk customers | Uses Customer Prediction UI and Batch Prioritization CSV export to access top 3 drivers for personalized retention calls. |
| **MLOps / Platform Engineer** | System reliability, drift monitoring, API SLAs | Monitors API `/health`, latency, PSI feature drift, and experiment tracking via MLflow. |

---

## 4. Business Mathematics & Financial Modeling

### 4.1 Customer Lifetime Value (CLV)

As specified by the enterprise project mandate:
$$\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$$

*Rationale:* In subscription contracts, realized lifetime value directly reflects the revenue stream generated over the account's historical tenure. For newly enrolled customers ($\text{tenure} = 0$), baseline projected CLV initializes to $1 \times \text{MonthlyCharges}$.

### 4.2 Retention Priority Score

$$\text{Retention Priority Score} = P(\text{Churn}) \times \text{CLV}$$

This formulation represents the **Expected Value of Lost Revenue**. A customer with 90% churn probability but only $30 CLV represents $27 expected loss, whereas a customer with 40% churn probability and $3,500 CLV represents $1,400 expected loss. Retention teams must prioritize the latter.

### 4.3 Cost-Sensitive Decision Optimization

Standard ML pipelines default to an arbitrary threshold $\tau = 0.50$, implicitly assuming that False Positives and False Negatives carry equal economic cost. In customer retention:

- **False Negative (FN) Cost:** Customer churns unnoticed. Cost = $\text{CLV} \times \text{Churn Loss Factor}$.
- **False Positive (FP) Cost:** Unnecessary retention contact / promotional gift provided to a customer who would have stayed anyway. Cost = $\text{Retention Offer Cost} = \$35.00$.

The optimal classification threshold $\tau^*$ is selected to minimize total business loss:
$$\tau^* = \arg\min_{\tau} \sum_{i} \left[ \text{FN}_i(\tau) \cdot (\text{CLV}_i \cdot \text{Loss Factor}) + \text{FP}_i(\tau) \cdot \text{Offer Cost} \right]$$

### 4.4 Currency Conversion Policy (INR Impact)

To support regional executive reporting without inventing arbitrary real-time currency rates:

- Base dataset currency: USD (\$)
- Configurable exchange rate: ₹83.50 per USD (configurable in `.env` and `configs/base.yaml`).

---

## 5. Decision Workflow

```
[Customer Profile / CSV Upload]
           │
           ▼
[FastAPI Ingestion / Preprocessing Pipeline]
           │
           ▼
[Calibrated Classifier: Predict Churn Probability]
           │
           ├────────────────────────────┐
           ▼                            ▼
[SHAP Explainer Engine]         [Business Calculation Engine]
  - Top 3 Risk Drivers            - CLV = MonthlyCharges * tenure
  - Directional Impact (+/-)      - Priority = P(Churn) * CLV
           │                            │
           └──────────────┬─────────────┘
                          ▼
             [Risk Stratification Tier]
             (Critical / High / Medium / Low)
                          │
                          ▼
            [Retention Action Strategy]
             - Critical & High CLV: Dedicated Account Rep Call + Customized Bundle
             - High Risk, Low CLV: Automated Digital Survey + Discount Incentive
             - Low Risk: Standard Maintenance
```

---

## 6. Assumptions & Constraints

1. **Synthetic/Anonymized Cohort:** The dataset reflects 7,043 customer accounts in California under CC BY-SA 4.0 license.
2. **Contract Modality:** Month-to-month accounts exhibit ~35.5% churn, while 2-year contracts exhibit <5% churn. Interventions on month-to-month contracts focus on tenure lock-in.
3. **Data Immutability:** Raw CSV files are never edited in place. All cleaning transformations must be reproducible and isolated to downstream stages.
4. **Latency Requirement:** Single-record API inference response $< 100\text{ ms}$, batch inference (1,000 records) $< 2.0\text{ s}$.

---

## 7. Success Criteria & KPIs

| Metric | Target | Evaluation Method |
| --- | --- | --- |
| **Model Discrimination (ROC-AUC)** | $\ge 0.83$ | Holdout Test Set Evaluation |
| **Precision-Recall AUC (PR-AUC)** | $\ge 0.62$ | Holdout Test Set Evaluation |
| **Churn Recall (High-Risk Segment)** | $\ge 75\%$ | Optimized Threshold Evaluation |
| **API Availability & Health** | $99.9\%$ uptime | `/health` endpoint probe |
| **Retention Campaign Efficiency** | $+35\%$ cost savings vs. blanket discount | Business Simulation Matrix |
