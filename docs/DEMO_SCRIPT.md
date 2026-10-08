# Live Demo & Defense Script (8–10 Minutes)

## Telco Customer Churn Prediction & Retention Optimization Platform

**Audience:** Internship Evaluation Panel, Senior Engineers, and Executive Mentors  
**Presenter Role:** Senior ML / MLOps Lead  

---

## ⏱️ Minute-by-Minute Presentation Plan

### [00:00 - 01:30] Phase 1: Business Framing & Problem Statement

- **Narrative:** "Good morning, evaluators. In telecommunications, losing an account costs 5x more than retaining one. Our CFO asked a crucial strategic question: *'Which customers will churn next quarter, why are they leaving, and what is the ROI of targeted retention vs. blanket discounts?'*
- Rather than giving everyone a margin-eroding blanket discount, we built an end-to-end Machine Learning and Decision Intelligence platform that identifies high-risk customers, explains their exact drivers, and prioritizes them by **Retention Priority Score** ($P(\text{Churn}) \times \text{CLV}$)."

### [01:30 - 03:00] Phase 2: Executive Overview Dashboard

- **Screen:** Open Streamlit Dashboard (`http://localhost:8501`) on **Executive Overview**.
- **Demonstration:**
  - Highlight Portfolio KPIs: 7,043 analyzed California subscribers, 26.5% baseline churn rate, \$64.76 average monthly billing, and \$16.0M portfolio CLV.
  - Explain the **CFO ROI Model**: A blanket 15% discount across all customers costs over \$205,000 quarterly, whereas a targeted retention campaign focusing on the top-decile risk cohort (\$35 offer) costs only \$24,640—saving **over 80% in promotional capital** while preserving high-CLV accounts.
  - Point to the **Contract Comparison Chart**: Month-to-month contracts churn at 42.7%, while 2-year contracts churn at only 2.8%.

### [03:00 - 05:00] Phase 3: Single Customer Diagnostic & SHAP Explainability

- **Screen:** Navigate to **Customer Prediction** tab.
- **Demonstration:**
  - Enter a sample high-risk profile: Month-to-month contract, tenure = 4 months, fiber optic broadband, electronic check payment, monthly charges = \$85.00.
  - Click **⚡ Evaluate Churn Risk & Explain**.
  - Review Output:
    - **Risk Badge:** `CRITICAL CHURN RISK` (~82% Churn Propensity).
    - **Financials:** CLV = \$340.00 (₹28,390), Retention Priority Score = 278.8.
    - **Top 3 SHAP Drivers:**
      1. `Month-to-Month Contract` (🔺 Escalates Risk)
      2. `Account Tenure (4 months)` (🔺 Low loyalty hazard)
      3. `Electronic Check Payment` (🔺 Payment friction)
    - **Prescriptive Retention Action:** Recommend targeted 1-year contract discount and tech support onboarding.

### [05:00 - 06:30] Phase 4: Retention Prioritization Engine (Cohort Bulk Scoring)

- **Screen:** Navigate to **Retention Prioritization** tab.
- **Demonstration:**
  - Click **Load Pre-Loaded Holdout Cohort (300 Customers)** or upload custom CSV.
  - Watch real-time batch inference and Tree-SHAP driver extraction.
  - Inspect the **Ranked Call List**: Accounts sorted descending by Priority Score.
  - Show the **Priority Bubble Chart**: Explaining that a customer with 50% churn risk and \$4,000 CLV is ranked much higher than a customer with 90% churn risk and \$100 CLV.
  - Demonstrate CSV export download for frontline retention call centers.

### [06:30 - 07:30] Phase 5: Model Governance & Holdout Insights

- **Screen:** Navigate to **Model Insights** tab.
- **Demonstration:**
  - Show actual measured metrics on the unseen holdout test split:
    - **ROC-AUC:** 0.8439
    - **PR-AUC:** 0.6582
    - **Recall:** 93.85% (catches 351 out of 374 actual churners!)
  - Explain the **Cost-Sensitive Threshold ($\tau^* = 0.23$)**: Minimizing the financial loss of uncontacted high-CLV churners.
  - Point to the confusion matrix and global SHAP beeswarm summary plots.

### [07:30 - 08:30] Phase 6: Production Engineering, API & Docker

- **Screen:** Open Swagger UI (`http://localhost:8000/docs`).
- **Demonstration:**
  - Show `/health`, `/metadata`, `/predict`, `/predict/batch`.
  - Execute a live prediction curl request in terminal.
  - Show `docker-compose.yml` and container health checks.
  - Show automated test suite passing (18/18 pytest tests).

### [08:30 - 09:30] Phase 7: Conclusion & Business Impact

- **Wrap-up:** "In summary, we have transitioned telecom customer retention from reactive blanket discounting to proactive, explainable, cost-optimal decision intelligence. The platform is modular, tested, containerized, and ready for deployment. Thank you, and I look forward to your questions."
