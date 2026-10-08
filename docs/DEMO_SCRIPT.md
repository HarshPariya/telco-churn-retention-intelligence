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

- **Screen:** Open Streamlit Dashboard (`http://localhost:8501`) on **Customer Retention Overview**.
- **Demonstration:**
  - Highlight Portfolio KPIs: 7,043 analyzed California subscribers, 26.5% baseline churn rate, $64.76 average monthly billing, and portfolio CLV.
  - Explain the **Illustrative Economics Scenario**: A blanket 15% discount across all customers costs over $205,000 quarterly, whereas a targeted retention campaign focusing on the top-decile risk cohort ($35 offer) costs only $24,640—illustrating the potential efficiency of targeted outreach.
  - Point to the **Contract Comparison Chart**: Month-to-month contracts show an observed churn rate of 42.7%, while 2-year contracts show 2.8%.

### [03:00 - 05:00] Phase 3: Single Customer Diagnostic & SHAP Explainability

- **Screen:** Navigate to **Customer Prediction** tab.
- **Demonstration:**
  - Enter a sample high-risk profile: Month-to-month contract, tenure = 4 months, fiber optic broadband, electronic check payment, monthly charges = $79.85.
  - Click **Predict Churn Risk**.
  - Review Output:
    - **Risk Badge:** `HIGH RISK` / `CRITICAL RISK` with soft warm background.
    - **Financials:** Customer Lifetime Value (CLV) and Retention Priority Score.
    - **Top 3 Drivers:**
      1. `Month-to-month contract` (Increases predicted risk)
      2. `Account tenure` (Increases predicted risk)
      3. `Electronic check payment` (Increases predicted risk)
    - **Suggested Review:** Neutral workflow recommendation for retention specialist review before contacting subscriber.

### [05:00 - 06:30] Phase 4: Retention Prioritization Engine (Cohort Bulk Scoring)

- **Screen:** Navigate to **Retention Prioritization** tab.
- **Demonstration:**
  - Click **Load Sample Cohort (300 Customers)** or upload custom CSV.
  - Watch real-time batch inference and driver extraction.
  - Inspect the **Ranked Retention Queue**: Accounts sorted descending by Retention Priority score ($P(\text{Churn}) \times \text{CLV}$).
  - Show the **Priority Bubble Chart**: Explaining that a customer with moderate churn risk and high CLV is prioritized higher than a customer with high churn risk but negligible tenure/CLV.
  - Demonstrate CSV export download for frontline retention teams.

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
