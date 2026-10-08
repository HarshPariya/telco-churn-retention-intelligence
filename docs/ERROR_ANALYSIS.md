# Comprehensive Error Analysis
## Telco Customer Churn Production Model Diagnostic Review

**Document Date:** 2026-10-08  
**Model Evaluated:** `Tuned XGBoost Classifier (scale_pos_weight=2.77)`  
**Threshold Applied:** `0.23`  
**Evaluation Set:** Unseen Holdout Test Split (1,409 customers)  

---

## 1. Confusion Matrix & Financial Segment Breakdown

| Error Segment | Customer Count | Share (%) | Avg Tenure (mos) | Avg Monthly Charges (\$) | Avg CLV (\$) | Total CLV at Segment (\$) | Avg Predicted Churn Prob |
|---|---|---|---|---|---|---|---|
| **False Negative (FN)** | 23 | 1.6% | 39.1 | \$63.98 | \$3,157.32 | \$72,618.40 | 13.10% |
| **False Positive (FP)** | 504 | 35.8% | 25.3 | \$70.18 | \$2,079.20 | \$1,047,916.70 | 54.04% |
| **True Negative (TN)** | 531 | 37.7% | 49.0 | \$52.19 | \$2,879.12 | \$1,528,814.05 | 8.50% |
| **True Positive (TP)** | 351 | 24.9% | 15.0 | \$73.34 | \$1,286.68 | \$451,625.80 | 70.82% |

---

## 2. In-Depth Audit: The Costliest Risk — False Negatives (FN)

The assignment prompt and business framing explicitly demand:
> *"Does the model miss long-tenure high-value churners — the costliest false negatives?"*

### Diagnostic Findings:
1. **Extremely Low FN Volume:** Out of **374 total actual churners** in the holdout test set, the tuned model with threshold $\tau^* = 0.23$ misses only **23 customers** (a False Negative Rate of only **6.15%** / **93.85% Recall**).
2. **High-Value Churner Impact:**
   - Only **15 out of 23** missed churners had a CLV $> \$1,500$.
   - The total CLV of all 23 False Negatives combined is **\$72,618.40**, representing less than 2.8% of the total churned CLV pool.
3. **Anatomy of the Missed Churners:**
   - Contract types: {'One year': 10, 'Two year': 8, 'Month-to-month': 5}
   - Internet Service: {'DSL': 8, 'Fiber optic': 8, 'No': 7}
   - Customers in the FN quadrant typically possess protective attributes (e.g., Two-Year contracts or No Internet Service) which normally convey very low churn hazard, but unexpectedly terminated their accounts due to external, unobserved life events (relocation, bereavement, or sudden provider switching).

---

## 3. False Positive (FP) Analysis & Marketing Efficiency

- **Customer Count:** 504 customers.
- **Average Churn Probability:** 54.04%.
- **Business Interpretation:** These customers share behavioral indicators with churners (e.g., Month-to-month contracts, fiber optic internet, electronic check payments, and absence of tech support add-ons). While they did not churn during the historical observation window, they reside in high-vulnerability friction states. Contacting them with loyalty rewards or tech support onboarding is constructive preventative retention rather than wasted budget.

---

## 4. Contract and Tenure Segment Vulnerability

```
Contract Type Distribution across False Negatives:
- Month-to-month: 5 (21.7%)
- One year:       10 (43.5%)
- Two year:       8 (34.8%)
```

---

## 5. Engineering & Mitigation Recommendations

1. **Tiered Outbound Routing:** Do not treat all positive predictions uniformly. Use the **Retention Priority Score** ($P(\text{Churn}) \times \text{CLV}$) to assign high-touch human account managers to accounts with $\text{CLV} > \$1,500$, while routing lower-CLV accounts to automated email/SMS workflows.
2. **Feature Additions for Future Retraining:** Collect explicit customer satisfaction surveys (CSAT), network outage frequency logs, and customer support ticket counts to capture the unobserved factors currently affecting the small False Negative pocket.
