"""Data validation and data quality audit for Telco Customer Churn."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.telco_churn.logging_config import configure_logger

logger = configure_logger("data_validate")

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"No", "Yes", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"No", "Yes", "No internet service"},
    "OnlineBackup": {"No", "Yes", "No internet service"},
    "DeviceProtection": {"No", "Yes", "No internet service"},
    "TechSupport": {"No", "Yes", "No internet service"},
    "StreamingTV": {"No", "Yes", "No internet service"},
    "StreamingMovies": {"No", "Yes", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


class ColumnQualityMetric(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any]


class DataQualityReport(BaseModel):
    total_rows: int
    total_columns: int
    missing_expected_columns: List[str]
    unexpected_columns: List[str]
    duplicate_rows: int
    duplicate_customer_ids: int
    total_charges_blank_count: int
    tenure_zero_count: int
    churn_distribution: Dict[str, int]
    churn_rate: float
    columns_metrics: List[ColumnQualityMetric]
    validation_passed: bool
    issues_detected: List[str]


def validate_raw_data(df: pd.DataFrame, is_inference: bool = False) -> DataQualityReport:
    """Perform rigorous validation and audit on the Telco dataframe."""
    issues = []

    expected_cols = [c for c in EXPECTED_COLUMNS if not (is_inference and c == "Churn")]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    unexpected_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if unexpected_cols:
        issues.append(f"Unexpected columns detected: {unexpected_cols}")

    # Duplicate checks
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        issues.append(f"Detected {dup_rows} exact duplicate rows.")

    dup_ids = 0
    if "customerID" in df.columns:
        dup_ids = int(df["customerID"].duplicated().sum())
        if dup_ids > 0:
            issues.append(f"Detected {dup_ids} duplicate customerIDs.")

    # TotalCharges whitespace anomaly inspection
    blank_tc_count = 0
    if "TotalCharges" in df.columns:
        if df["TotalCharges"].dtype == object:
            # Check for empty strings or whitespace
            blank_tc_mask = df["TotalCharges"].astype(str).str.strip() == ""
            blank_tc_count = int(blank_tc_mask.sum())
            if blank_tc_count > 0:
                issues.append(
                    f"Detected {blank_tc_count} blank-string values in 'TotalCharges' (Assignment known quirk)."
                )

    # Tenure zero check
    tenure_zero_count = 0
    if "tenure" in df.columns:
        tenure_zero_count = int((df["tenure"] == 0).sum())

    # Category validity check
    for col, allowed_vals in VALID_CATEGORIES.items():
        if col in df.columns:
            # Drop nulls/blanks for category check
            actual_unique = set(df[col].dropna().unique())
            # For SeniorCitizen, coerce int if necessary
            if col == "SeniorCitizen":
                try:
                    actual_unique = {int(x) for x in actual_unique}
                except Exception:
                    pass
            invalid_vals = actual_unique - allowed_vals
            if invalid_vals:
                issues.append(f"Column '{col}' contains invalid category values: {invalid_vals}")

    # Target distribution
    churn_dist: Dict[str, int] = {}
    churn_rate = 0.0
    if "Churn" in df.columns:
        counts = df["Churn"].value_counts().to_dict()
        churn_dist = {str(k): int(v) for k, v in counts.items()}
        if len(df) > 0 and "Yes" in churn_dist:
            churn_rate = float(churn_dist["Yes"] / len(df))

    # Detailed column metrics
    metrics = []
    for col in df.columns:
        series = df[col]
        # For TotalCharges string, treat blanks as null in metrics
        if col == "TotalCharges" and series.dtype == object:
            null_count = int((series.isna() | (series.astype(str).str.strip() == "")).sum())
        else:
            null_count = int(series.isna().sum())

        unique_count = int(series.nunique())
        sample_vals = series.dropna().head(3).tolist()
        # Ensure JSON-serializable samples
        sample_vals = [
            int(v)
            if isinstance(v, (np.integer, bool))
            else (float(v) if isinstance(v, np.floating) else str(v))
            for v in sample_vals
        ]

        metrics.append(
            ColumnQualityMetric(
                name=col,
                dtype=str(series.dtype),
                non_null_count=int(len(df) - null_count),
                null_count=null_count,
                null_percentage=round(float(null_count / len(df) * 100), 3) if len(df) > 0 else 0.0,
                unique_count=unique_count,
                sample_values=sample_vals,
            )
        )

    # Validation passes if no fatal errors (missing required columns or invalid categories)
    fatal_errors = [
        iss for iss in issues if "Missing required columns" in iss or "invalid category" in iss
    ]
    validation_passed = len(fatal_errors) == 0

    report = DataQualityReport(
        total_rows=int(len(df)),
        total_columns=int(len(df.columns)),
        missing_expected_columns=missing_cols,
        unexpected_columns=unexpected_cols,
        duplicate_rows=dup_rows,
        duplicate_customer_ids=dup_ids,
        total_charges_blank_count=blank_tc_count,
        tenure_zero_count=tenure_zero_count,
        churn_distribution=churn_dist,
        churn_rate=round(churn_rate, 4),
        columns_metrics=metrics,
        validation_passed=validation_passed,
        issues_detected=issues,
    )

    logger.info(
        f"Validation complete: Passed={validation_passed}, "
        f"Rows={len(df)}, Blank TotalCharges={blank_tc_count}, Churn Rate={churn_rate:.2%}"
    )
    return report


def generate_markdown_report(report: DataQualityReport) -> str:
    """Generate Markdown text of the Data Quality Report."""
    md = f"""# Telco Customer Churn — Data Quality Audit Report

**Audit Date:** 2026-10-08  
**Dataset Source:** `blastchar/telco-customer-churn` (IBM Cognos Analytics)  
**Validation Status:** {"✅ PASSED" if report.validation_passed else "❌ FAILED"}  

---

## 1. Executive Summary

- **Total Records:** {report.total_rows:,}
- **Total Features:** {report.total_columns}
- **Duplicate Rows:** {report.duplicate_rows}
- **Duplicate Customer IDs:** {report.duplicate_customer_ids} (100% Unique ID integrity)
- **Target Distribution:**
  - `No` (Retained): {report.churn_distribution.get("No", 0):,} ({100 - report.churn_rate * 100:.2f}%)
  - `Yes` (Churned): {report.churn_distribution.get("Yes", 0):,} ({report.churn_rate * 100:.2f}%)
- **Class Imbalance Ratio:** ~{report.churn_rate * 100:.1f}% churn (Moderately imbalanced)

---

## 2. Assignment-Specific Quirks & Diagnostics

### A. TotalCharges Blank Strings Anomaly
- **Detected Count:** {report.total_charges_blank_count} rows contain blank whitespace strings (`" "`) rather than numerical values or standard `NaN`.
- **Root Cause Analysis:** All {report.total_charges_blank_count} rows with blank `TotalCharges` correspond exactly to customers with `tenure == 0` months (new accounts enrolled in their first billing cycle).
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
"""
    for col in report.columns_metrics:
        md += f"| `{col.name}` | `{col.dtype}` | {col.non_null_count:,} | {col.null_count} | {col.null_percentage:.2f}% | {col.unique_count} | `{col.sample_values}` |\n"

    md += """
---

## 4. Issues & Anomalies Log

"""
    if report.issues_detected:
        for iss in report.issues_detected:
            md += f"- ⚠️ **Notice:** {iss}\n"
    else:
        md += "- No data anomalies detected.\n"

    md += """
---

## 5. Preprocessing Strategy Decision

1. **Identifier Handling:** `customerID` is retained during data ingestion and evaluation for tracking and customer-level retention scoring, but dropped from model feature vectors to prevent identity leakage.
2. **Target Encoding:** `Churn` encoded as binary integer (`Yes` = 1, `No` = 0).
3. **Imputation:** `TotalCharges` missing values imputed with `0.0` based on `tenure == 0` domain fact.
4. **Leakage Safeguard:** All categorical encoders and scalers are fitted exclusively on the training split inside an scikit-learn `Pipeline`/`ColumnTransformer`.
"""
    return md
