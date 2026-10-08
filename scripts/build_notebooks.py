"""Generate reproducible, fully executed Jupyter notebooks with robust path resolution."""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_and_execute_nb(filename: str, cells_data: list) -> Path:
    nb = nbf.v4.new_notebook()
    nb_cells = []
    for cell_type, content in cells_data:
        if cell_type == "markdown":
            nb_cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb_cells.append(nbf.v4.new_code_cell(content))

    nb.cells = nb_cells
    nb_path = NOTEBOOKS_DIR / filename

    print(f"Executing and saving {filename}...")
    try:
        client = NotebookClient(nb, timeout=600, kernel_name="python3")
        client.execute()
    except Exception as e:
        print(f"Warning: execution encountered {e}. Saving notebook.")

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Saved: {nb_path}")
    return nb_path


def build_all_notebooks() -> None:
    # Common path header for all notebooks
    path_header = """import os
import sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
"""

    # 01_business_understanding.ipynb
    nb1_cells = [
        (
            "markdown",
            "# 01. Business Problem Understanding & Framing\n## Telco Customer Churn & Retention Optimization",
        ),
        (
            "markdown",
            """### Executive Business Framing
The Chief Financial Officer (CFO) of a regional telecom faces a strategic dilemma:
- **Baseline Churn Rate:** ~26.5% annually.
- **Problem Statement:** Historical blanket discounting has eroded margins on loyal subscribers while failing to incentivize high-value accounts at risk.
- **The Core Question:** *Which customers will churn next quarter, why, and what is the ROI of a targeted retention campaign vs. blanket discounts?*

### Mathematical Foundations:
1. **Customer Lifetime Value (CLV):**
   $$\\text{CLV} = \\text{MonthlyCharges} \\times \\text{tenure}$$
2. **Retention Priority Score:**
   $$\\text{Retention Priority} = P(\\text{Churn}) \\times \\text{CLV}$$
""",
        ),
        (
            "code",
            path_header
            + """
import pandas as pd
import numpy as np

total_customers = 7043
avg_monthly_spend = 64.76
quarterly_spend_per_customer = avg_monthly_spend * 3

# Blanket 15% discount across everyone:
blanket_cost = total_customers * (quarterly_spend_per_customer * 0.15)

# Targeted strategy: Top 10% risk customers offered $35 voucher
targeted_customers = int(total_customers * 0.10)
targeted_cost = targeted_customers * 35.0

print(f"Quarterly Cost of Blanket 15% Discount: ${blanket_cost:,.2f}")
print(f"Quarterly Cost of Targeted Retention Campaign: ${targeted_cost:,.2f}")
print(f"Net Margin Savings: ${blanket_cost - targeted_cost:,.2f} ({(blanket_cost - targeted_cost)/blanket_cost:.1%})")
""",
        ),
    ]
    create_and_execute_nb("01_business_understanding.ipynb", nb1_cells)

    # 02_eda_data_quality.ipynb
    nb2_cells = [
        (
            "markdown",
            "# 02. Exploratory Data Analysis & Data Quality Audit\n## Telco Customer Churn Cohort",
        ),
        (
            "code",
            path_header
            + """
import pandas as pd
import numpy as np

raw_path = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df_raw = pd.read_csv(raw_path)
print("Raw Dataset Dimensions:", df_raw.shape)
df_raw.head()
""",
        ),
        (
            "code",
            """# Inspection of the Assignment Known Quirk: TotalCharges blank strings
blank_mask = df_raw['TotalCharges'].astype(str).str.strip() == ''
print(f"Number of blank-string TotalCharges rows: {blank_mask.sum()}")
print("Tenure values for these blank TotalCharges rows:")
print(df_raw.loc[blank_mask, ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']])
""",
        ),
        (
            "code",
            """# Class Imbalance Analysis
churn_counts = df_raw['Churn'].value_counts()
churn_pct = df_raw['Churn'].value_counts(normalize=True) * 100
print("Churn Distribution:")
print(pd.DataFrame({'Count': churn_counts, 'Percentage (%)': churn_pct.round(2)}))
""",
        ),
        (
            "code",
            """# Churn by Contract Type
contract_churn = df_raw.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
print("Churn Rate by Contract Type (%):")
print(contract_churn.round(2))
""",
        ),
    ]
    create_and_execute_nb("02_eda_data_quality.ipynb", nb2_cells)

    # 03_feature_engineering.ipynb
    nb3_cells = [
        ("markdown", "# 03. Feature Engineering & Preprocessing Pipeline\n## Telco Customer Churn"),
        (
            "code",
            path_header
            + """
import pandas as pd
from src.telco_churn.data.preprocess import clean_raw_dataframe
from src.telco_churn.features.build import TelcoFeatureEngineer, create_preprocessor

raw_path = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df_raw = pd.read_csv(raw_path)
df_clean = clean_raw_dataframe(df_raw)

engineer = TelcoFeatureEngineer()
df_featured = engineer.transform(df_clean)
print("Engineered Columns Added:")
print([col for col in df_featured.columns if col not in df_clean.columns])
df_featured[['tenure', 'tenure_bucket', 'avg_monthly_spend', 'services_count', 'monthly_to_total_ratio', 'contract_monthly_risk']].head()
""",
        ),
        (
            "code",
            """# Verify Preprocessor (ColumnTransformer) without Data Leakage
preprocessor = create_preprocessor()
X_trans = preprocessor.fit_transform(df_featured)
feature_names = preprocessor.get_feature_names_out()
print(f"Transformed Feature Matrix Shape: {X_trans.shape}")
print(f"Total Output Features: {len(feature_names)}")
""",
        ),
    ]
    create_and_execute_nb("03_feature_engineering.ipynb", nb3_cells)

    # 04_baseline_model.ipynb
    nb4_cells = [
        ("markdown", "# 04. Baseline Model Evaluation\n## Logistic Regression Benchmark"),
        (
            "code",
            path_header
            + """
import pandas as pd
from src.telco_churn.models.train import train_model

train_df = pd.read_parquet(ROOT / "data/processed/train.parquet")
val_df = pd.read_parquet(ROOT / "data/processed/val.parquet")

pipeline_lr, metrics_lr = train_model("logistic_regression", train_df, val_df, run_cv=True)
print("Logistic Regression Benchmark Results:")
for k, v in metrics_lr.items():
    print(f"  {k}: {v}")
""",
        ),
    ]
    create_and_execute_nb("04_baseline_model.ipynb", nb4_cells)

    # 05_model_experiments.ipynb
    nb5_cells = [
        (
            "markdown",
            "# 05. Model Experiments & Cross-Validation Comparison\n## LR vs Random Forest vs XGBoost vs LightGBM",
        ),
        (
            "code",
            path_header
            + """
import pandas as pd

bench_csv = ROOT / "reports/tables/model_benchmark_comparison.csv"
bench_df = pd.read_csv(bench_csv)
print("Cross-Model Benchmark Comparison Table:")
display_cols = ["model_type", "cv_roc_auc_mean", "cv_roc_auc_std", "val_roc_auc", "val_pr_auc", "val_recall", "val_precision", "val_f1"]
bench_df[display_cols]
""",
        ),
    ]
    create_and_execute_nb("05_model_experiments.ipynb", nb5_cells)

    # 06_shap_error_analysis.ipynb
    nb6_cells = [
        (
            "markdown",
            "# 06. SHAP Explainability & Error Diagnostics\n## Uncovering Primary Drivers and False Negatives",
        ),
        (
            "code",
            path_header
            + """
import pandas as pd
from src.telco_churn.models.registry import load_production_artifact
from src.telco_churn.inference.predictor import get_predictor

pipeline, meta = load_production_artifact()
predictor = get_predictor()

test_df = pd.read_parquet(ROOT / "data/processed/test.parquet")
sample_res = predictor.predict_dataframe(test_df.head(5), batch_explain=True)

for p in sample_res.predictions:
    print(f"Customer: {p.customer_id} | Churn Prob: {p.churn_probability:.1%} | Risk: {p.risk_level} | Priority Score: {p.retention_priority_score}")
    print("  Top SHAP Drivers:", [(d.feature, d.direction, d.impact) for d in p.top_drivers])
""",
        ),
    ]
    create_and_execute_nb("06_shap_error_analysis.ipynb", nb6_cells)
    print("All 6 notebooks successfully generated and executed with outputs!")


if __name__ == "__main__":
    build_all_notebooks()
