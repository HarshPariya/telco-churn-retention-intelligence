"""Retention Prioritization view: Customer cohort ranking, filtering, and export.

Engineered with a warm light enterprise visual system.
"""

import io

import pandas as pd
import streamlit as st

from dashboard.components.cards import render_empty_state, render_metric_card
from dashboard.components.charts import plot_priority_scatter, plot_risk_distribution
from dashboard.components.tokens import (
    COLOR_BORDER,
    COLOR_PRIMARY_BRAND,
    COLOR_PRIMARY_SURFACE,
    COLOR_PRIMARY_TEXT,
    COLOR_SECONDARY_TEXT,
    FONT_FAMILY,
    render_clean_html,
)
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.inference.predictor import get_predictor


def render_retention_prioritization() -> None:
    config = load_config()

    render_clean_html(
        f"""
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 1.75rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 4px; font-family: {FONT_FAMILY};">
                Retention Prioritization
            </h1>
            <p style="color: {COLOR_SECONDARY_TEXT}; font-size: 0.95rem; margin-bottom: 12px; font-family: {FONT_FAMILY};">
                Rank customers by predicted churn risk and customer value so retention teams can focus their effort where it matters most.
            </p>
        </div>
        """
    )

    # Restrained Methodology Card
    render_clean_html(
        f"""
        <div style="
            background: {COLOR_PRIMARY_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: {FONT_FAMILY};
        ">
            <div>
                <span style="font-size: 0.70rem; color: {COLOR_PRIMARY_BRAND}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Prioritization Logic</span>
                <div style="font-size: 0.96rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin-top: 2px;">
                    Retention Priority = Churn Probability × Customer Lifetime Value (CLV)
                </div>
            </div>
            <div style="font-size: 0.80rem; color: {COLOR_SECONDARY_TEXT}; text-align: right;">
                <b>CLV Definition:</b> Monthly Charges × Tenure
            </div>
        </div>
        """
    )

    try:
        predictor = get_predictor()
    except Exception:
        st.error(
            "The prediction service is temporarily unavailable. Please verify that the model artifact is loaded."
        )
        return

    # Upload or Sample Cohort Section
    col_upload, col_sample = st.columns([1.5, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload customer CSV file",
            type=["csv"],
            help="Supported format: CSV with standard customer demographic, account, and service attributes.",
        )
    with col_sample:
        render_clean_html("<div style='height: 28px;'></div>")
        load_sample = st.button("Load Sample Cohort (300 Customers)", width="stretch")

    df_to_process = None

    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            required_cols = {"Contract", "tenure", "MonthlyCharges"}
            if not required_cols.issubset(set(df_to_process.columns)):
                st.error("Uploaded file is missing required customer columns (Contract, tenure, MonthlyCharges).")
                return
            st.success(f"Cohort loaded successfully ({len(df_to_process)} customer records).")
        except Exception:
            st.error("Error reading uploaded CSV. Please check formatting.")
            return
    elif load_sample:
        # Load from raw data sample
        raw_path = PROJECT_ROOT / config.data.raw_path
        if raw_path.exists():
            full_df = pd.read_csv(raw_path)
            df_to_process = full_df.sample(n=min(300, len(full_df)), random_state=42).copy()
            st.session_state["cohort_df"] = df_to_process
        else:
            st.error("Raw reference dataset is not available.")
            return
    elif "cohort_df" in st.session_state:
        df_to_process = st.session_state["cohort_df"]

    if df_to_process is None:
        render_empty_state(
            title="No cohort loaded",
            description="Upload a customer CSV or click 'Load Sample Cohort' to generate prioritized retention actions.",
        )
        return

    # Process and rank cohort
    with st.spinner("Calculating churn risk probabilities and retention priority..."):
        try:
            records = df_to_process.to_dict(orient="records")
            # Predict in batch
            batch_result = predictor.predict_batch(records)
            predictions = batch_result.predictions
        except Exception as e:
            st.error(f"Inference processing failed: {str(e)}")
            return

    results_list = []
    for row, pred in zip(records, predictions, strict=False):
        cust_id = row.get("customerID", row.get("CustomerID", f"CUST-{pred.customer_id}"))
        tenure_val = float(row.get("tenure", 1))
        monthly_val = float(row.get("MonthlyCharges", 0.0))
        clv = tenure_val * monthly_val

        # Priority calculation: Risk Probability * Customer Value
        priority_score = pred.churn_probability * clv

        top_driver = "Tenure"
        if pred.top_factors:
            top_driver = pred.top_factors[0].feature_name.replace("_", " ").title()

        results_list.append(
            {
                "Customer ID": cust_id,
                "Churn Probability": pred.churn_probability,
                "Risk": pred.risk_tier,
                "Contract": row.get("Contract", "Month-to-month"),
                "Monthly Charges": monthly_val,
                "Tenure (Mo)": int(tenure_val),
                "CLV": clv,
                "Retention Priority": priority_score,
                "Top Driver": top_driver,
            }
        )

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="Retention Priority", ascending=False).reset_index(drop=True)

    # Summary KPIs
    total_evaluated = len(results_df)
    high_risk_count = int(results_df["Risk"].isin(["CRITICAL", "HIGH"]).sum())
    total_at_risk_clv = float(results_df[results_df["Risk"].isin(["CRITICAL", "HIGH"])]["CLV"].sum())
    highest_priority_id = results_df.iloc[0]["Customer ID"] if not results_df.empty else "N/A"

    render_clean_html("<div style='height: 8px;'></div>")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Customers Evaluated", f"{total_evaluated:,}", "Cohort volume", color_class="olive")
    with k2:
        render_metric_card("Critical / High Risk", f"{high_risk_count:,}", "Requires active review", color_class="terracotta")
    with k3:
        render_metric_card(
            "At-Risk Customer Value",
            f"${total_at_risk_clv:,.0f}",
            "Cumulative exposed CLV",
            color_class="ochre",
        )
    with k4:
        render_metric_card(
            "Highest Priority Account",
            str(highest_priority_id),
            "Top ranking review",
            color_class="sage",
        )

    # Charts Grid
    render_clean_html(
        f"""
        <div style="font-size: 1.15rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 24px 0 4px 0; font-family: {FONT_FAMILY};">
            Cohort breakdown & priority matrix
        </div>
        <div style="height: 1px; background-color: {COLOR_BORDER}; margin-bottom: 16px;"></div>
        """
    )

    c_left, c_right = st.columns([1, 1.4])
    with c_left:
        render_clean_html(
            f"""
            <div style="font-size: 0.96rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 2px;">
                Cohort Risk Distribution
            </div>
            <div style="font-size: 0.80rem; color: {COLOR_SECONDARY_TEXT}; margin-bottom: 8px;">
                Breakdown of accounts across calibrated risk levels.
            </div>
            """
        )
        risk_counts = results_df["Risk"].value_counts().to_dict()
        fig_donut = plot_risk_distribution(risk_counts)
        st.plotly_chart(fig_donut, width="stretch", config={"displayModeBar": False})

    with c_right:
        render_clean_html(
            f"""
            <div style="font-size: 0.96rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 2px;">
                Retention Priority Matrix
            </div>
            <div style="font-size: 0.80rem; color: {COLOR_SECONDARY_TEXT}; margin-bottom: 8px;">
                Bubble size reflects overall Retention Priority (Probability × CLV).
            </div>
            """
        )
        scatter_df = results_df.rename(
            columns={
                "Customer ID": "customer_id",
                "Churn Probability": "churn_probability",
                "CLV": "clv",
                "Retention Priority": "retention_priority_score",
                "Risk": "risk_level",
            }
        )
        fig_scatter = plot_priority_scatter(scatter_df)
        st.plotly_chart(fig_scatter, width="stretch", config={"displayModeBar": False})

    # Filterable Queue Table
    render_clean_html(
        f"""
        <div style="font-size: 1.15rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 28px 0 4px 0; font-family: {FONT_FAMILY};">
            Ranked retention queue
        </div>
        <div style="height: 1px; background-color: {COLOR_BORDER}; margin-bottom: 12px;"></div>
        """
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        selected_tiers = st.multiselect(
            "Filter by Risk Level",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        )
    with f2:
        available_contracts = list(results_df["Contract"].unique())
        selected_contracts = st.multiselect(
            "Filter by Contract",
            options=available_contracts,
            default=available_contracts,
        )
    with f3:
        min_clv = st.number_input(
            "Minimum CLV ($)",
            min_value=0.0,
            value=0.0,
            step=100.0,
        )
    with f4:
        min_priority = st.number_input(
            "Minimum Priority Score",
            min_value=0.0,
            value=0.0,
            step=50.0,
        )

    # Apply Filters
    mask = (
        results_df["Risk"].isin(selected_tiers)
        & results_df["Contract"].isin(selected_contracts)
        & (results_df["CLV"] >= min_clv)
        & (results_df["Retention Priority"] >= min_priority)
    )
    filtered_df = results_df[mask].reset_index(drop=True)

    display_cols = [
        "Customer ID",
        "Churn Probability",
        "Risk",
        "Contract",
        "CLV",
        "Retention Priority",
        "Top Driver",
    ]

    st.dataframe(
        filtered_df[display_cols].style.format(
            {
                "Churn Probability": "{:.1%}",
                "CLV": "${:,.2f}",
                "Retention Priority": "{:,.1f}",
            }
        ),
        width="stretch",
        height=380,
    )

    # Download Action
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Prioritized CSV",
        data=csv_buffer.getvalue(),
        file_name="prioritized_customer_retention_queue.csv",
        mime="text/csv",
        width="stretch",
    )
