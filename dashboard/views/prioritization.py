"""Retention Prioritization view with CSV cohort scoring, filtering, and export."""

import io

import pandas as pd
import streamlit as st

from dashboard.components.charts import plot_priority_scatter, plot_risk_distribution
from dashboard.components.risk_card import render_metric_card
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.inference.predictor import get_predictor


def render_retention_prioritization() -> None:
    config = load_config()

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">
                Retention Prioritization
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem; margin-bottom: 14px;">
                Score a customer cohort and identify high-value customers with the greatest predicted churn risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Formula Callout Banner
    st.markdown(
        """
        <div style="
            background: rgba(79, 70, 229, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div>
                <span style="font-size: 0.82rem; color: #a5b4fc; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Prioritization Logic</span>
                <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; margin-top: 2px;">
                    Retention Priority = Churn Probability × Customer Lifetime Value (CLV)
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; text-align: right;">
                CLV = Monthly Charges × Tenure
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor = get_predictor()
    except Exception:
        st.error(
            "The prediction model is currently unavailable. Please verify the model service or run `python scripts/train_model.py`."
        )
        return

    # Upload or Sample Cohort
    col_upload, col_sample = st.columns([1.5, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Customer Cohort CSV",
            type=["csv"],
            help="Upload a CSV file containing customer demographics, contract details, and subscribed services.",
        )
    with col_sample:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        use_sample = st.button(
            "📂 Load Sample Cohort (300 Customers)", width="stretch"
        )

    df_to_process = None

    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            required_cols = {"Contract", "tenure", "MonthlyCharges"}
            if not required_cols.issubset(set(df_to_process.columns)):
                st.error("This file is missing required customer fields (Contract, tenure, MonthlyCharges).")
                return
            st.success(f"Cohort CSV loaded successfully ({len(df_to_process)} customer records).")
        except Exception:
            st.error("Error reading uploaded CSV file. Please verify file format and columns.")
            return
    elif use_sample or "sample_cohort_loaded" in st.session_state:
        st.session_state["sample_cohort_loaded"] = True
        test_path = PROJECT_ROOT / config.data.test_path
        if test_path.exists():
            df_test = pd.read_parquet(test_path)
            df_to_process = df_test.head(300).copy()
        else:
            df_raw = pd.read_csv(PROJECT_ROOT / config.data.raw_path)
            df_to_process = df_raw.head(300).copy()

    if df_to_process is None:
        st.info("👆 Please upload a customer CSV file or click 'Load Sample Cohort' to begin cohort evaluation.")
        return

    # Run Batch Scoring
    with st.spinner(f"Scoring {len(df_to_process)} customer profiles & computing SHAP attributions..."):
        try:
            batch_response = predictor.predict_dataframe(df_to_process, batch_explain=True)
        except Exception:
            st.error("Cohort scoring encountered an error. Please verify input data schema.")
            return

    results_list = []
    for p in batch_response.predictions:
        top_driver_name = p.top_drivers[0].feature if p.top_drivers else "N/A"
        top_driver_dir = "↑" if (p.top_drivers and p.top_drivers[0].direction == "INCREASES_CHURN") else "↓"
        contract_val = df_to_process.loc[
            df_to_process.get("customerID", df_to_process.index) == p.customer_id, "Contract"
        ].values if "Contract" in df_to_process.columns else ["Unknown"]
        contract_str = str(contract_val[0]) if len(contract_val) > 0 else "Month-to-month"

        results_list.append(
            {
                "Customer ID": p.customer_id,
                "Churn Probability": p.churn_probability,
                "Risk": p.risk_level,
                "Contract": contract_str,
                "CLV ($)": p.clv,
                "Priority Score": p.retention_priority_score,
                "Top Driver": f"{top_driver_name} {top_driver_dir}",
            }
        )

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="Priority Score", ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1
    results_df.index.name = "Rank"

    # KPI Summary Cards
    total_scored = len(results_df)
    high_risk_count = int(results_df["Risk"].isin(["CRITICAL", "HIGH"]).sum())
    total_at_risk_clv = float(results_df[results_df["Risk"].isin(["CRITICAL", "HIGH"])]["CLV ($)"].sum())
    avg_churn_prob = float(results_df["Churn Probability"].mean())

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;' />", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Customers Scored", f"{total_scored:,}", "Cohort Volume", "indigo")
    with k2:
        render_metric_card("High-Risk Customers", f"{high_risk_count:,}", "Critical & High Risk", "rose")
    with k3:
        render_metric_card(
            "At-Risk Portfolio CLV",
            f"${total_at_risk_clv:,.0f}",
            f"≈ ₹{total_at_risk_clv * config.business.usd_to_inr_rate:,.0f} Exposure",
            "amber",
        )
    with k4:
        render_metric_card(
            "Average Churn Probability",
            f"{avg_churn_prob:.1%}",
            "Cohort Average",
            "emerald",
        )

    # Interactive Visualizations
    v_left, v_right = st.columns([1, 1.4])
    with v_left:
        st.subheader("Risk Tier Distribution")
        risk_counts = results_df["Risk"].value_counts().to_dict()
        fig_donut = plot_risk_distribution(risk_counts)
        st.plotly_chart(fig_donut, width="stretch")

    with v_right:
        st.subheader("Retention Priority Matrix")
        st.caption("Bubble size indicates Retention Priority Score (Probability × CLV):")
        scatter_df = results_df.rename(
            columns={
                "Customer ID": "customer_id",
                "Churn Probability": "churn_probability",
                "CLV ($)": "clv",
                "Priority Score": "retention_priority_score",
                "Risk": "risk_level",
            }
        )
        fig_scatter = plot_priority_scatter(scatter_df)
        st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;' />", unsafe_allow_html=True)

    # Filters Section
    st.subheader("📋 Prioritized Customer Queue")
    st.caption("Customers ranked descending by Retention Priority Score:")

    f1, f2, f3 = st.columns(3)
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
        & (results_df["Priority Score"] >= min_priority)
    )
    filtered_df = results_df[mask].reset_index()
    filtered_df["Rank"] = filtered_df.index + 1

    st.dataframe(
        filtered_df[
            [
                "Rank",
                "Customer ID",
                "Churn Probability",
                "Risk",
                "Contract",
                "CLV ($)",
                "Priority Score",
                "Top Driver",
            ]
        ].style.format(
            {
                "Churn Probability": "{:.1%}",
                "CLV ($)": "${:,.2f}",
                "Priority Score": "{:,.1f}",
            }
        ),
        width="stretch",
        height=380,
    )

    # Download Filtered Results
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Prioritized CSV",
        data=csv_buffer.getvalue(),
        file_name="prioritized_retention_queue.csv",
        mime="text/csv",
        width="stretch",
    )
