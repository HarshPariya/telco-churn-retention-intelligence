"""Retention Prioritization page with CSV batch evaluation and ranking."""

import io

import pandas as pd
import streamlit as st

from dashboard.components.charts import plot_priority_scatter, plot_risk_distribution
from dashboard.components.risk_card import render_metric_card
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.inference.predictor import get_predictor


def render_prioritization_page() -> None:
    config = load_config()
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">
                Retention Campaign Prioritization Engine
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Upload a subscriber cohort to rank accounts by <b>Retention Priority Score</b> (Churn Probability × CLV), isolating high-value churn hazards for targeted intervention.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Inference engine could not be loaded: {e}")
        return

    # Data Source Selection: Upload CSV or use holdout test sample
    col_upload, col_sample = st.columns([1.5, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Customer Cohort CSV",
            type=["csv"],
            help="Upload CSV matching Telco schema. Must contain demographic, contract, and service features.",
        )
    with col_sample:
        use_sample = st.button(
            "📂 Load Pre-Loaded Holdout Cohort (300 Customers)", use_container_width=True
        )

    df_to_process = None

    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV loaded successfully ({len(df_to_process)} records).")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
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
        st.info(
            "👆 Please upload a customer CSV file or click 'Load Pre-Loaded Holdout Cohort' to begin."
        )
        return

    # Run Batch Prediction
    with st.spinner(
        f"Scoring {len(df_to_process)} accounts & generating SHAP attribution drivers..."
    ):
        batch_response = predictor.predict_dataframe(df_to_process, batch_explain=True)

    # Convert to presentation DataFrame
    results_list = []
    for p in batch_response.predictions:
        driver_summary = ", ".join([f"{d.feature} ({d.direction[:3]})" for d in p.top_drivers])
        results_list.append(
            {
                "Customer ID": p.customer_id,
                "Churn Probability": p.churn_probability,
                "Predicted Churn": "YES" if p.churn_prediction == 1 else "NO",
                "Risk Tier": p.risk_level,
                "CLV ($)": p.clv,
                "CLV (₹)": p.clv_inr,
                "Priority Score": p.retention_priority_score,
                "Priority (₹)": p.retention_priority_inr,
                "Top Drivers": driver_summary,
            }
        )

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="Priority Score", ascending=False).reset_index(drop=True)

    # Summary KPI Cards Row
    total_scored = len(results_df)
    critical_count = int((results_df["Risk Tier"] == "CRITICAL").sum())
    high_count = int((results_df["Risk Tier"] == "HIGH").sum())
    total_priority_clv = float(
        results_df[results_df["Risk Tier"].isin(["CRITICAL", "HIGH"])]["CLV ($)"].sum()
    )

    st.markdown(
        "<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;' />",
        unsafe_allow_html=True,
    )
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card("Total Accounts Scored", f"{total_scored:,}", "Cohort Volume", "indigo")
    with kpi2:
        render_metric_card("Critical Churn Risk", f"{critical_count:,}", "Immediate Hazard", "rose")
    with kpi3:
        render_metric_card("High Churn Risk", f"{high_count:,}", "Preventative Window", "amber")
    with kpi4:
        render_metric_card(
            "At-Risk Revenue Pool",
            f"${total_priority_clv:,.0f}",
            f"₹{total_priority_clv * config.business.usd_to_inr_rate:,.0f} Exposure",
            "emerald",
        )

    # Interactive Visualizations
    c_left, c_right = st.columns([1, 1.4])
    with c_left:
        st.subheader("Risk Tier Distribution")
        risk_counts = results_df["Risk Tier"].value_counts().to_dict()
        fig_donut = plot_risk_distribution(risk_counts)
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        st.subheader("Retention Priority Matrix")
        st.caption("Bubble size corresponds to Retention Priority Score (Probability × CLV):")
        # Format columns for plot
        scatter_df = results_df.rename(
            columns={
                "Customer ID": "customer_id",
                "Churn Probability": "churn_probability",
                "CLV ($)": "clv",
                "Priority Score": "retention_priority_score",
                "Risk Tier": "risk_level",
            }
        )
        fig_scatter = plot_priority_scatter(scatter_df)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Filterable Data Table
    st.subheader("📋 Ranked Retention Call List (Sorted by Priority Score)")
    filter_tier = st.multiselect(
        "Filter by Risk Tier",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    )

    filtered_df = results_df[results_df["Risk Tier"].isin(filter_tier)]
    st.dataframe(
        filtered_df.style.format(
            {
                "Churn Probability": "{:.1%}",
                "CLV ($)": "${:,.2f}",
                "CLV (₹)": "₹{:,.0f}",
                "Priority Score": "{:,.1f}",
                "Priority (₹)": "₹{:,.0f}",
            }
        ),
        use_container_width=True,
        height=400,
    )

    # Download CSV button
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Prioritized Retention List (CSV)",
        data=csv_buffer.getvalue(),
        file_name="telco_retention_prioritized_cohort.csv",
        mime="text/csv",
        use_container_width=True,
    )
