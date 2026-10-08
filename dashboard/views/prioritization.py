"""Retention Prioritization view: Customer cohort ranking, filtering, and export."""

import io

import pandas as pd
import streamlit as st

from dashboard.components.cards import render_empty_state, render_metric_card
from dashboard.components.charts import plot_priority_scatter, plot_risk_distribution
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.inference.predictor import get_predictor


def render_retention_prioritization() -> None:
    config = load_config()

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 1.65rem; font-weight: 700; color: #172033; margin-bottom: 4px;">
                Retention Prioritization
            </h1>
            <p style="color: #5B6577; font-size: 0.95rem; margin-bottom: 12px;">
                Rank customers by predicted churn risk and customer value to focus retention effort where it matters most.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Methodology Card
    st.markdown(
        """
        <div style="
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div>
                <span style="font-size: 0.72rem; color: #2563EB; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Prioritization Logic</span>
                <div style="font-size: 1.0rem; font-weight: 700; color: #172033; margin-top: 2px;">
                    Retention Priority = Churn Probability × Customer Lifetime Value (CLV)
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #5B6577; text-align: right;">
                <b>CLV Definition:</b> Monthly Charges × Tenure
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
            help="Supported file format: CSV containing customer demographic, account, and subscribed service columns.",
        )
    with col_sample:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
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
            st.error("Unable to process the uploaded CSV file. Please verify file format.")
            return
    elif load_sample or "sample_cohort_loaded" in st.session_state:
        st.session_state["sample_cohort_loaded"] = True
        test_path = PROJECT_ROOT / config.data.test_path
        if test_path.exists():
            df_test = pd.read_parquet(test_path)
            df_to_process = df_test.head(300).copy()
        else:
            df_raw = pd.read_csv(PROJECT_ROOT / config.data.raw_path)
            df_to_process = df_raw.head(300).copy()

    # Empty State if no cohort loaded
    if df_to_process is None:
        render_empty_state(
            message="No customer cohort has been loaded yet.",
            subtext="Upload a CSV file or click 'Load Sample Cohort' to evaluate customer retention priorities.",
        )
        return

    # Run Batch Scoring
    with st.spinner(f"Evaluating {len(df_to_process)} accounts and computing contributing factors..."):
        try:
            batch_response = predictor.predict_dataframe(df_to_process, batch_explain=True)
        except Exception:
            st.error("Unable to score this cohort. Please ensure input columns match the standard customer schema.")
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
                "CLV": p.clv,
                "Retention Priority": p.retention_priority_score,
                "Top Driver": f"{top_driver_name} {top_driver_dir}",
            }
        )

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by="Retention Priority", ascending=False).reset_index(drop=True)

    # Summary KPIs
    total_evaluated = len(results_df)
    high_risk_count = int(results_df["Risk"].isin(["CRITICAL", "HIGH"]).sum())
    total_at_risk_clv = float(results_df[results_df["Risk"].isin(["CRITICAL", "HIGH"])]["CLV"].sum())
    highest_priority_id = results_df.iloc[0]["Customer ID"] if not results_df.empty else "N/A"

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Customers Evaluated", f"{total_evaluated:,}", "Cohort volume", "blue")
    with k2:
        render_metric_card("Critical / High Risk", f"{high_risk_count:,}", "Requires active review", "rose")
    with k3:
        render_metric_card(
            "At-Risk Customer Value",
            f"${total_at_risk_clv:,.0f}",
            "Cumulative exposed CLV",
            "amber",
        )
    with k4:
        render_metric_card(
            "Highest Priority Account",
            str(highest_priority_id),
            "Top ranking retention review",
            "emerald",
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Charts Grid
    c_left, c_right = st.columns([1, 1.4])
    with c_left:
        st.markdown(
            """
            <div style="font-size: 1.0rem; font-weight: 600; color: #172033; margin-bottom: 2px;">
                Cohort Risk Distribution
            </div>
            <div style="font-size: 0.82rem; color: #5B6577; margin-bottom: 8px;">
                Breakdown of accounts across risk levels.
            </div>
            """,
            unsafe_allow_html=True,
        )
        risk_counts = results_df["Risk"].value_counts().to_dict()
        fig_donut = plot_risk_distribution(risk_counts)
        st.plotly_chart(fig_donut, width="stretch")

    with c_right:
        st.markdown(
            """
            <div style="font-size: 1.0rem; font-weight: 600; color: #172033; margin-bottom: 2px;">
                Retention Priority Matrix
            </div>
            <div style="font-size: 0.82rem; color: #5B6577; margin-bottom: 8px;">
                Bubble size reflects overall Retention Priority (Probability × CLV).
            </div>
            """,
            unsafe_allow_html=True,
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
        st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Filterable Queue Table
    st.markdown(
        """
        <div style="font-size: 1.05rem; font-weight: 600; color: #172033; margin-bottom: 2px;">
            Ranked Retention Queue
        </div>
        <div style="font-size: 0.82rem; color: #5B6577; margin-bottom: 12px;">
            Customers sorted descending by Retention Priority score:
        </div>
        """,
        unsafe_allow_html=True,
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

    # Download action
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv_buffer.getvalue(),
        file_name="prioritized_customer_retention_queue.csv",
        mime="text/csv",
        width="stretch",
    )
