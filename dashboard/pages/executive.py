"""Executive Overview page for C-Suite and Retention Directors."""

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    plot_churn_by_contract,
    plot_churn_by_tenure_bucket,
)
from dashboard.components.risk_card import render_metric_card
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.models.registry import load_production_artifact


def render_executive_page() -> None:
    config = load_config()
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">
                Executive Retention Intelligence
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Strategic overview answering: Which customers will churn next quarter, why, and what is the ROI of targeted retention vs. blanket discounts?
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        pipeline, metadata = load_production_artifact()
    except Exception:
        st.warning("Production model artifact not yet found. Please run training pipeline.")

    # Load cleaned dataset for aggregate statistics
    interim_path = PROJECT_ROOT / config.data.interim_path
    if interim_path.exists():
        df = pd.read_parquet(interim_path)
    else:
        df = pd.read_csv(PROJECT_ROOT / config.data.raw_path)
        if "Churn" in df.columns and df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    total_customers = len(df)
    churn_rate = float(df["Churn"].mean()) if "Churn" in df.columns else 0.2654
    avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 64.76
    total_clv = (
        float((df["MonthlyCharges"] * df["tenure"]).sum())
        if "MonthlyCharges" in df.columns and "tenure" in df.columns
        else 0.0
    )

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            title="Total Subscribers",
            value=f"{total_customers:,}",
            subtext="Analyzed California Accounts",
            color_class="indigo",
        )
    with col2:
        render_metric_card(
            title="Historical Churn Rate",
            value=f"{churn_rate:.1%}",
            subtext="Annual Baseline Churn",
            color_class="rose",
        )
    with col3:
        render_metric_card(
            title="Avg Monthly Billing",
            value=f"${avg_monthly:.2f}",
            subtext=f"≈ ₹{avg_monthly * config.business.usd_to_inr_rate:,.0f} / mo",
            color_class="emerald",
        )
    with col4:
        render_metric_card(
            title="Total Portfolio CLV",
            value=f"${total_clv / 1e6:.2f}M",
            subtext=f"₹{total_clv * config.business.usd_to_inr_rate / 1e7:.1f} Cr Realized",
            color_class="amber",
        )

    st.markdown(
        "<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;' />",
        unsafe_allow_html=True,
    )

    # Financial ROI Framing
    st.subheader("💡 CFO ROI Framing: Targeted Retention vs. Blanket Discounts")
    with st.expander("Explore the Retention Economics Model", expanded=True):
        st.markdown(
            rf"""
            - **Blanket Discount Strategy:** Providing a standard 15% discount across all 7,043 customers costs **\${total_customers * avg_monthly * 0.15 * 3:,.2f}** over a quarter, giving margin relief to ~73% of customers who would have stayed anyway.
            - **Targeted AI Retention Strategy:** Contacting only the top-decile risk cohort identified by **Retention Priority Score** ($\sim 704$ accounts) with a targeted incentive (\${config.business.retention_offer_cost:.2f}) costs only **\${704 * config.business.retention_offer_cost:,.2f}** — saving over **75% in campaign spend** while preventing high-CLV churn!
            - **Configurable INR Conversion:** Calculated at baseline ₹{config.business.usd_to_inr_rate:.2f}/USD.
            """
        )

    # Charts Grid
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Contract Type Churn Comparison")
        st.caption(
            "Month-to-month contracts demonstrate >8x higher churn propensity than 2-year contracts."
        )
        fig_contract = plot_churn_by_contract(df)
        st.plotly_chart(fig_contract, width="stretch")

    with col_right:
        st.subheader("Churn Hazard by Tenure Cohort")
        st.caption(
            "The first 12 months exhibit the steepest drop-off curve (first-year retention cliff)."
        )
        fig_tenure = plot_churn_by_tenure_bucket(df)
        st.plotly_chart(fig_tenure, width="stretch")
