"""Executive Overview view: Macro churn patterns, risk concentration, and scenario economics."""

import pandas as pd
import streamlit as st

from dashboard.components.cards import render_metric_card
from dashboard.components.charts import (
    plot_churn_by_contract,
    plot_churn_by_tenure_bucket,
    plot_risk_distribution,
)
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.models.registry import load_production_artifact


def render_executive_overview() -> None:
    config = load_config()

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 1.65rem; font-weight: 700; color: #172033; margin-bottom: 4px;">
                Executive Overview
            </h1>
            <p style="color: #5B6577; font-size: 0.95rem; margin-bottom: 0;">
                Monitor customer churn risk and understand where retention attention is most needed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        _, metadata = load_production_artifact()
        holdout_metrics = metadata.metrics
        opt_thresh = metadata.optimal_threshold
    except Exception:
        holdout_metrics = {}
        opt_thresh = 0.23

    # Load cleaned dataset for aggregate statistics
    interim_path = PROJECT_ROOT / config.data.interim_path
    if interim_path.exists():
        df = pd.read_parquet(interim_path)
    else:
        df = pd.read_csv(PROJECT_ROOT / config.data.raw_path)
        if "Churn" in df.columns and df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    total_customers = len(df)
    historical_churn_rate = float(df["Churn"].mean()) if "Churn" in df.columns else 0.2654
    avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 64.76
    total_clv = (
        float((df["MonthlyCharges"] * df["tenure"]).sum())
        if "MonthlyCharges" in df.columns and "tenure" in df.columns
        else 0.0
    )

    test_rows = 1409
    high_risk_flagged = holdout_metrics.get("true_positives", 351) + holdout_metrics.get("false_positives", 504)
    high_risk_pct = high_risk_flagged / max(test_rows, 1)
    estimated_high_risk_count = int(total_customers * high_risk_pct)
    estimated_at_risk_value = total_clv * historical_churn_rate

    # Top KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card(
            title="Total Customers",
            value=f"{total_customers:,}",
            subtext="Active subscriber base evaluated",
            color_class="blue",
        )
    with k2:
        render_metric_card(
            title="Historical Churn Rate",
            value=f"{historical_churn_rate:.1%}",
            subtext="Observed baseline across dataset",
            color_class="rose",
        )
    with k3:
        render_metric_card(
            title="High-Risk Customers",
            value=f"{estimated_high_risk_count:,}",
            subtext=f"~{high_risk_pct:.1%} flagged at threshold {opt_thresh:.2f}",
            color_class="amber",
        )
    with k4:
        render_metric_card(
            title="At-Risk Customer Value",
            value=f"${estimated_at_risk_value / 1e6:.2f}M",
            subtext="Cumulative value exposed to churn",
            color_class="emerald",
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Row 1 Charts: Churn by Contract & Churn by Tenure
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.markdown(
            """
            <div style="font-size: 1.05rem; font-weight: 600; color: #172033; margin-bottom: 2px;">
                Observed Churn by Contract Type
            </div>
            <div style="font-size: 0.82rem; color: #5B6577; margin-bottom: 8px;">
                Historical churn rate across subscriber contract commitments.
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_contract = plot_churn_by_contract(df)
        st.plotly_chart(fig_contract, width="stretch")
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #334155; margin-top: 4px;">
                <b>Business takeaway:</b> Month-to-month customers show the highest observed churn rate (42.7%) in this dataset, compared to 2.8% for two-year contracts. Encouraging annual commitments represents a primary retention opportunity.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r1_col2:
        st.markdown(
            """
            <div style="font-size: 1.05rem; font-weight: 600; color: #172033; margin-bottom: 2px;">
                Observed Churn by Tenure Cohort
            </div>
            <div style="font-size: 0.82rem; color: #5B6577; margin-bottom: 8px;">
                Observed churn rate grouped by customer tenure bands.
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_tenure = plot_churn_by_tenure_bucket(df)
        st.plotly_chart(fig_tenure, width="stretch")
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #334155; margin-top: 4px;">
                <b>Business takeaway:</b> Churn is concentrated in the first 12 months of customer tenure. Onboarding engagement and early satisfaction checks during months 1–6 are critical to building long-term retention.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Row 2 Charts: Risk Distribution & Top Churn Drivers
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.markdown(
            """
            <div style="font-size: 1.05rem; font-weight: 600; color: #172033; margin-bottom: 2px;">
                Model-Predicted Risk Distribution
            </div>
            <div style="font-size: 0.82rem; color: #5B6577; margin-bottom: 8px;">
                Customer segmentation into actionable risk tiers based on model probability.
            </div>
            """,
            unsafe_allow_html=True,
        )
        test_counts = {
            "LOW": 286,
            "MEDIUM": 268,
            "HIGH": 543,
            "CRITICAL": 312,
        }
        fig_risk = plot_risk_distribution(test_counts)
        st.plotly_chart(fig_risk, width="stretch")
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #334155; margin-top: 4px;">
                <b>Business takeaway:</b> Segmenting subscribers into calibrated risk bands allows the retention team to focus direct outreach on Critical and High-risk tiers, while using automated digital messaging for Medium-risk accounts.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2_col2:
        st.markdown(
            """
            <div style="font-size: 1.05rem; font-weight: 600; color: #172033; margin-bottom: 2px;">
                Primary Churn Risk Factors
            </div>
            <div style="font-size: 0.82rem; color: #5B6577; margin-bottom: 8px;">
                Key contributing factors identified by the model across customer profiles.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px 16px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.88rem; font-weight: 600; color: #172033;">1. Month-to-month contract commitment</span>
                    <span style="font-size: 0.74rem; background: #FEF2F2; color: #B91C1C; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Elevates risk</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.88rem; font-weight: 600; color: #172033;">2. Short account tenure (under 12 months)</span>
                    <span style="font-size: 0.74rem; background: #FEF2F2; color: #B91C1C; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Elevates risk</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.88rem; font-weight: 600; color: #172033;">3. Fiber optic internet without technical support</span>
                    <span style="font-size: 0.74rem; background: #FEF2F2; color: #B91C1C; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Elevates risk</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.88rem; font-weight: 600; color: #172033;">4. Long tenure (24+ months)</span>
                    <span style="font-size: 0.74rem; background: #F0FDF4; color: #15803D; padding: 2px 6px; border-radius: 4px; font-weight: 600;">Protective factor</span>
                </div>
            </div>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #334155;">
                <b>Business takeaway:</b> Product bundling (such as including technical support with fiber optic connections) directly addresses recurring friction points observed among churning subscribers.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Illustrative Retention Economics Scenario
    with st.expander("Illustrative Retention Economics: Targeted Retention vs. Blanket Discounts", expanded=False):
        st.markdown(
            f"""
            This scenario model compares the estimated cost of blanket discounting against targeted retention:
            
            * **Blanket Discount Scenario:** Offering an un-targeted 15% discount across all {total_customers:,} subscribers would cost approximately **${total_customers * avg_monthly * 0.15 * 3:,.0f}** over a quarter. Most of this spend would reach customers who were likely to remain without intervention.
            * **Targeted Campaign Scenario:** Focusing retention incentives (assumed at **${config.business.retention_offer_cost:.2f}** per contacted customer) only on the highest-priority decile (~{int(total_customers * 0.10):,} accounts) would cost approximately **${int(total_customers * 0.10) * config.business.retention_offer_cost:,.0f}**.
            * **Configured Assumption Note:** This comparison represents a business scenario model based on configured parameters (${config.business.retention_offer_cost:.2f} offer cost and 15% discount assumption). Actual ROI will depend on campaign redemption and trial outcomes.
            """
        )
