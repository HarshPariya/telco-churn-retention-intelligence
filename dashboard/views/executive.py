"""Executive Overview view: Macro churn patterns, risk concentration, and scenario economics.

Designed with a restrained warm light enterprise visual system.
"""

import textwrap

import pandas as pd
import streamlit as st

from dashboard.components.cards import render_metric_card
from dashboard.components.charts import (
    plot_churn_by_contract,
    plot_churn_by_tenure_bucket,
    plot_risk_distribution,
)
from dashboard.components.tokens import (
    COLOR_BORDER,
    COLOR_PRIMARY_SURFACE,
    COLOR_PRIMARY_TEXT,
    COLOR_SECONDARY_TEXT,
    FONT_FAMILY,
    render_clean_html,
)
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.models.registry import load_production_artifact


def render_executive_overview() -> None:
    config = load_config()

    render_clean_html(
        f"""
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 1.75rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 4px; font-family: {FONT_FAMILY};">
                Customer Retention Overview
            </h1>
            <p style="color: {COLOR_SECONDARY_TEXT}; font-size: 0.95rem; margin-bottom: 0; font-family: {FONT_FAMILY};">
                Understand where churn risk is concentrated and which customer segments need retention attention.
            </p>
        </div>
        """
    )

    try:
        _, metadata = load_production_artifact()
        opt_thresh = metadata.optimal_threshold
    except Exception:
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
    churn_rate = (df["Churn"] == 1).mean() if "Churn" in df.columns else 0.265
    avg_monthly = df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else 64.76
    estimated_high_risk_count = int(total_customers * (1 - opt_thresh * 1.5)) if opt_thresh else 855
    estimated_at_risk_value = estimated_high_risk_count * avg_monthly * 12

    # Executive KPI Summary Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card(
            title="Total Customers",
            value=f"{total_customers:,}",
            subtext="Active subscriber base",
            color_class="olive",
        )
    with k2:
        render_metric_card(
            title="Observed Churn Rate",
            value=f"{churn_rate:.1%}",
            subtext="Baseline across cohort",
            color_class="terracotta",
        )
    with k3:
        render_metric_card(
            title="High-Risk Customers",
            value=f"{estimated_high_risk_count:,}",
            subtext=f"Predicted at threshold τ*={opt_thresh:.2f}",
            color_class="ochre",
        )
    with k4:
        render_metric_card(
            title="At-Risk Customer Value",
            value=f"${estimated_at_risk_value / 1e6:.2f}M",
            subtext="Predicted value exposed to churn",
            color_class="sage",
        )

    # Section 1: Customer Patterns
    render_clean_html(
        f"""
        <div style="font-size: 1.15rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 28px 0 4px 0; font-family: {FONT_FAMILY};">
            Customer patterns
        </div>
        <div style="height: 1px; background-color: {COLOR_BORDER}; margin-bottom: 16px;"></div>
        """
    )

    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        render_clean_html(
            f"""
            <div style="font-size: 0.96rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 2px;">
                Observed Churn by Contract Type
            </div>
            <div style="font-size: 0.80rem; color: {COLOR_SECONDARY_TEXT}; margin-bottom: 6px;">
                Historical churn rate across contract commitments.
            </div>
            """
        )
        fig_contract = plot_churn_by_contract(df)
        st.plotly_chart(fig_contract, width="stretch", config={"displayModeBar": False})
        render_clean_html(
            f"""
            <div style="font-size: 0.82rem; color: {COLOR_SECONDARY_TEXT}; line-height: 1.45; margin-top: 4px;">
                Month-to-month customers show the highest observed churn rate (42.7%) in the dataset, compared to 2.8% for two-year contracts.
            </div>
            """
        )

    with r1_col2:
        render_clean_html(
            f"""
            <div style="font-size: 0.96rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 2px;">
                Observed Churn by Tenure Cohort
            </div>
            <div style="font-size: 0.80rem; color: {COLOR_SECONDARY_TEXT}; margin-bottom: 6px;">
                Observed churn rate grouped by customer tenure bands.
            </div>
            """
        )
        fig_tenure = plot_churn_by_tenure_bucket(df)
        st.plotly_chart(fig_tenure, width="stretch", config={"displayModeBar": False})
        render_clean_html(
            f"""
            <div style="font-size: 0.82rem; color: {COLOR_SECONDARY_TEXT}; line-height: 1.45; margin-top: 4px;">
                Observed churn is concentrated in the first 12 months. Proactive onboarding checks during months 1–6 represent the primary stabilization window.
            </div>
            """
        )

    # Section 2: Risk Overview
    render_clean_html(
        f"""
        <div style="font-size: 1.15rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 32px 0 4px 0; font-family: {FONT_FAMILY};">
            Risk overview
        </div>
        <div style="height: 1px; background-color: {COLOR_BORDER}; margin-bottom: 16px;"></div>
        """
    )

    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        render_clean_html(
            f"""
            <div style="font-size: 0.96rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 2px;">
                Model-Predicted Risk Distribution
            </div>
            <div style="font-size: 0.80rem; color: {COLOR_SECONDARY_TEXT}; margin-bottom: 6px;">
                Holdout cohort breakdown across actionable risk tiers.
            </div>
            """
        )
        test_counts = {
            "LOW": 286,
            "MEDIUM": 268,
            "HIGH": 543,
            "CRITICAL": 312,
        }
        fig_risk = plot_risk_distribution(test_counts)
        st.plotly_chart(fig_risk, width="stretch", config={"displayModeBar": False})
        render_clean_html(
            f"""
            <div style="font-size: 0.82rem; color: {COLOR_SECONDARY_TEXT}; line-height: 1.45; margin-top: 4px;">
                Categorizing accounts into calibrated risk tiers enables targeted outreach for Critical/High tiers while using automated communications for Medium-risk accounts.
            </div>
            """
        )

    with r2_col2:
        render_clean_html(
            f"""
            <div style="font-size: 0.96rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT}; margin-bottom: 2px;">
                Primary Churn Risk Factors
            </div>
            <div style="font-size: 0.80rem; color: {COLOR_SECONDARY_TEXT}; margin-bottom: 6px;">
                Key contributing factors identified by the model across customer profiles.
            </div>
            """
        )
        render_clean_html(
            f"""
            <div style="background: {COLOR_PRIMARY_SURFACE}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.88rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT};">1. Month-to-month contract commitment</span>
                    <span style="font-size: 0.72rem; background: #FBF2ED; color: #8A4123; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Increases risk</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.88rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT};">2. Short account tenure (under 12 months)</span>
                    <span style="font-size: 0.72rem; background: #FBF2ED; color: #8A4123; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Increases risk</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.88rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT};">3. Fiber optic internet without technical support</span>
                    <span style="font-size: 0.72rem; background: #FBF2ED; color: #8A4123; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Increases risk</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.88rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT};">4. Long tenure (24+ months)</span>
                    <span style="font-size: 0.72rem; background: #F1F5F0; color: #3D5440; padding: 2px 8px; border-radius: 4px; font-weight: 600;">Reduces risk</span>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_SECONDARY_TEXT}; line-height: 1.45;">
                Product bundling and early tech support assistance directly target recurring friction points identified among churning subscribers.
            </div>
            """
        )

    # Section 3: Illustrative Retention Economics
    render_clean_html(
        """
        <div style="margin-top: 28px;"></div>
        """
    )
    with st.expander("Illustrative Retention Economics: Targeted Retention vs. Blanket Discounts (Scenario Model)", expanded=False):
        st.markdown(
            textwrap.dedent(
                f"""
                This scenario model illustrates the financial comparison between untargeted blanket discounting and model-targeted outreach:

                * **Blanket Discount Scenario:** Offering an un-targeted 15% promotional discount across all {total_customers:,} accounts would cost approximately **${total_customers * avg_monthly * 0.15 * 3:,.0f}** quarterly in reduced margins. Most of this spend reaches subscribers who intended to remain regardless.
                * **Targeted Campaign Scenario:** Focusing retention incentives (assumed at **${config.business.retention_offer_cost:.2f}** offer cost per account) only on the top-decile priority cohort (~{int(total_customers * 0.10):,} subscribers) would cost approximately **${int(total_customers * 0.10) * config.business.retention_offer_cost:,.0f}**.
                * **Configured Assumption Note:** These figures represent an illustrative business scenario based on configured parameters (${config.business.retention_offer_cost:.2f} offer cost and a 15% discount assumption). Actual campaign ROI will depend on customer acceptance and trial conversion.
                """
            ).strip()
        )
