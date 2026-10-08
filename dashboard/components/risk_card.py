"""Reusable UI card components and metric pills with modern design styling."""

import streamlit as st


def render_metric_card(
    title: str, value: str, subtext: str = "", trend: str = "neutral", color_class: str = "indigo"
) -> None:
    """Render a premium glassmorphic KPI metric card."""
    color_map = {
        "indigo": ("#6366f1", "rgba(99, 102, 241, 0.1)"),
        "rose": ("#f43f5e", "rgba(244, 63, 94, 0.1)"),
        "emerald": ("#10b981", "rgba(16, 185, 129, 0.1)"),
        "amber": ("#f59e0b", "rgba(245, 158, 11, 0.1)"),
        "cyan": ("#06b6d4", "rgba(6, 182, 212, 0.1)"),
    }
    primary_col, bg_col = color_map.get(color_class, ("#6366f1", "rgba(99, 102, 241, 0.1)"))

    html = f"""
    <div style="
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    ">
        <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
        <div style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 6px 0;">{value}</div>
        <div style="font-size: 0.82rem; color: #cbd5e1; display: flex; align-items: center; gap: 4px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {primary_col}; margin-right: 6px;"></span>
            {subtext}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_risk_badge(risk_level: str, probability: float) -> None:
    """Render a dynamic risk classification pill badge."""
    badge_styles = {
        "CRITICAL": ("#ef4444", "rgba(239, 68, 68, 0.15)", "🔥 Critical Churn Risk"),
        "HIGH": ("#f59e0b", "rgba(245, 158, 11, 0.15)", "⚠️ High Churn Risk"),
        "MEDIUM": ("#3b82f6", "rgba(59, 130, 246, 0.15)", "⚡ Moderate Risk"),
        "LOW": ("#10b981", "rgba(16, 185, 129, 0.15)", "✅ Low Churn Risk"),
    }
    border_col, bg_col, label = badge_styles.get(
        risk_level, ("#94a3b8", "rgba(148, 163, 184, 0.15)", risk_level)
    )

    html = f"""
    <div style="
        background: {bg_col};
        border: 1.5px solid {border_col};
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    ">
        <span style="font-size: 1.25rem; font-weight: 700; color: {border_col}; text-transform: uppercase; letter-spacing: 0.05em;">
            {label}
        </span>
        <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-top: 4px;">
            {probability:.1%}
        </div>
        <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 2px;">
            Calibrated Churn Propensity
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_driver_card(feature: str, direction: str, impact: float, rank: int) -> None:
    """Render a clean card displaying a single SHAP driver."""
    is_increase = direction == "INCREASES_CHURN"
    arrow = "🔺 Escalates Risk" if is_increase else "🔻 Protects Retention"
    color = "#f43f5e" if is_increase else "#10b981"
    bg = "rgba(244, 63, 94, 0.08)" if is_increase else "rgba(16, 185, 129, 0.08)"

    html = f"""
    <div style="
        background: {bg};
        border-left: 4px solid {color};
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div>
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">DRIVER #{rank}</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">{feature}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; font-weight: 600; color: {color};">{arrow}</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Impact: +{impact:.3f}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
