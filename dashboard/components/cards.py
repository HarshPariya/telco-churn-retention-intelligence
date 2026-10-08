"""Reusable enterprise UI card and badge components with light theme styling."""

import streamlit as st


def render_metric_card(
    title: str,
    value: str,
    subtext: str = "",
    trend: str = "neutral",
    color_class: str = "blue",
) -> None:
    """Render a clean enterprise KPI card with light styling."""
    accent_map = {
        "blue": "#2563EB",
        "indigo": "#4F46E5",
        "rose": "#B91C1C",
        "emerald": "#15803D",
        "amber": "#B45309",
    }
    accent_color = accent_map.get(color_class, "#2563EB")

    html = f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    ">
        <div style="font-size: 0.76rem; color: #5B6577; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">
            {title}
        </div>
        <div style="font-size: 1.85rem; font-weight: 700; color: #172033; margin: 4px 0 6px 0; line-height: 1.2;">
            {value}
        </div>
        <div style="font-size: 0.80rem; color: #5B6577; display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: {accent_color};"></span>
            {subtext}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_risk_badge(risk_level: str, probability: float) -> None:
    """Render a clear, accessible risk tier banner with explicit text and probability."""
    badge_styles = {
        "CRITICAL": {
            "bg": "#FEF2F2",
            "border": "#FCA5A5",
            "text": "#B91C1C",
            "label": "CRITICAL RISK",
            "desc": "High probability of service cancellation. Immediate intervention required.",
        },
        "HIGH": {
            "bg": "#FFF7ED",
            "border": "#FDBA74",
            "text": "#C2410C",
            "label": "HIGH RISK",
            "desc": "Elevated churn risk exceeding operational decision threshold.",
        },
        "MEDIUM": {
            "bg": "#FEFCE8",
            "border": "#FDE047",
            "text": "#854D0E",
            "label": "MEDIUM RISK",
            "desc": "Moderate churn propensity. Suitable for automated digital nurture.",
        },
        "LOW": {
            "bg": "#F0FDF4",
            "border": "#86EFAC",
            "text": "#15803D",
            "label": "LOW RISK",
            "desc": "Stable subscriber account. Maintain standard service relationship.",
        },
    }
    style = badge_styles.get(
        risk_level,
        {
            "bg": "#F8FAFC",
            "border": "#E2E8F0",
            "text": "#475569",
            "label": risk_level,
            "desc": "Estimated churn propensity.",
        },
    )

    html = f"""
    <div style="
        background: {style['bg']};
        border: 1px solid {style['border']};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div>
            <div style="font-size: 0.78rem; font-weight: 700; color: {style['text']}; letter-spacing: 0.05em;">
                {style['label']}
            </div>
            <div style="font-size: 0.84rem; color: #5B6577; margin-top: 2px;">
                {style['desc']}
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 2.1rem; font-weight: 800; color: {style['text']}; line-height: 1;">
                {probability:.1%}
            </div>
            <div style="font-size: 0.74rem; color: #5B6577; text-transform: uppercase; font-weight: 600; margin-top: 2px;">
                Estimated Churn Probability
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_driver_card(feature: str, direction: str, impact: float, rank: int) -> None:
    """Render a clean card displaying a single contributing risk driver."""
    is_risk = direction == "INCREASES_CHURN"
    border_color = "#B91C1C" if is_risk else "#15803D"
    badge_bg = "#FEF2F2" if is_risk else "#F0FDF4"
    badge_text = "#B91C1C" if is_risk else "#15803D"
    direction_label = "Increases predicted churn risk" if is_risk else "Protects retention (reduces risk)"

    html = f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 3px solid {border_color};
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div>
            <div style="font-size: 0.72rem; color: #5B6577; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">
                Factor #{rank}
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #172033; margin-top: 1px;">
                {feature}
            </div>
        </div>
        <div style="text-align: right;">
            <span style="
                background: {badge_bg};
                color: {badge_text};
                font-size: 0.76rem;
                font-weight: 600;
                padding: 3px 8px;
                border-radius: 4px;
                display: inline-block;
            ">
                {direction_label}
            </span>
            <div style="font-size: 0.72rem; color: #5B6577; margin-top: 3px;">
                Relative impact: +{impact:.3f}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state(message: str, subtext: str = "") -> None:
    """Render a standard clean empty state container."""
    html = f"""
    <div style="
        background: #FFFFFF;
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        padding: 36px 24px;
        text-align: center;
        margin: 16px 0;
    ">
        <div style="font-size: 0.98rem; font-weight: 600; color: #172033;">
            {message}
        </div>
        <div style="font-size: 0.85rem; color: #5B6577; margin-top: 4px;">
            {subtext}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
