"""Reusable enterprise UI card, badge, and empty state components.

Engineered with a warm light enterprise visual system:
Ivory/Cream surfaces, dark espresso typography, warm neutral borders,
and semantic muted olive / terracotta / ochre accents.
"""

from typing import Optional

from dashboard.components.tokens import (
    COLOR_BORDER,
    COLOR_PRIMARY_BRAND,
    COLOR_PRIMARY_SURFACE,
    COLOR_PRIMARY_TEXT,
    COLOR_RISK_CRITICAL,
    COLOR_RISK_HIGH,
    COLOR_RISK_LOW,
    COLOR_RISK_MEDIUM,
    COLOR_SECONDARY_ACCENT,
    COLOR_SECONDARY_TEXT,
    COLOR_TERTIARY_TEXT,
    RISK_BADGE_CONFIG,
    render_clean_html,
)


def render_metric_card(
    title: str,
    value: str,
    subtext: str = "",
    trend: str = "neutral",
    color_class: str = "olive",
    accent_bar: bool = True,
) -> None:
    """Render a clean, warm enterprise KPI card."""
    accent_map = {
        "olive": COLOR_PRIMARY_BRAND,
        "terracotta": COLOR_SECONDARY_ACCENT,
        "sage": COLOR_RISK_LOW,
        "ochre": COLOR_RISK_MEDIUM,
        "high": COLOR_RISK_HIGH,
        "critical": COLOR_RISK_CRITICAL,
        # Backward-compat aliases mapped to warm enterprise palette
        "blue": COLOR_PRIMARY_BRAND,
        "indigo": COLOR_PRIMARY_BRAND,
        "rose": COLOR_RISK_CRITICAL,
        "emerald": COLOR_RISK_LOW,
        "amber": COLOR_RISK_MEDIUM,
    }
    accent_color = accent_map.get(color_class, COLOR_PRIMARY_BRAND)

    border_top_css = f"border-top: 3px solid {accent_color};" if accent_bar else ""

    html = f"""
    <div style="background: {COLOR_PRIMARY_SURFACE}; border: 1px solid {COLOR_BORDER}; {border_top_css} border-radius: 10px; padding: 14px 16px; box-shadow: 0 3px 12px rgba(45, 41, 36, 0.04); margin-bottom: 12px; min-height: 105px;">
        <div style="font-size: 0.72rem; color: {COLOR_SECONDARY_TEXT}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">
            {title}
        </div>
        <div style="font-size: 1.75rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin: 3px 0 4px 0; line-height: 1.2;">
            {value}
        </div>
        <div style="font-size: 0.78rem; color: {COLOR_SECONDARY_TEXT};">
            {subtext}
        </div>
    </div>
    """
    render_clean_html(html)


def render_risk_badge(
    risk_level: str,
    probability: float,
    clv: Optional[float] = None,
    priority: Optional[float] = None,
) -> None:
    """Render an accessible, soft warm risk tier summary banner."""
    style = RISK_BADGE_CONFIG.get(
        risk_level,
        {
            "bg": "#F7F3EC",
            "border": COLOR_BORDER,
            "text": COLOR_PRIMARY_TEXT,
            "label": risk_level,
            "desc": "Estimated churn assessment.",
        },
    )

    clv_html = ""
    if clv is not None and priority is not None:
        clv_html = f"""
        <div style="border-left: 1px solid {style['border']}; padding-left: 18px; margin-left: 18px;">
            <div style="font-size: 0.72rem; color: {COLOR_SECONDARY_TEXT}; text-transform: uppercase; font-weight: 600;">Customer Value (CLV)</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; margin-top: 2px;">${clv:,.2f}</div>
            <div style="font-size: 0.74rem; color: {COLOR_SECONDARY_TEXT}; margin-top: 1px;">Priority Score: <b>{priority:,.1f}</b></div>
        </div>
        """

    html = f"""
    <div style="background: {style['bg']}; border: 1px solid {style['border']}; border-radius: 10px; padding: 16px 20px; margin-bottom: 18px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;">
        <div style="flex: 1; min-width: 200px;">
            <div style="font-size: 0.80rem; font-weight: 700; color: {style['text']}; letter-spacing: 0.05em;">{style['label']}</div>
            <div style="font-size: 0.84rem; color: {COLOR_SECONDARY_TEXT}; margin-top: 3px;">{style['desc']}</div>
        </div>
        <div style="display: flex; align-items: center; text-align: right;">
            <div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {style['text']}; line-height: 1;">{probability:.1%}</div>
                <div style="font-size: 0.72rem; color: {COLOR_SECONDARY_TEXT}; text-transform: uppercase; font-weight: 600; margin-top: 3px;">Estimated Churn Probability</div>
            </div>
            {clv_html}
        </div>
    </div>
    """
    render_clean_html(html)


def render_driver_card(feature: str, direction: str, impact: float, rank: int) -> None:
    """Render a restrained warm card displaying an individual contributing factor."""
    is_risk = direction == "INCREASES_CHURN"
    border_color = COLOR_RISK_HIGH if is_risk else COLOR_RISK_LOW
    badge_bg = "#FBF2ED" if is_risk else "#F1F5F0"
    badge_text = "#8A4123" if is_risk else "#3D5440"
    direction_label = "Increases predicted risk" if is_risk else "Reduces predicted risk"

    html = f"""
    <div style="background: {COLOR_PRIMARY_SURFACE}; border: 1px solid {COLOR_BORDER}; border-left: 3px solid {border_color}; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 0.70rem; color: {COLOR_SECONDARY_TEXT}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">Factor #{rank}</div>
            <div style="font-size: 0.94rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT}; margin-top: 1px;">{feature}</div>
        </div>
        <div style="text-align: right;">
            <span style="background: {badge_bg}; color: {badge_text}; font-size: 0.75rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; display: inline-block;">{direction_label}</span>
            <div style="font-size: 0.72rem; color: {COLOR_TERTIARY_TEXT}; margin-top: 3px;">Relative impact: +{impact:.3f}</div>
        </div>
    </div>
    """
    render_clean_html(html)


def render_empty_state(
    message: str = "",
    subtext: str = "",
    *,
    title: str = "",
    description: str = "",
) -> None:
    """Render a restrained warm empty state container."""
    display_title = message or title or "No data available"
    display_subtext = subtext or description or ""

    html = f"""
    <div style="background: {COLOR_PRIMARY_SURFACE}; border: 1px dashed {COLOR_BORDER}; border-radius: 10px; padding: 32px 24px; text-align: center; margin: 16px 0;">
        <div style="font-size: 0.96rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT};">{display_title}</div>
        <div style="font-size: 0.84rem; color: {COLOR_SECONDARY_TEXT}; margin-top: 4px;">{display_subtext}</div>
    </div>
    """
    render_clean_html(html)
