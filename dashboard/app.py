"""Main Streamlit Application for Telco Customer Churn & Retention Intelligence Platform.

Single entry-point enterprise architecture with a warm light enterprise visual system:
Warm ivory background (#F5F0E7), cream surfaces (#FFFDF8), dark espresso typography (#2D2924),
and muted olive (#5E6B4A) and terracotta (#A56B4F) accents.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.status import render_sidebar_status
from dashboard.components.tokens import (
    COLOR_BORDER,
    COLOR_PAGE_BG,
    COLOR_PRIMARY_BRAND,
    COLOR_PRIMARY_SURFACE,
    COLOR_PRIMARY_TEXT,
    COLOR_SECONDARY_SURFACE,
    COLOR_SECONDARY_TEXT,
)
from dashboard.views.executive import render_executive_overview
from dashboard.views.model_insights import render_model_insights
from dashboard.views.prediction import render_customer_prediction
from dashboard.views.prioritization import render_retention_prioritization
from src.telco_churn.config import load_config

# Page Configuration - Strictly Warm Light Enterprise Visual System
st.set_page_config(
    page_title="Telco Churn - Retention Intelligence",
    page_icon="TC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [class*="st-"] {{
        font-family: 'Inter', ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }}

    /* Warm Ivory Page Background & Espresso Typography */
    .stApp {{
        background-color: {COLOR_PAGE_BG} !important;
        color: {COLOR_PRIMARY_TEXT} !important;
    }}

    /* Warm Enterprise Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #FBF8F2 !important;
        border-right: 1px solid {COLOR_BORDER} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
        padding-top: 1.5rem;
    }}

    /* Primary Action Buttons (Clean Enterprise Green) */
    button,
    button[kind="primary"],
    button[kind="secondary"],
    .stButton > button,
    .stButton button,
    .stDownloadButton > button,
    .stDownloadButton button,
    .stFormSubmitButton > button,
    .stFormSubmitButton button,
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {{
        background-color: #15803D !important;
        border: 1px solid #166534 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.35rem !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10) !important;
        cursor: pointer !important;
    }}

    button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    .stButton > button:hover,
    .stButton button:hover,
    .stDownloadButton > button:hover,
    .stDownloadButton button:hover,
    .stFormSubmitButton > button:hover,
    .stFormSubmitButton button:hover,
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #166534 !important;
        border-color: #14532D !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
    }}

    button:active,
    button[kind="primary"]:active,
    button[kind="secondary"]:active,
    .stButton > button:active,
    .stButton button:active,
    .stDownloadButton > button:active,
    .stDownloadButton button:active,
    .stFormSubmitButton > button:active,
    .stFormSubmitButton button:active,
    div[data-testid="stButton"] > button:active,
    div[data-testid="stDownloadButton"] > button:active,
    div[data-testid="stFormSubmitButton"] > button:active {{
        background-color: #14532D !important;
        transform: translateY(0);
    }}

    /* High-contrast crisp white typography inside ALL buttons */
    button,
    button *,
    button p,
    button span,
    button div,
    .stButton button,
    .stButton button *,
    .stButton button p,
    .stButton button span,
    .stButton button div,
    .stButton div[data-testid="stMarkdownContainer"] p,
    .stDownloadButton button,
    .stDownloadButton button *,
    .stDownloadButton button p,
    .stDownloadButton button span,
    .stDownloadButton button div,
    .stDownloadButton div[data-testid="stMarkdownContainer"] p,
    .stFormSubmitButton button,
    .stFormSubmitButton button *,
    .stFormSubmitButton button p,
    .stFormSubmitButton button span,
    .stFormSubmitButton button div,
    .stFormSubmitButton div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stButton"] button p,
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stFormSubmitButton"] button p {{
        color: #FFFFFF !important;
        font-family: 'Inter', ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        line-height: 1.4 !important;
        text-shadow: 0 1px 1px rgba(0, 0, 0, 0.20) !important;
        -webkit-font-smoothing: antialiased !important;
    }}

    /* Warm Inputs and Selects */
    div[data-baseweb="select"] > div {{
        background-color: {COLOR_PRIMARY_SURFACE} !important;
        border-color: {COLOR_BORDER} !important;
        color: {COLOR_PRIMARY_TEXT} !important;
        border-radius: 8px !important;
    }}
    input[type="text"], input[type="number"] {{
        background-color: {COLOR_PRIMARY_SURFACE} !important;
        color: {COLOR_PRIMARY_TEXT} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 8px !important;
    }}
    input[type="text"]:focus, input[type="number"]:focus {{
        border-color: #15803D !important;
        box-shadow: 0 0 0 1px #15803D !important;
    }}

    /* DataFrames & Tables */
    div[data-testid="stDataFrame"] {{
        background-color: {COLOR_PRIMARY_SURFACE};
        border-radius: 8px;
        border: 1px solid {COLOR_BORDER};
    }}

    /* Clean Enterprise Navigation */
    div[data-testid="stRadio"] > div {{
        gap: 4px;
    }}
    div[data-testid="stRadio"] label {{
        padding: 7px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 500;
        color: {COLOR_SECONDARY_TEXT};
        transition: all 0.15s ease;
    }}
    div[data-testid="stRadio"] label:hover {{
        background-color: {COLOR_SECONDARY_SURFACE};
        color: {COLOR_PRIMARY_TEXT};
    }}

    /* Typography (Scoped so buttons are never overridden by dark text) */
    h1, h2, h3, h4 {{
        color: {COLOR_PRIMARY_TEXT} !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    p:not(button *):not(.stButton *):not(.stDownloadButton *):not(.stFormSubmitButton *),
    span:not(button *):not(.stButton *):not(.stDownloadButton *):not(.stFormSubmitButton *),
    label:not(button *):not(.stButton *):not(.stDownloadButton *):not(.stFormSubmitButton *) {{
        color: {COLOR_PRIMARY_TEXT};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    _ = load_config()

    # Sidebar Enterprise Brand Header
    with st.sidebar:
        st.markdown(
            f"""
            <div style="margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background-color: {COLOR_PRIMARY_BRAND}; border-radius: 6px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; color: #FFFDF8; font-weight: 800; font-size: 14px; letter-spacing: -0.02em;">
                        TC
                    </div>
                    <div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: {COLOR_PRIMARY_TEXT}; letter-spacing: -0.01em; line-height: 1.2;">TELCO CHURN</div>
                        <div style="font-size: 0.70rem; color: {COLOR_SECONDARY_TEXT}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Retention Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size: 0.72rem; color: {COLOR_SECONDARY_TEXT}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>NAVIGATION</div>",
            unsafe_allow_html=True,
        )

        nav_options = [
            "Executive Overview",
            "Customer Prediction",
            "Retention Prioritization",
            "Model Insights",
        ]

        # Query param recovery for deep-linking
        query_view = st.query_params.get("view", None)
        default_index = 0
        if query_view:
            for idx, opt in enumerate(nav_options):
                if query_view.lower() in opt.lower():
                    default_index = idx
                    break

        selected_page = st.radio(
            label="Navigation Menu",
            options=nav_options,
            index=default_index,
            label_visibility="collapsed",
        )

        st.markdown(
            f"<div style='height: 1px; background-color: {COLOR_BORDER}; margin: 20px 0;'></div>",
            unsafe_allow_html=True,
        )

        # Verified Model Status Section
        render_sidebar_status()

    # Route to Selected View (Deterministic single-source dispatch)
    if selected_page == "Executive Overview":
        render_executive_overview()
    elif selected_page == "Customer Prediction":
        render_customer_prediction()
    elif selected_page == "Retention Prioritization":
        render_retention_prioritization()
    elif selected_page == "Model Insights":
        render_model_insights()


if __name__ == "__main__":
    main()
