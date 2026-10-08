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
    COLOR_PRIMARY_BRAND_DARK,
    COLOR_PRIMARY_SURFACE,
    COLOR_PRIMARY_TEXT,
    COLOR_SECONDARY_SURFACE,
    COLOR_SECONDARY_TEXT,
    FONT_FAMILY,
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
    html, body, [class*="css"] {{
        font-family: {FONT_FAMILY};
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

    /* Primary Action Buttons (Muted Olive) */
    .stButton > button {{
        background-color: {COLOR_PRIMARY_BRAND} !important;
        color: #FFFDF8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: 1px solid {COLOR_PRIMARY_BRAND_DARK} !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 3px rgba(45, 41, 36, 0.08) !important;
    }}
    .stButton > button:hover {{
        background-color: {COLOR_PRIMARY_BRAND_DARK} !important;
        border-color: #38422C !important;
        color: #FFFDF8 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(45, 41, 36, 0.12) !important;
    }}
    .stButton > button:active {{
        background-color: #38422C !important;
        transform: translateY(0);
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
        border-color: {COLOR_PRIMARY_BRAND} !important;
        box-shadow: 0 0 0 1px {COLOR_PRIMARY_BRAND} !important;
    }}

    /* Prevent washed-out dimming during script reruns */
    div[data-testid="stAppViewBlockContainer"] {{
        opacity: 1 !important;
        transition: none !important;
    }}
    .stApp [data-testid="stAppViewBlockContainer"] {{
        opacity: 1 !important;
    }}
    div[data-testid="stAppViewContainer"] {{
        opacity: 1 !important;
    }}

    /* High-Contrast Crisp Multiselect Tags */
    div[data-baseweb="tag"] {{
        background-color: #EAE3D6 !important;
        border: 1px solid #CFC4B4 !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }}
    div[data-baseweb="tag"] span {{
        color: {COLOR_PRIMARY_TEXT} !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }}
    div[data-baseweb="tag"] svg {{
        fill: {COLOR_PRIMARY_TEXT} !important;
    }}

    /* DataFrames & Tables */
    div[data-testid="stDataFrame"] {{
        background-color: {COLOR_PRIMARY_SURFACE} !important;
        border-radius: 8px !important;
        border: 1px solid {COLOR_BORDER} !important;
        box-shadow: 0 1px 3px rgba(45, 41, 36, 0.04) !important;
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

    /* Typography */
    h1, h2, h3, h4 {{
        color: {COLOR_PRIMARY_TEXT} !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    p, span, label {{
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
