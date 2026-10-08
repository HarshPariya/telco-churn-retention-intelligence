"""Main Streamlit Application for Telco Customer Churn & Retention Intelligence Platform.

Single entry-point enterprise architecture ensuring consistent routing, state management,
and an accessible, professional light visual theme.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.status import render_sidebar_status
from dashboard.views.executive import render_executive_overview
from dashboard.views.model_insights import render_model_insights
from dashboard.views.prediction import render_customer_prediction
from dashboard.views.prioritization import render_retention_prioritization
from src.telco_churn.config import load_config

# Page Configuration - Strictly Light Enterprise Visual System
st.set_page_config(
    page_title="Telco Churn - Retention Intelligence",
    page_icon="TC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS (Enterprise Light Theme, High Contrast, Accessible Spacing)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Professional Light Theme Page Background and Typography */
    .stApp {
        background-color: #F5F7FA;
        color: #172033;
    }

    /* Enterprise Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem;
    }

    /* Primary Buttons (Restrained Corporate Blue) */
    .stButton > button {
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid #1D4ED8;
        border-radius: 6px;
        padding: 0.5rem 1.25rem;
        transition: all 0.15s ease-in-out;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
        border-color: #1E40AF;
        color: #FFFFFF;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.08);
    }
    .stButton > button:active {
        background-color: #1E40AF;
        transform: translateY(0);
    }

    /* Form Fields and Inputs */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #172033 !important;
        border-radius: 6px !important;
    }
    input[type="text"], input[type="number"] {
        background-color: #FFFFFF !important;
        color: #172033 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }
    input[type="text"]:focus, input[type="number"]:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 1px #2563EB !important;
    }

    /* Tables & DataFrames */
    div[data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
    }

    /* Clean Enterprise Radio Navigation */
    div[data-testid="stRadio"] > div {
        gap: 4px;
    }
    div[data-testid="stRadio"] label {
        padding: 7px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #334155;
        transition: all 0.15s ease;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #F1F5F9;
        color: #0F172A;
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #172033 !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    p, span, label {
        color: #172033;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    _ = load_config()

    # Sidebar Enterprise Brand Header
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background-color: #2563EB; border-radius: 6px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-weight: 800; font-size: 15px; letter-spacing: -0.02em;">
                        TC
                    </div>
                    <div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #172033; letter-spacing: -0.01em; line-height: 1.2;">TELCO CHURN</div>
                        <div style="font-size: 0.70rem; color: #5B6577; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Retention Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.72rem; color: #5B6577; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>NAVIGATION</div>",
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
            "<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;' />",
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
