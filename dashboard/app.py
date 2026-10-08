"""Main Streamlit Application for Telco Customer Churn & Retention Optimization Platform.

Single entry-point architecture ensuring consistent routing, session management, and state.
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

# Page Configuration
st.set_page_config(
    page_title="Telco Retention Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS (Modern Slate/Navy Theme, Accessible Contrast, Responsive Layout)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Primary Accent Buttons */
    .stButton > button {
        background: #4f46e5;
        color: #ffffff;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 0.55rem 1.25rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
    }
    .stButton > button:hover {
        background: #4338ca;
        border-color: rgba(255, 255, 255, 0.25);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35);
    }

    /* Form Fields and Inputs */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    input[type="text"], input[type="number"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
    }
    input[type="text"]:focus, input[type="number"]:focus {
        border-color: #6366f1 !important;
    }

    /* Tables & DataFrames */
    div[data-testid="stDataFrame"] {
        background-color: rgba(30, 41, 59, 0.45);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Custom Radio Navigation Styling */
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        padding: 6px 10px;
        border-radius: 6px;
        transition: background-color 0.15s ease;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: rgba(255, 255, 255, 0.04);
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    _ = load_config()

    # Sidebar Brand Header
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: linear-gradient(135deg, #4f46e5, #06b6d4); border-radius: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                        📡
                    </div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">TELCO CHURN</div>
                        <div style="font-size: 0.72rem; color: #818cf8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Retention Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>NAVIGATION</div>",
            unsafe_allow_html=True,
        )

        nav_options = [
            "🏠 Executive Overview",
            "🎯 Customer Prediction",
            "📊 Retention Prioritization",
            "🧠 Model Insights",
        ]

        # Query param or session state recovery
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
            "<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;' />",
            unsafe_allow_html=True,
        )

        # Verified Model Status Section
        render_sidebar_status()

    # Route to Selected View
    if "Executive Overview" in selected_page:
        render_executive_overview()
    elif "Customer Prediction" in selected_page:
        render_customer_prediction()
    elif "Retention Prioritization" in selected_page:
        render_retention_prioritization()
    elif "Model Insights" in selected_page:
        render_model_insights()


if __name__ == "__main__":
    main()
