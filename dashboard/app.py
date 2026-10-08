"""Main Streamlit Application for Telco Customer Churn & Retention Optimization Platform."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.pages.executive import render_executive_page
from dashboard.pages.model_insights import render_model_insights_page
from dashboard.pages.prediction import render_prediction_page
from dashboard.pages.prioritization import render_prioritization_page
from src.telco_churn.config import load_config
from src.telco_churn.models.registry import load_production_artifact

# Page Configuration
st.set_page_config(
    page_title="Telco Retention Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS (Modern Tailwind-inspired aesthetics, glassmorphism, fluid styling)
CUSTOM_CSS = """
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #090d16 100%);
        color: #f8fafc;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: #ffffff;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
    }

    /* Form and Inputs */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
    }
    input[type="text"], input[type="number"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }

    /* DataFrame Tables */
    div[data-testid="stDataFrame"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Header styling */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    config = load_config()

    # Sidebar Header & Brand
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <div style="background: linear-gradient(135deg, #4f46e5, #06b6d4); border-radius: 10px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 22px;">
                    📡
                </div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; line-height: 1.2;">TELCO CHURN</div>
                    <div style="font-size: 0.75rem; color: #818cf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Retention Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation Options
        page = st.radio(
            "Navigate Modules",
            options=[
                "Executive Overview",
                "Customer Prediction",
                "Retention Prioritization",
                "Model Insights",
            ],
            index=0,
        )

        st.markdown(
            "<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;' />",
            unsafe_allow_html=True,
        )

        # Model Status Badge
        try:
            _, meta = load_production_artifact()
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                    <div style="font-size: 0.72rem; color: #10b981; font-weight: 700; text-transform: uppercase;">● MODEL SERVING ONLINE</div>
                    <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc; margin-top: 4px;">{meta.model_name}</div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">Ver: {meta.model_version} | Thresh: {meta.optimal_threshold:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                """
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                    <div style="font-size: 0.72rem; color: #f59e0b; font-weight: 700; text-transform: uppercase;">⚠️ MODEL UNINITIALIZED</div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">Run python scripts/train_model.py</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style="font-size: 0.75rem; color: #64748b; line-height: 1.5;">
                <b>Platform:</b> Telco Track A<br/>
                <b>Currency Base:</b> USD ($)<br/>
                <b>Configured INR Rate:</b> ₹{config.business.usd_to_inr_rate:.2f}<br/>
                <b>Campaign Offer Cost:</b> ${config.business.retention_offer_cost:.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Route to selected page
    if page == "Executive Overview":
        render_executive_page()
    elif page == "Customer Prediction":
        render_prediction_page()
    elif page == "Retention Prioritization":
        render_prioritization_page()
    elif page == "Model Insights":
        render_model_insights_page()


if __name__ == "__main__":
    main()
