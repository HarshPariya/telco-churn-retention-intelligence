"""Sidebar status component with enterprise light styling."""

import streamlit as st

from src.telco_churn.models.registry import load_production_artifact


def render_sidebar_status() -> None:
    """Render compact, verified model status in sidebar."""
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #5B6577; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
            MODEL STATUS
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        _, meta = load_production_artifact()
        st.markdown(
            f"""
            <div style="
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                padding: 10px 12px;
                margin-bottom: 16px;
            ">
                <div style="font-size: 0.80rem; color: #15803D; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background-color: #15803D;"></span>
                    Healthy
                </div>
                <div style="font-size: 0.84rem; font-weight: 600; color: #172033; margin-top: 3px;">
                    {meta.model_name}
                </div>
                <div style="font-size: 0.76rem; color: #5B6577; margin-top: 2px;">
                    Version: <b>{meta.model_version}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            """
            <div style="
                background: #FFFBEB;
                border: 1px solid #FDE68A;
                border-radius: 6px;
                padding: 10px 12px;
                margin-bottom: 16px;
            ">
                <div style="font-size: 0.80rem; color: #B45309; font-weight: 700;">
                    ⚠️ Unavailable
                </div>
                <div style="font-size: 0.76rem; color: #5B6577; margin-top: 3px;">
                    Model artifact not initialized.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #5B6577; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
            ABOUT
        </div>
        <div style="font-size: 0.78rem; color: #5B6577; line-height: 1.45;">
            Predict customer churn, understand the main risk drivers, and prioritize high-value customers for retention.
        </div>
        """,
        unsafe_allow_html=True,
    )
