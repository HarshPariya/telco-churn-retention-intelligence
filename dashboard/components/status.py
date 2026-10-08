"""Sidebar status component with warm light enterprise styling."""

import streamlit as st

from dashboard.components.tokens import (
    COLOR_BORDER,
    COLOR_PRIMARY_SURFACE,
    COLOR_PRIMARY_TEXT,
    COLOR_RISK_LOW,
    COLOR_SECONDARY_TEXT,
)
from src.telco_churn.models.registry import load_production_artifact


def render_sidebar_status() -> None:
    """Render compact, verified model status in sidebar."""
    st.markdown(
        f"""
        <div style="font-size: 0.72rem; color: {COLOR_SECONDARY_TEXT}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
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
                background: {COLOR_PRIMARY_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 10px 12px;
                margin-bottom: 16px;
            ">
                <div style="font-size: 0.80rem; color: {COLOR_RISK_LOW}; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background-color: {COLOR_RISK_LOW};"></span>
                    Ready
                </div>
                <div style="font-size: 0.84rem; font-weight: 600; color: {COLOR_PRIMARY_TEXT}; margin-top: 3px;">
                    {meta.model_name}
                </div>
                <div style="font-size: 0.76rem; color: {COLOR_SECONDARY_TEXT}; margin-top: 2px;">
                    Version: <b>{meta.model_version}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            f"""
            <div style="
                background: #FDF9F3;
                border: 1px solid #E6D0AA;
                border-radius: 8px;
                padding: 10px 12px;
                margin-bottom: 16px;
            ">
                <div style="font-size: 0.80rem; color: #B18445; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background-color: #B18445;"></span>
                    Unavailable
                </div>
                <div style="font-size: 0.76rem; color: {COLOR_SECONDARY_TEXT}; margin-top: 3px;">
                    Model artifact not initialized.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="font-size: 0.72rem; color: {COLOR_SECONDARY_TEXT}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
            ABOUT
        </div>
        <div style="font-size: 0.78rem; color: {COLOR_SECONDARY_TEXT}; line-height: 1.45;">
            Predict churn risk, understand the main risk factors, and prioritize high-value customers for retention review.
        </div>
        """,
        unsafe_allow_html=True,
    )
