"""Sidebar status and model health indicator component."""

import streamlit as st

from src.telco_churn.models.registry import load_production_artifact


def render_sidebar_status() -> None:
    """Render verified model status in sidebar."""
    st.markdown(
        "<div style='font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>MODEL STATUS</div>",
        unsafe_allow_html=True,
    )

    try:
        _, meta = load_production_artifact()
        st.markdown(
            f"""
            <div style="
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.35);
                border-radius: 8px;
                padding: 12px 14px;
                margin-bottom: 16px;
            ">
                <div style="font-size: 0.78rem; color: #10b981; font-weight: 700;">● Model Ready</div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc; margin-top: 4px;">{meta.model_name}</div>
                <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 2px;">
                    Version: <b>{meta.model_version}</b><br/>
                    Threshold: <b>{meta.optimal_threshold:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            """
            <div style="
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.35);
                border-radius: 8px;
                padding: 12px 14px;
                margin-bottom: 16px;
            ">
                <div style="font-size: 0.78rem; color: #f59e0b; font-weight: 700;">⚠️ Model Unavailable</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Run `python scripts/train_model.py`</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">ABOUT</div>
        <p style="font-size: 0.8rem; color: #64748b; line-height: 1.45; margin-bottom: 0;">
            Predict customer churn, understand the main risk drivers, and prioritize high-value customers for retention.
        </p>
        """,
        unsafe_allow_html=True,
    )
