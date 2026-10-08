"""Model Insights, Governance, and Explainability page."""

import pandas as pd
import streamlit as st

from dashboard.components.risk_card import render_metric_card
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.models.registry import load_production_artifact


def render_model_insights_page() -> None:
    config = load_config()
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">
                Model Intelligence & Governance
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem;">
                Rigorous benchmarking, holdout test metrics, SHAP global feature attributions, and cost-sensitive threshold diagnostics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        pipeline, metadata = load_production_artifact()
        metrics = metadata.metrics
    except Exception as e:
        st.error(f"Could not load model metadata: {e}")
        return

    # Algorithm & Version Badge
    st.info(
        f"**Active Model:** `{metadata.algorithm}` | "
        f"**Version:** `{metadata.model_version}` | "
        f"**Optimal Threshold ($\\tau^*$):** `{metadata.optimal_threshold:.2f}` | "
        f"**Trained On:** `{metadata.dataset_info.get('train_rows', 0):,}` rows"
    )

    # Core Metrics Row (Measured on Unseen Holdout Test Set)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card(
            "ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}", "Holdout Discrimination", "indigo"
        )
    with m2:
        render_metric_card(
            "PR-AUC", f"{metrics.get('pr_auc', 0.0):.4f}", "Precision-Recall AUC", "emerald"
        )
    with m3:
        render_metric_card(
            "Recall (Churn)", f"{metrics.get('recall', 0.0):.1%}", "Identified Churners", "rose"
        )
    with m4:
        render_metric_card(
            "Precision", f"{metrics.get('precision', 0.0):.1%}", "Positive Accuracy", "amber"
        )
    with m5:
        render_metric_card("F1 Score", f"{metrics.get('f1', 0.0):.4f}", "Harmonic Balance", "cyan")

    st.markdown(
        "<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;' />",
        unsafe_allow_html=True,
    )

    # Diagnostic Plots Section
    st.subheader("📊 Diagnostic Curves & Confusion Matrix")
    p_col1, p_col2 = st.columns(2)

    cm_path = (
        PROJECT_ROOT / config.artifacts.figures_dir / "production_xgboost_confusion_matrix.png"
    )
    curves_path = (
        PROJECT_ROOT / config.artifacts.figures_dir / "production_xgboost_roc_pr_curves.png"
    )

    with p_col1:
        if cm_path.exists():
            st.image(str(cm_path), caption="Holdout Test Confusion Matrix", use_column_width=True)
        else:
            st.info("Confusion matrix image generating...")

    with p_col2:
        if curves_path.exists():
            st.image(
                str(curves_path), caption="ROC & Precision-Recall Curves", use_column_width=True
            )
        else:
            st.info("ROC & PR curve image generating...")

    st.markdown(
        "<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;' />",
        unsafe_allow_html=True,
    )

    # SHAP Global Explainability Section
    st.subheader("🔍 SHAP Global Feature Impact Analysis")
    st.caption(
        "Game-theoretic Shapley attributions revealing which features push predictions towards churn vs. retention:"
    )

    shap_col1, shap_col2 = st.columns(2)
    shap_summary_path = PROJECT_ROOT / config.artifacts.figures_dir / "shap_summary_plot.png"
    shap_bar_path = PROJECT_ROOT / config.artifacts.figures_dir / "shap_feature_importance.png"

    with shap_col1:
        if shap_summary_path.exists():
            st.image(
                str(shap_summary_path), caption="SHAP Beeswarm Summary Plot", use_column_width=True
            )
        else:
            st.info("SHAP summary plot generating...")

    with shap_col2:
        if shap_bar_path.exists():
            st.image(
                str(shap_bar_path), caption="Top 12 Features Mean |SHAP|", use_column_width=True
            )
        else:
            st.info("SHAP feature importance bar plot generating...")

    st.markdown(
        "<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;' />",
        unsafe_allow_html=True,
    )

    # Multi-Model Benchmark Comparison Table
    st.subheader("🏆 Model Comparison & Selection Leaderboard")
    benchmark_csv = PROJECT_ROOT / config.artifacts.tables_dir / "model_benchmark_comparison.csv"
    if benchmark_csv.exists():
        bench_df = pd.read_csv(benchmark_csv)
        display_cols = [
            c
            for c in [
                "model_type",
                "cv_roc_auc_mean",
                "cv_roc_auc_std",
                "cv_f1_mean",
                "val_roc_auc",
                "val_pr_auc",
                "val_recall",
                "val_precision",
                "val_f1",
            ]
            if c in bench_df.columns
        ]

        st.dataframe(
            bench_df[display_cols]
            .style.highlight_max(
                subset=["val_roc_auc", "val_pr_auc", "val_recall", "val_f1"],
                color="#064e3b",
            )
            .format(
                {
                    "cv_roc_auc_mean": "{:.4f}",
                    "cv_roc_auc_std": "{:.4f}",
                    "cv_f1_mean": "{:.4f}",
                    "val_roc_auc": "{:.4f}",
                    "val_pr_auc": "{:.4f}",
                    "val_recall": "{:.4f}",
                    "val_precision": "{:.4f}",
                    "val_f1": "{:.4f}",
                }
            ),
            width="stretch",
        )
    else:
        st.info("Benchmark comparison table not found.")
