"""Model Performance, Governance, and Explainability view."""

import pandas as pd
import streamlit as st

from dashboard.components.risk_card import render_metric_card
from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.models.registry import load_production_artifact


def render_model_insights() -> None:
    config = load_config()

    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">
                Model Performance & Explainability
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem; margin-bottom: 12px;">
                Comprehensive evaluation metrics, multi-model benchmark comparison, global SHAP feature attribution, and cost-sensitive threshold curves.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        _, metadata = load_production_artifact()
        metrics = metadata.metrics
        ds_info = metadata.dataset_info
    except Exception:
        st.error("Could not load production model metadata. Please verify models/ directory.")
        return

    # Production Metadata Badge
    st.info(
        f"**Production Model:** `{metadata.algorithm}` | "
        f"**Version:** `{metadata.model_version}` | "
        f"**Optimal Threshold ($\\tau^*$):** `{metadata.optimal_threshold:.2f}` | "
        f"**Training Set Size:** `{ds_info.get('train_rows', 4507):,}` accounts | "
        f"**Holdout Set Size:** `{ds_info.get('test_rows', 1409):,}` accounts"
    )

    # Core Metrics Row (Measured on Unseen Holdout Test Set)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        render_metric_card(
            "ROC-AUC", f"{metrics.get('roc_auc', 0.8439):.4f}", "Holdout Discrimination", "indigo"
        )
    with m2:
        render_metric_card(
            "PR-AUC", f"{metrics.get('pr_auc', 0.6582):.4f}", "Precision-Recall AUC", "emerald"
        )
    with m3:
        render_metric_card(
            "Recall (Churn)", f"{metrics.get('recall', 0.9385):.1%}", "Identified Churners", "rose"
        )
    with m4:
        render_metric_card(
            "Precision", f"{metrics.get('precision', 0.4105):.1%}", "Flagged Precision", "amber"
        )
    with m5:
        render_metric_card(
            "F1 Score", f"{metrics.get('f1', 0.5712):.4f}", "Harmonic Mean", "cyan"
        )
    with m6:
        render_metric_card(
            "Brier Score", f"{metrics.get('brier_score', 0.1631):.4f}", "Probability Calibration", "indigo"
        )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;' />", unsafe_allow_html=True)

    # Multi-Model Benchmark Comparison Table
    st.subheader("🏆 Model Benchmarking & Selection Comparison")
    st.caption("5-Fold Stratified Cross-Validation on training cohort and holdout test verification:")

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
            .rename(
                columns={
                    "model_type": "Model Architecture",
                    "cv_roc_auc_mean": "CV ROC-AUC",
                    "cv_roc_auc_std": "CV Std",
                    "cv_f1_mean": "CV F1",
                    "val_roc_auc": "Val ROC-AUC",
                    "val_pr_auc": "Val PR-AUC",
                    "val_recall": "Val Recall",
                    "val_precision": "Val Precision",
                    "val_f1": "Val F1",
                }
            )
            .style.highlight_max(
                subset=["Val ROC-AUC", "Val PR-AUC", "Val Recall", "Val F1"],
                color="#064e3b",
            )
            .format(
                {
                    "CV ROC-AUC": "{:.4f}",
                    "CV Std": "{:.4f}",
                    "CV F1": "{:.4f}",
                    "Val ROC-AUC": "{:.4f}",
                    "Val PR-AUC": "{:.4f}",
                    "Val Recall": "{:.4f}",
                    "Val Precision": "{:.4f}",
                    "Val F1": "{:.4f}",
                }
            ),
            width="stretch",
        )
    else:
        st.info("Benchmark comparison table not found.")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;' />", unsafe_allow_html=True)

    # Diagnostic Curves & Confusion Matrix
    st.subheader("📊 Holdout Confusion Matrix & Discriminative Curves")
    p1, p2 = st.columns(2)

    cm_path = PROJECT_ROOT / config.artifacts.figures_dir / "production_xgboost_confusion_matrix.png"
    curves_path = PROJECT_ROOT / config.artifacts.figures_dir / "production_xgboost_roc_pr_curves.png"

    with p1:
        if cm_path.exists():
            st.image(str(cm_path), caption="Holdout Test Confusion Matrix (Threshold = 0.23)", width="stretch")
        else:
            st.info("Confusion matrix plot generating...")

    with p2:
        if curves_path.exists():
            st.image(str(curves_path), caption="ROC & Precision-Recall Curves", width="stretch")
        else:
            st.info("ROC & PR curves plot generating...")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;' />", unsafe_allow_html=True)

    # SHAP Global Explainability Section
    st.subheader("🔍 SHAP Global Feature Impact Analysis")
    st.caption("Game-theoretic Shapley attributions revealing which features push predictions towards churn vs. retention:")

    s1, s2 = st.columns(2)
    shap_summary_path = PROJECT_ROOT / config.artifacts.figures_dir / "shap_summary_plot.png"
    shap_bar_path = PROJECT_ROOT / config.artifacts.figures_dir / "shap_feature_importance.png"

    with s1:
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), caption="SHAP Beeswarm Summary Plot", width="stretch")
        else:
            st.info("SHAP summary plot generating...")

    with s2:
        if shap_bar_path.exists():
            st.image(str(shap_bar_path), caption="Top 12 Features Mean |SHAP| Importance", width="stretch")
        else:
            st.info("SHAP feature importance bar plot generating...")
