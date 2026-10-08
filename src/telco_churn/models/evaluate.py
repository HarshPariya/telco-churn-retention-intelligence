"""Model evaluation metrics and diagnostic plotting."""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.telco_churn.config import PROJECT_ROOT
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("model_evaluate")


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)
    y_pred = (y_p >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_t, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Core metrics
    acc = float(accuracy_score(y_t, y_pred))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_t, y_p))
    pr_auc = float(average_precision_score(y_t, y_p))
    brier = float(brier_score_loss(y_t, y_p))

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
    threshold: float = 0.50,
    save_plots: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute metrics and optionally generate diagnostic plots."""
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)

    if save_plots:
        if output_dir is None:
            output_dir = PROJECT_ROOT / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        y_t = np.asarray(y_true).astype(int)
        y_p = np.asarray(y_prob).astype(float)
        y_pred = (y_p >= threshold).astype(int)

        # 1. Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_t, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Retained (0)", "Churn (1)"],
            yticklabels=["Retained (0)", "Churn (1)"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {model_name} (Threshold={threshold:.2f})")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        cm_path = output_dir / f"{model_name}_confusion_matrix.png"
        fig.savefig(cm_path, dpi=200)
        plt.close(fig)

        # 2. ROC & PR Curves Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fpr, tpr, _ = roc_curve(y_t, y_p)
        ax1.plot(
            fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})", color="#1f77b4", lw=2
        )
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax1.set_title("Receiver Operating Characteristic (ROC)")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate (Recall)")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        prec_arr, rec_arr, _ = precision_recall_curve(y_t, y_p)
        ax2.plot(
            rec_arr,
            prec_arr,
            label=f"PR Curve (AUC = {metrics['pr_auc']:.3f})",
            color="#2ca02c",
            lw=2,
        )
        ax2.set_title("Precision-Recall Curve")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = output_dir / f"{model_name}_roc_pr_curves.png"
        fig.savefig(curves_path, dpi=200)
        plt.close(fig)

        metrics["confusion_matrix_plot"] = str(cm_path)
        metrics["roc_pr_plot"] = str(curves_path)

    return metrics
