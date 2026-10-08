"""Evaluate registered production model on test split and generate summary metrics."""

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.logging_config import configure_logger
from src.telco_churn.models.evaluate import evaluate_predictions
from src.telco_churn.models.registry import load_production_artifact

logger = configure_logger("evaluate_model")


def main() -> None:
    config = load_config()
    logger.info("Loading production model artifact...")
    pipeline, metadata = load_production_artifact()

    test_path = PROJECT_ROOT / config.data.test_path
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}")

    test_df = pd.read_parquet(test_path)
    X_test = test_df.drop(columns=[config.data.target_column, "customerID"], errors="ignore")
    y_test = test_df[config.data.target_column].astype(int)

    logger.info(f"Evaluating model on test dataset ({len(test_df)} records)...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    figures_dir = PROJECT_ROOT / config.artifacts.figures_dir
    metrics = evaluate_predictions(
        y_true=y_test.values,
        y_prob=y_prob,
        model_name="evaluation_holdout",
        threshold=metadata.optimal_threshold,
        save_plots=True,
        output_dir=figures_dir,
    )

    logger.info("--- Holdout Evaluation Metrics ---")
    for k, v in metrics.items():
        if not isinstance(v, (list, dict)):
            logger.info(f"  {k}: {v}")

    out_metrics_file = PROJECT_ROOT / config.artifacts.tables_dir / "latest_evaluation_metrics.json"
    out_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation metrics saved to {out_metrics_file}")


if __name__ == "__main__":
    main()
