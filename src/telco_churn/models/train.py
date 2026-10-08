"""Model training, cross-validation, and multi-model benchmarking pipeline."""

from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.telco_churn.config import load_config
from src.telco_churn.features.build import TelcoFeatureEngineer, create_preprocessor
from src.telco_churn.logging_config import configure_logger
from src.telco_churn.models.evaluate import compute_metrics

logger = configure_logger("model_train")


def build_pipeline(
    model_type: str = "logistic_regression",
    model_params: Optional[Dict[str, Any]] = None,
    use_class_weight: bool = True,
    scale_pos_weight: float = 2.77,  # ~ (1 - 0.265) / 0.265
) -> Pipeline:
    """Build a unified scikit-learn Pipeline with feature engineering, preprocessing, and classifier."""
    config = load_config()
    feature_engineer = TelcoFeatureEngineer(
        tenure_bins=config.features.tenure_bucket_bins,
        tenure_labels=config.features.tenure_bucket_labels,
    )
    preprocessor = create_preprocessor()

    params = model_params.copy() if model_params else {}

    if model_type == "logistic_regression":
        base_params = {
            "max_iter": 1000,
            "random_state": config.random_seed,
            "solver": "lbfgs",
            "class_weight": "balanced" if use_class_weight else None,
        }
        base_params.update(params)
        classifier = LogisticRegression(**base_params)

    elif model_type == "random_forest":
        base_params = {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "random_state": config.random_seed,
            "n_jobs": -1,
            "class_weight": "balanced" if use_class_weight else None,
        }
        base_params.update(params)
        classifier = RandomForestClassifier(**base_params)

    elif model_type == "xgboost":
        base_params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight if use_class_weight else 1.0,
            "random_state": config.random_seed,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        base_params.update(params)
        classifier = xgb.XGBClassifier(**base_params)

    elif model_type == "lightgbm":
        base_params = {
            "n_estimators": 150,
            "max_depth": 5,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "scale_pos_weight": scale_pos_weight if use_class_weight else 1.0,
            "random_state": config.random_seed,
            "verbose": -1,
            "n_jobs": -1,
        }
        base_params.update(params)
        classifier = lgb.LGBMClassifier(**base_params)  # type: ignore[arg-type]

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Choose from lr, rf, xgboost, lightgbm."
        )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", feature_engineer),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def train_model(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    model_params: Optional[Dict[str, Any]] = None,
    use_class_weight: bool = True,
    run_cv: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train model pipeline with cross-validation and validation set evaluation."""
    config = load_config()
    target_col = config.data.target_column

    X_train = train_df.drop(columns=[target_col, "customerID"], errors="ignore")
    y_train = train_df[target_col].astype(int)

    pipeline = build_pipeline(
        model_type=model_type,
        model_params=model_params,
        use_class_weight=use_class_weight,
    )

    cv_results = {}
    if run_cv:
        logger.info(f"Running 5-fold Stratified CV for {model_type}...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_seed)
        roc_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_results = {
            "cv_roc_auc_mean": round(float(np.mean(roc_scores)), 4),
            "cv_roc_auc_std": round(float(np.std(roc_scores)), 4),
            "cv_f1_mean": round(float(np.mean(f1_scores)), 4),
            "cv_f1_std": round(float(np.std(f1_scores)), 4),
        }
        logger.info(
            f"{model_type} CV ROC-AUC: {cv_results['cv_roc_auc_mean']:.4f} (+/- {cv_results['cv_roc_auc_std']:.4f})"
        )

    # Fit final model on full training set
    logger.info(f"Fitting {model_type} on full training split ({len(train_df)} rows)...")
    pipeline.fit(X_train, y_train)

    val_metrics = {}
    if val_df is not None:
        X_val = val_df.drop(columns=[target_col, "customerID"], errors="ignore")
        y_val = val_df[target_col].astype(int)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_prob)
        logger.info(
            f"{model_type} Val Metrics: ROC-AUC={val_metrics['roc_auc']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, Recall={val_metrics['recall']:.4f}"
        )

    result_summary = {
        "model_type": model_type,
        "class_weighting": use_class_weight,
        **cv_results,
        **{f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, list)},
    }

    return pipeline, result_summary


def run_model_comparison(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    models: Optional[List[str]] = None,
    track_mlflow: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Run systematic benchmarking across all candidate models and track in MLflow."""
    config = load_config()
    if models is None:
        models = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

    if track_mlflow:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

    records = []
    trained_pipelines: Dict[str, Pipeline] = {}

    for model_name in models:
        logger.info(f"--- Benchmarking Candidate: {model_name.upper()} ---")
        if track_mlflow:
            with mlflow.start_run(run_name=f"benchmark_{model_name}"):
                pipeline, summary = train_model(model_name, train_df, val_df, run_cv=True)
                mlflow.log_params({"model_type": model_name, "class_weighted": True})
                # Log metrics
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
        else:
            pipeline, summary = train_model(model_name, train_df, val_df, run_cv=True)

        records.append(summary)
        trained_pipelines[model_name] = pipeline

    comparison_df = pd.DataFrame(records)
    logger.info("Model Benchmarking Completed:\n" + comparison_df.to_string())
    return comparison_df, trained_pipelines
