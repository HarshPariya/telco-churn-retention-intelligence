"""Hyperparameter tuning with Stratified Cross-Validation."""

from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.telco_churn.config import load_config
from src.telco_churn.logging_config import configure_logger
from src.telco_churn.models.train import build_pipeline

logger = configure_logger("model_tune")

TUNING_PARAM_GRIDS = {
    "random_forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [6, 8, 12],
        "classifier__min_samples_split": [5, 10],
        "classifier__min_samples_leaf": [2, 4],
    },
    "xgboost": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 4, 6],
        "classifier__learning_rate": [0.03, 0.08],
        "classifier__subsample": [0.8, 1.0],
    },
    "lightgbm": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 6],
        "classifier__num_leaves": [15, 31],
        "classifier__learning_rate": [0.03, 0.08],
    },
}


def tune_hyperparameters(
    model_type: str,
    train_df: pd.DataFrame,
    param_grid: Any = None,
    scoring: str = "roc_auc",
    cv_folds: int = 5,
    track_mlflow: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Execute grid search hyperparameter tuning using StratifiedKFold on training data."""
    config = load_config()
    target_col = config.data.target_column

    X_train = train_df.drop(columns=[target_col, "customerID"], errors="ignore")
    y_train = train_df[target_col].astype(int)

    base_pipeline = build_pipeline(model_type=model_type, use_class_weight=True)

    if param_grid is None:
        param_grid = TUNING_PARAM_GRIDS.get(model_type, {})

    if not param_grid:
        logger.warning(f"No parameter grid defined for {model_type}. Returning base pipeline.")
        base_pipeline.fit(X_train, y_train)
        return base_pipeline, {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.random_seed)

    logger.info(f"Starting GridSearchCV for {model_type} with scoring='{scoring}'...")
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    if track_mlflow:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        with mlflow.start_run(run_name=f"tune_{model_type}"):
            grid_search.fit(X_train, y_train)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric(f"best_cv_{scoring}", float(grid_search.best_score_))
    else:
        grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    tuning_summary = {
        "model_type": model_type,
        "best_params": grid_search.best_params_,
        "best_score": round(float(grid_search.best_score_), 4),
        "scoring_metric": scoring,
    }

    logger.info(
        f"Tuning finished for {model_type}: Best {scoring} = {tuning_summary['best_score']:.4f} "
        f"with params: {tuning_summary['best_params']}"
    )

    return best_pipeline, tuning_summary
