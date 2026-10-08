"""Model registry and production artifact serialization."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from src.telco_churn.config import PROJECT_ROOT, load_config
from src.telco_churn.logging_config import configure_logger

logger = configure_logger("model_registry")


class ProductionModelMetadata(BaseModel):
    model_name: str
    model_version: str
    algorithm: str
    trained_at: str
    optimal_threshold: float
    feature_names: List[str]
    metrics: Dict[str, Any]
    business_assumptions: Dict[str, Any]
    dataset_info: Dict[str, Any]


class ModelRegistry:
    """Manages local model artifact versioning and metadata."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            config = load_config()
            self.base_dir = PROJECT_ROOT / config.artifacts.model_dir
        else:
            self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_production_model(
        self,
        pipeline: Pipeline,
        algorithm: str,
        threshold: float,
        feature_names: List[str],
        metrics: Dict[str, Any],
        business_assumptions: Dict[str, Any],
        dataset_info: Dict[str, Any],
        version: str = "v1.0.0",
    ) -> Path:
        """Serialize pipeline and metadata to registry storage."""
        model_path = self.base_dir / "production_pipeline.joblib"
        metadata_path = self.base_dir / "metadata.json"

        # Save pipeline
        joblib.dump(pipeline, model_path)
        logger.info(f"Production pipeline artifact saved to {model_path}")

        # Save versioned backup
        versioned_path = self.base_dir / f"pipeline_{version}.joblib"
        joblib.dump(pipeline, versioned_path)

        metadata = ProductionModelMetadata(
            model_name="telco-churn-retention-model",
            model_version=version,
            algorithm=algorithm,
            trained_at=datetime.now(timezone.utc).isoformat(),
            optimal_threshold=round(float(threshold), 4),
            feature_names=feature_names,
            metrics=metrics,
            business_assumptions=business_assumptions,
            dataset_info=dataset_info,
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))
        logger.info(f"Model metadata successfully written to {metadata_path}")

        return model_path

    def load_production_model(self) -> Tuple[Pipeline, ProductionModelMetadata]:
        """Load production pipeline and metadata."""
        model_path = self.base_dir / "production_pipeline.joblib"
        metadata_path = self.base_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Production model artifact not found at {model_path}. "
                f"Please run 'python scripts/train_model.py' to generate it."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}.")

        pipeline = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        metadata = ProductionModelMetadata.model_validate(meta_dict)

        logger.info(
            f"Loaded production model '{metadata.model_name}' ({metadata.algorithm}) "
            f"version {metadata.model_version} with threshold {metadata.optimal_threshold}"
        )
        return pipeline, metadata


def save_production_artifact(
    pipeline: Pipeline,
    algorithm: str,
    threshold: float,
    feature_names: List[str],
    metrics: Dict[str, Any],
    business_assumptions: Dict[str, Any],
    dataset_info: Dict[str, Any],
    version: str = "v1.0.0",
) -> Path:
    registry = ModelRegistry()
    return registry.save_production_model(
        pipeline=pipeline,
        algorithm=algorithm,
        threshold=threshold,
        feature_names=feature_names,
        metrics=metrics,
        business_assumptions=business_assumptions,
        dataset_info=dataset_info,
        version=version,
    )


def load_production_artifact() -> Tuple[Pipeline, ProductionModelMetadata]:
    registry = ModelRegistry()
    return registry.load_production_model()
