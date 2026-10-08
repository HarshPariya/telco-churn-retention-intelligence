"""Pydantic schemas for the FastAPI service."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.telco_churn.inference.predictor import (
    BatchPredictionResponse,
    CustomerDriver,
    CustomerPredictionRequest,
    CustomerPredictionResponse,
)


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    model_name: str
    model_version: str
    optimal_threshold: float
    timestamp: str


class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    algorithm: str
    trained_at: str
    optimal_threshold: float
    feature_names: List[str]
    metrics: Dict[str, Any]
    business_assumptions: Dict[str, Any]
    dataset_info: Dict[str, Any]


class BatchCustomerRequest(BaseModel):
    customers: List[CustomerPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of customer profiles for batch prediction",
    )


class ErrorDetail(BaseModel):
    loc: Optional[List[str]] = None
    msg: str
    type: str


class ErrorResponse(BaseModel):
    error: str
    details: Optional[Any] = None
