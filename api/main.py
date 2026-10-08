"""FastAPI production service for Telco Customer Churn & Retention Optimization."""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_app_config, get_model_predictor
from api.logging_config import api_logger
from api.schemas import (
    BatchCustomerRequest,
    BatchPredictionResponse,
    CustomerPredictionRequest,
    CustomerPredictionResponse,
    HealthResponse,
    MetadataResponse,
)
from src.telco_churn.inference.predictor import TelcoChurnPredictor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager."""
    api_logger.info("Starting up Telco Churn Prediction API service...")
    try:
        # Pre-load production model artifact once
        predictor = get_model_predictor()
        api_logger.info(
            f"Pre-warmed predictor: Model={predictor.metadata.algorithm}, "
            f"Threshold={predictor.metadata.optimal_threshold}, Version={predictor.metadata.model_version}"
        )
    except Exception as e:
        api_logger.error(f"Failed to load production model at startup: {e}", exc_info=True)
        raise RuntimeError("Service startup aborted due to model load failure.") from e

    yield
    api_logger.info("Shutting down Telco Churn Prediction API service...")


app = FastAPI(
    title="Telco Churn Prediction & Retention Optimization API",
    description=(
        "Production-grade REST API delivering customer churn probability, "
        "cost-sensitive risk categorization, customer lifetime value (CLV), "
        "retention priority ranking, and exact SHAP feature attribution drivers."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Structured request latency logging."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    api_logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Latency: {duration_ms:.2f}ms"
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Clean validation error formatting without leaking internals."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(part) for part in err.get("loc", []))
        errors.append({"field": loc, "message": err.get("msg")})
    api_logger.warning(f"Validation error on {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Request validation failed", "details": errors},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Shield internal stack traces while logging server errors."""
    api_logger.error(f"Unhandled server error processing {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error occurred. Please contact technical operations."},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["System"],
)
def health_check(
    predictor: TelcoChurnPredictor = Depends(get_model_predictor),
) -> HealthResponse:
    """Return health status, active model algorithm, threshold, and version."""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=predictor.metadata.model_name,
        model_version=predictor.metadata.model_version,
        optimal_threshold=predictor.metadata.optimal_threshold,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Model & System Metadata",
    tags=["System"],
)
def get_metadata(
    predictor: TelcoChurnPredictor = Depends(get_model_predictor),
) -> MetadataResponse:
    """Return comprehensive metadata regarding the production model, evaluation metrics, and features."""
    meta = predictor.metadata
    return MetadataResponse(
        model_name=meta.model_name,
        model_version=meta.model_version,
        algorithm=meta.algorithm,
        trained_at=meta.trained_at,
        optimal_threshold=meta.optimal_threshold,
        feature_names=meta.feature_names,
        metrics=meta.metrics,
        business_assumptions=meta.business_assumptions,
        dataset_info=meta.dataset_info,
    )


@app.post(
    "/predict",
    response_model=CustomerPredictionResponse,
    summary="Predict Churn for Single Customer",
    tags=["Inference"],
)
def predict_single_customer(
    request: CustomerPredictionRequest,
    predictor: TelcoChurnPredictor = Depends(get_model_predictor),
) -> CustomerPredictionResponse:
    """Evaluate a single customer profile, returning churn probability, risk level, CLV, priority, and top 3 SHAP drivers."""
    try:
        return predictor.predict_single(request)
    except Exception as e:
        api_logger.error(
            f"Prediction failed for customer '{request.customer_id}': {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(e)}",
        ) from e


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predict Customer Churn",
    tags=["Inference"],
)
def predict_batch_customers(
    request: BatchCustomerRequest,
    predictor: TelcoChurnPredictor = Depends(get_model_predictor),
) -> BatchPredictionResponse:
    """Process a batch of customer profiles, returning ranked predictions, at-risk CLV, and drivers."""
    if not request.customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list must contain at least 1 customer profile.",
        )

    try:
        rows = []
        for c in request.customers:
            d = c.model_dump()
            d["customerID"] = d.pop("customer_id", "CUST-UNKNOWN")
            rows.append(d)

        df = pd.DataFrame(rows)
        return predictor.predict_dataframe(df, batch_explain=True)
    except Exception as e:
        api_logger.error(
            f"Batch prediction failure for {len(request.customers)} records: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failure: {str(e)}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    cfg = get_app_config()
    uvicorn.run("api.main:app", host=cfg.api.host, port=cfg.api.port, reload=False)
