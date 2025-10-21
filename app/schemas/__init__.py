"""Schemas Pydantic v2 para request/response."""

from .health import HealthResponse, ModelStatus
from .model import ModelInfo, ModelMetrics
from .prediction import (
    PredictionRequest,
    PredictionResponse,
    ForecastRequest,
    ForecastResponse,
    ForecastStep,
    DiagnosisRequest,
    DiagnosisResponse
)
from .metrics import MetricsResponse, ModelMetrics as ModelMetricsData

__all__ = [
    "HealthResponse",
    "ModelStatus",
    "ModelInfo",
    "ModelMetrics",
    "PredictionRequest",
    "PredictionResponse",
    "ForecastRequest",
    "ForecastResponse",
    "ForecastStep",
    "DiagnosisRequest",
    "DiagnosisResponse",
    "MetricsResponse",
    "ModelMetricsData"
]

