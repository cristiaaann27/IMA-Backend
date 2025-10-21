from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    mae: Optional[float] = Field(default=None, description="MAE")
    rmse: Optional[float] = Field(default=None, description="RMSE")
    r2: Optional[float] = Field(default=None, description="R²")
    precision: Optional[float] = Field(default=None, description="Precision")
    recall: Optional[float] = Field(default=None, description="Recall")
    f1_score: Optional[float] = Field(default=None, description="F1-Score")
    mape: Optional[float] = Field(default=None, description="MAPE")


class AllModelMetrics(BaseModel):
    lstm: ModelMetrics = Field(description="Métricas del modelo LSTM")
    xgboost: ModelMetrics = Field(description="Métricas del modelo XGBoost")


class OnlineMetrics(BaseModel):
    total_predictions: int = Field(description="Total de predicciones")
    total_forecasts: int = Field(description="Total de pronósticos")
    total_diagnoses: int = Field(description="Total de diagnósticos")
    avg_prediction_latency_ms: float = Field(description="Latencia promedio predicción")
    avg_forecast_latency_ms: float = Field(description="Latencia promedio pronóstico")
    p95_prediction_latency_ms: Optional[float] = Field(
        default=None,
        description="Latencia P95 predicción"
    )
    p99_prediction_latency_ms: Optional[float] = Field(
        default=None,
        description="Latencia P99 predicción"
    )


class ErrorMetrics(BaseModel):
    total_errors: int = Field(description="Total de errores")
    error_rate: float = Field(description="Tasa de error (%)")
    errors_by_type: Dict[str, int] = Field(description="Errores por tipo")


class MetricsResponse(BaseModel):
    timestamp: datetime = Field(description="Timestamp de las métricas")
    model_metrics: AllModelMetrics = Field(description="Métricas offline de todos los modelos")
    online_metrics: OnlineMetrics = Field(description="Métricas online")
    error_metrics: ErrorMetrics = Field(description="Métricas de errores")
    uptime_seconds: float = Field(description="Tiempo de uptime")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2025-01-15T12:00:00Z",
                "model_metrics": {
                    "lstm": {
                        "mae": 0.9292,
                        "rmse": 3.4836,
                        "r2": 0.8612,
                        "precision": 0.9583,
                        "recall": 0.9658,
                        "f1_score": 0.9620,
                        "mape": 74.82
                    },
                    "xgboost": {
                        "mae": 0.2493,
                        "rmse": 2.8976,
                        "r2": 0.9039,
                        "precision": 0.9922,
                        "recall": 0.9939,
                        "f1_score": 0.9930,
                        "mape": 7354587.54
                    }
                },
                "online_metrics": {
                    "total_predictions": 1523,
                    "total_forecasts": 342,
                    "total_diagnoses": 87,
                    "avg_prediction_latency_ms": 42.5,
                    "avg_forecast_latency_ms": 125.3,
                    "p95_prediction_latency_ms": 78.2,
                    "p99_prediction_latency_ms": 95.6
                },
                "error_metrics": {
                    "total_errors": 12,
                    "error_rate": 0.62,
                    "errors_by_type": {
                        "validation_error": 8,
                        "prediction_error": 3,
                        "timeout": 1
                    }
                },
                "uptime_seconds": 86400.0
            }
        }
    }

