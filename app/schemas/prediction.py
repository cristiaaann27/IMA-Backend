from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class AlertLevel(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"
    CRITICA = "critica"
    
    @property
    def severity(self) -> int:
        return {"baja": 0, "media": 1, "alta": 2, "critica": 3}[self.value]


class TimeSeriesPoint(BaseModel):
    timestamp: datetime = Field(description="Timestamp del punto")
    rh_2m_pct: float = Field(ge=0, le=100, description="Humedad relativa a 2m (%)")
    temp_2m_c: float = Field(ge=-50, le=60, description="Temperatura a 2m (°C)")
    wind_speed_2m_ms: float = Field(ge=0, le=100, description="Velocidad viento a 2m (m/s)")
    wind_dir_2m_deg: float = Field(ge=0, le=360, description="Dirección viento a 2m (°)")


class PredictionRequest(BaseModel):
    lookback_data: List[TimeSeriesPoint] = Field(
        description="Ventana de observación (lookback)",
        min_length=1
    )
    
    @field_validator("lookback_data")
    @classmethod
    def validate_lookback_length(cls, v):
        """Valida que lookback tenga al menos 1 punto."""
        if len(v) < 1:
            raise ValueError("lookback_data debe tener al menos 1 punto")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "lookback_data": [
                    {
                        "timestamp": "2023-01-01T00:00:00",
                        "rh_2m_pct": 75.5,
                        "temp_2m_c": 22.3,
                        "wind_speed_2m_ms": 3.2,
                        "wind_dir_2m_deg": 180.0
                    },
                    {
                        "timestamp": "2023-01-01T01:00:00",
                        "rh_2m_pct": 78.2,
                        "temp_2m_c": 21.8,
                        "wind_speed_2m_ms": 2.9,
                        "wind_dir_2m_deg": 175.0
                    }
                ]
            }
        }
    }


class DiagnosisInfo(BaseModel):
    level: AlertLevel = Field(description="Nivel de alerta")
    triggered_rules: List[str] = Field(description="Reglas activadas")
    recommendation: str = Field(description="Recomendación")


class PredictionResponse(BaseModel):
    prediction_mm_hr: float = Field(description="Predicción de precipitación (mm/hr)")
    rain_event_prob: float = Field(
        ge=0,
        le=1,
        description="Probabilidad de evento de lluvia (0-1)"
    )
    diagnosis: DiagnosisInfo = Field(description="Diagnóstico")
    latency_ms: float = Field(description="Latencia de la predicción (ms)")
    timestamp: datetime = Field(description="Timestamp de la predicción")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction_mm_hr": 2.45,
                "rain_event_prob": 0.87,
                "diagnosis": {
                    "level": "alta",
                    "triggered_rules": [
                        "RH_HIGH_TEMP_DROP",
                        "HIGH_PRECIP_PRED"
                    ],
                    "recommendation": "Lluvia probable inminente. Asegurar drenajes..."
                },
                "latency_ms": 45.3,
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }
    }


class ForecastRequest(BaseModel):
    lookback_data: List[TimeSeriesPoint] = Field(
        description="Ventana de observación inicial"
    )
    horizon: int = Field(
        ge=1,
        le=24,
        description="Horizonte de pronóstico (horas)",
        default=6
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "lookback_data": [
                    {
                        "timestamp": "2023-01-01T00:00:00",
                        "rh_2m_pct": 75.5,
                        "temp_2m_c": 22.3,
                        "wind_speed_2m_ms": 3.2,
                        "wind_dir_2m_deg": 180.0
                    }
                ],
                "horizon": 6
            }
        }
    }


class ForecastStep(BaseModel):
    timestamp: datetime = Field(description="Timestamp del paso")
    prediction_mm_hr: float = Field(description="Predicción (mm/hr)")
    rain_event_prob: float = Field(description="Probabilidad de lluvia")
    diagnosis_level: AlertLevel = Field(description="Nivel de alerta")


class ForecastResponse(BaseModel):
    forecast: List[ForecastStep] = Field(description="Pasos del pronóstico")
    latency_ms: float = Field(description="Latencia total (ms)")
    timestamp: datetime = Field(description="Timestamp de generación")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "forecast": [
                    {
                        "timestamp": "2023-01-01T01:00:00",
                        "prediction_mm_hr": 1.2,
                        "rain_event_prob": 0.65,
                        "diagnosis_level": "media"
                    },
                    {
                        "timestamp": "2023-01-01T02:00:00",
                        "prediction_mm_hr": 2.8,
                        "rain_event_prob": 0.82,
                        "diagnosis_level": "alta"
                    }
                ],
                "latency_ms": 125.7,
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }
    }


class DiagnosisRequest(BaseModel):
    current_conditions: TimeSeriesPoint = Field(description="Condiciones actuales")
    previous_conditions: Optional[List[TimeSeriesPoint]] = Field(
        default=None,
        description="Condiciones previas (para calcular deltas)"
    )
    predicted_precip_mm_hr: Optional[float] = Field(
        default=None,
        description="Predicción de precipitación (opcional)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "current_conditions": {
                    "timestamp": "2023-01-01T12:00:00",
                    "rh_2m_pct": 92.0,
                    "temp_2m_c": 18.5,
                    "wind_speed_2m_ms": 0.8,
                    "wind_dir_2m_deg": 180.0
                },
                "previous_conditions": [
                    {
                        "timestamp": "2023-01-01T10:00:00",
                        "rh_2m_pct": 85.0,
                        "temp_2m_c": 20.0,
                        "wind_speed_2m_ms": 1.5,
                        "wind_dir_2m_deg": 175.0
                    }
                ],
                "predicted_precip_mm_hr": 3.2
            }
        }
    }


class DiagnosisResponse(BaseModel):
    level: AlertLevel = Field(description="Nivel de alerta")
    triggered_rules: List[str] = Field(description="Reglas activadas")
    recommendation: str = Field(description="Recomendación operacional")
    metrics: dict = Field(description="Métricas utilizadas")
    timestamp: datetime = Field(description="Timestamp del diagnóstico")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "level": "alta",
                "triggered_rules": [
                    "RH_HIGH_TEMP_DROP: RH=92.0% >= 90.0%, ΔTemp_2h=-1.5°C <= -0.5°C",
                    "WIND_CALM_RH_HIGH: Viento=0.8 m/s <= 1.0, RH=92.0% >= 85.0%"
                ],
                "recommendation": "⚠️ ACCIÓN INMEDIATA: Lluvia muy probable...",
                "metrics": {
                    "rh_2m_pct": 92.0,
                    "temp_2m_c": 18.5,
                    "temp_delta_2h": -1.5,
                    "wind_speed_2m_ms": 0.8
                },
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }
    }

