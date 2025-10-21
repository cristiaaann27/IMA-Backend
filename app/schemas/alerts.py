from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.prediction import AlertLevel, TimeSeriesPoint


class AlertRequest(BaseModel):
    current_conditions: TimeSeriesPoint = Field(description="Condiciones actuales")
    previous_conditions: Optional[List[TimeSeriesPoint]] = Field(
        default=None,
        description="Condiciones previas (para calcular deltas)"
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
                        "timestamp": "2023-01-01T11:00:00",
                        "rh_2m_pct": 85.0,
                        "temp_2m_c": 20.0,
                        "wind_speed_2m_ms": 1.5,
                        "wind_dir_2m_deg": 175.0
                    }
                ]
            }
        }
    }


class WeatherAlertInfo(BaseModel):
    level: str = Field(description="Nivel de alerta (baja, media, alta, critica)")
    variable: str = Field(description="Variable que generó la alerta")
    value: float = Field(description="Valor actual de la variable")
    threshold: float = Field(description="Umbral que se superó")
    message: str = Field(description="Mensaje descriptivo de la alerta")
    timestamp: str = Field(description="Timestamp de la alerta")


class AlertResponse(BaseModel):
    alerts: List[WeatherAlertInfo] = Field(description="Lista de alertas generadas")
    total_alerts: int = Field(description="Total de alertas")
    has_media_or_higher: bool = Field(description="Indica si hay alertas de nivel MEDIA o superior")
    has_alta_or_higher: bool = Field(description="Indica si hay alertas de nivel ALTA o superior")
    timestamp: datetime = Field(description="Timestamp de la evaluación")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "alerts": [
                    {
                        "level": "alta",
                        "variable": "Humedad Relativa",
                        "value": 92.0,
                        "threshold": 90.0,
                        "message": "⚠️ ALERTA ALTA: Humedad muy elevada (92.0%). Alto riesgo de precipitación.",
                        "timestamp": "2025-01-15T12:00:00Z"
                    },
                    {
                        "level": "media",
                        "variable": "Velocidad del Viento",
                        "value": 12.5,
                        "threshold": 10.0,
                        "message": "🌬️ ALERTA MEDIA: Vientos moderados a fuertes (12.5 m/s).",
                        "timestamp": "2025-01-15T12:00:00Z"
                    }
                ],
                "total_alerts": 2,
                "has_media_or_higher": True,
                "has_alta_or_higher": True,
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }
    }
