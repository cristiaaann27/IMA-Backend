from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    LOADED = "loaded"
    NOT_LOADED = "not_loaded"
    ERROR = "error"


class HealthResponse(BaseModel):
    status: str = Field(description="Estado general del servicio")
    timestamp: datetime = Field(description="Timestamp del health check")
    version: str = Field(description="Versión de la API")
    model_status: ModelStatus = Field(description="Estado del modelo")
    model_loaded_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp de carga del modelo"
    )
    uptime_seconds: float = Field(description="Tiempo de uptime en segundos")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-15T10:30:00Z",
                "version": "1.0.0",
                "model_status": "loaded",
                "model_loaded_at": "2025-01-15T10:00:00Z",
                "uptime_seconds": 1800.5
            }
        }
    }

