from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class AlertHistoryEntry(BaseModel):
    entry_id: str
    level: str
    variable: str
    value: float
    threshold: float
    message: str
    timestamp: str
    notified_users: List[str]
    resolved_at: Optional[str] = None


class AlertHistoryRequest(BaseModel):
    from_date: Optional[str] = Field(
        default=None,
        description="Fecha inicial (ISO format)"
    )
    to_date: Optional[str] = Field(
        default=None,
        description="Fecha final (ISO format)"
    )
    level: Optional[str] = Field(
        default=None,
        description="Filtrar por nivel"
    )
    variable: Optional[str] = Field(
        default=None,
        description="Filtrar por variable"
    )
    limit: Optional[int] = Field(
        default=100,
        description="Número máximo de resultados"
    )


class AlertHistoryResponse(BaseModel):
    alerts: List[AlertHistoryEntry]
    total: int
    filters_applied: Dict[str, Optional[str]]


class AlertStatisticsResponse(BaseModel):
    total_alerts: int
    by_level: Dict[str, int]
    by_variable: Dict[str, int]
    most_common_variable: Optional[str]
    highest_level: Optional[str]
    date_range: Dict[str, Optional[str]]
