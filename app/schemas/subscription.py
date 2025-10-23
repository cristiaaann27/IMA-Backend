from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class SubscriptionCreate(BaseModel):
    user_id: str = Field(..., description="ID del usuario")
    min_level: str = Field(
        default="medio",
        description="Nivel mínimo de alertas (bajo, medio, alto, critico)"
    )
    variables: Optional[List[str]] = Field(
        default=None,
        description="Variables a monitorear (None o [] = todas)"
    )
    notification_frequency: Optional[Dict[str, int]] = Field(
        default=None,
        description="Frecuencia de notificaciones en minutos por nivel (bajo: 1440, medio: 720, alto: 180, critico: 30)"
    )


class SubscriptionUpdate(BaseModel):
    min_level: Optional[str] = Field(
        default=None,
        description="Nuevo nivel mínimo (bajo, medio, alto, critico)"
    )
    variables: Optional[List[str]] = Field(
        default=None,
        description="Nuevas variables a monitorear"
    )
    enabled: Optional[bool] = Field(
        default=None,
        description="Estado de la suscripción"
    )
    notification_frequency: Optional[Dict[str, int]] = Field(
        default=None,
        description="Frecuencia de notificaciones en minutos por nivel"
    )


class SubscriptionResponse(BaseModel):
    subscription_id: str
    user_id: str
    min_level: str
    variables: List[str]
    enabled: bool
    created_at: str
    notification_frequency: Dict[str, int]


class SubscriptionListResponse(BaseModel):
    subscriptions: List[SubscriptionResponse]
    total: int


class SubscriptionDeleteResponse(BaseModel):
    success: bool
    message: str
    subscription_id: str
