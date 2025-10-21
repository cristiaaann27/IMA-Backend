from typing import List, Optional
from pydantic import BaseModel, Field


class SubscriptionCreate(BaseModel):
    user_id: str = Field(..., description="ID del usuario")
    min_level: str = Field(
        default="media",
        description="Nivel mínimo de alertas (baja, media, alta, critica)"
    )
    variables: Optional[List[str]] = Field(
        default=None,
        description="Variables a monitorear (None o [] = todas)"
    )


class SubscriptionUpdate(BaseModel):
    min_level: Optional[str] = Field(
        default=None,
        description="Nuevo nivel mínimo"
    )
    variables: Optional[List[str]] = Field(
        default=None,
        description="Nuevas variables a monitorear"
    )
    enabled: Optional[bool] = Field(
        default=None,
        description="Estado de la suscripción"
    )


class SubscriptionResponse(BaseModel):
    subscription_id: str
    user_id: str
    min_level: str
    variables: List[str]
    enabled: bool
    created_at: str


class SubscriptionListResponse(BaseModel):
    subscriptions: List[SubscriptionResponse]
    total: int


class SubscriptionDeleteResponse(BaseModel):
    success: bool
    message: str
    subscription_id: str
