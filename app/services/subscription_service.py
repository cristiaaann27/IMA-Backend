import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timezone
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.prediction import AlertLevel

logger = get_logger(__name__)
settings = get_settings()


class AlertSubscription:
    def __init__(
        self,
        subscription_id: str,
        user_id: str,
        min_level: str = "media",
        variables: Optional[List[str]] = None,
        enabled: bool = True,
        created_at: Optional[str] = None
    ):
        self.subscription_id = subscription_id
        self.user_id = user_id
        self.min_level = min_level
        self.variables = variables or []  # [] significa todas las variables
        self.enabled = enabled
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> dict:
        return {
            "subscription_id": self.subscription_id,
            "user_id": self.user_id,
            "min_level": self.min_level,
            "variables": self.variables,
            "enabled": self.enabled,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AlertSubscription":
        """Crea una suscripción desde un diccionario."""
        return cls(**data)
    
    def matches_alert(self, alert_level: str, variable: str) -> bool:
 
        if not self.enabled:
            return False
        
        # Verificar nivel
        level_order = {"baja": 1, "media": 2, "alta": 3, "critica": 4}
        if level_order.get(alert_level, 0) < level_order.get(self.min_level, 0):
            return False
        
        # Verificar variable (lista vacía = todas las variables)
        if self.variables and variable not in self.variables:
            return False
        
        return True


class SubscriptionService:
    
    def __init__(self):
        self.subscriptions_file = settings.project_root / "data" / "subscriptions.json"
        self._ensure_file_exists()
        self.subscriptions: Dict[str, AlertSubscription] = self._load_subscriptions()
    
    def _ensure_file_exists(self):
        self.subscriptions_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.subscriptions_file.exists():
            self.subscriptions_file.write_text(json.dumps([], indent=2))
    
    def _load_subscriptions(self) -> Dict[str, AlertSubscription]:
        try:
            with open(self.subscriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                subscriptions = {
                    sub["subscription_id"]: AlertSubscription.from_dict(sub)
                    for sub in data
                }
                logger.info(f"Cargadas {len(subscriptions)} suscripciones")
                return subscriptions
        except Exception as e:
            logger.error(f"Error cargando suscripciones: {e}")
            return {}
    
    def _save_subscriptions(self):
        try:
            data = [sub.to_dict() for sub in self.subscriptions.values()]
            with open(self.subscriptions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Guardadas {len(data)} suscripciones")
        except Exception as e:
            logger.error(f"Error guardando suscripciones: {e}")
    
    def create_subscription(
        self,
        user_id: str,
        min_level: str = "media",
        variables: Optional[List[str]] = None
    ) -> AlertSubscription:

        subscription_id = f"sub_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        existing = self.get_user_subscriptions(user_id, enabled_only=True)
        for sub in existing:
            if sub.min_level == min_level and sub.variables == (variables or []):
                logger.info(f"Suscripción similar ya existe: {sub.subscription_id}")
                return sub
        
        # Crear nueva suscripción
        subscription = AlertSubscription(
            subscription_id=subscription_id,
            user_id=user_id,
            min_level=min_level,
            variables=variables or []
        )
        
        self.subscriptions[subscription_id] = subscription
        self._save_subscriptions()
        
        logger.info(f"Suscripción creada: {subscription_id} para usuario {user_id}")
        return subscription
    
    def get_subscription(self, subscription_id: str) -> Optional[AlertSubscription]:
        return self.subscriptions.get(subscription_id)
    
    def get_user_subscriptions(
        self, 
        user_id: str, 
        enabled_only: bool = False
    ) -> List[AlertSubscription]:
        subs = [
            sub for sub in self.subscriptions.values()
            if sub.user_id == user_id
        ]
        
        if enabled_only:
            subs = [sub for sub in subs if sub.enabled]
        
        return subs
    
    def update_subscription(
        self,
        subscription_id: str,
        min_level: Optional[str] = None,
        variables: Optional[List[str]] = None,
        enabled: Optional[bool] = None
    ) -> Optional[AlertSubscription]:
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return None
        
        if min_level is not None:
            subscription.min_level = min_level
        if variables is not None:
            subscription.variables = variables
        if enabled is not None:
            subscription.enabled = enabled
        
        self._save_subscriptions()
        logger.info(f"Suscripción actualizada: {subscription_id}")
        
        return subscription
    
    def delete_subscription(self, subscription_id: str) -> bool:

        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            self._save_subscriptions()
            logger.info(f"Suscripción eliminada: {subscription_id}")
            return True
        return False
    
    def disable_subscription(self, subscription_id: str) -> bool:

        subscription = self.subscriptions.get(subscription_id)
        if subscription:
            subscription.enabled = False
            self._save_subscriptions()
            logger.info(f"Suscripción desactivada: {subscription_id}")
            return True
        return False
    
    def get_matching_subscriptions(
        self, 
        alert_level: str, 
        variable: str
    ) -> List[AlertSubscription]:

        matching = [
            sub for sub in self.subscriptions.values()
            if sub.matches_alert(alert_level, variable)
        ]
        return matching


# Singleton
_subscription_service_instance = None

def get_subscription_service() -> SubscriptionService:
    global _subscription_service_instance
    if _subscription_service_instance is None:
        _subscription_service_instance = SubscriptionService()
    return _subscription_service_instance
