"""Servicio de historial de alertas."""

import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AlertHistoryEntry:
    """Representa una entrada en el historial de alertas."""
    
    def __init__(
        self,
        entry_id: str,
        level: str,
        variable: str,
        value: float,
        threshold: float,
        message: str,
        timestamp: str,
        notified_users: Optional[List[str]] = None,
        resolved_at: Optional[str] = None
    ):
        self.entry_id = entry_id
        self.level = level
        self.variable = variable
        self.value = value
        self.threshold = threshold
        self.message = message
        self.timestamp = timestamp
        self.notified_users = notified_users or []
        self.resolved_at = resolved_at
    
    def to_dict(self) -> dict:
        """Convierte la entrada a diccionario."""
        return {
            "entry_id": self.entry_id,
            "level": self.level,
            "variable": self.variable,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp,
            "notified_users": self.notified_users,
            "resolved_at": self.resolved_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AlertHistoryEntry":
        """Crea una entrada desde un diccionario."""
        return cls(**data)


class AlertHistoryService:
    """Servicio para gestionar el historial de alertas."""
    
    def __init__(self, max_history_days: int = 30):
        """
        Inicializa el servicio de historial.
        
        Args:
            max_history_days: Días máximos de historial a mantener
        """
        self.history_file = settings.project_root / "data" / "alert_history.json"
        self.max_history_days = max_history_days
        self._ensure_file_exists()
        self.history: List[AlertHistoryEntry] = self._load_history()
    
    def _ensure_file_exists(self):
        """Asegura que el archivo de historial existe."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self.history_file.write_text(json.dumps([], indent=2))
    
    def _load_history(self) -> List[AlertHistoryEntry]:
        """Carga historial desde archivo."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                history = [AlertHistoryEntry.from_dict(entry) for entry in data]
                logger.info(f"Cargadas {len(history)} entradas de historial")
                return history
        except Exception as e:
            logger.error(f"Error cargando historial: {e}")
            return []
    
    def _save_history(self):
        """Guarda historial en archivo."""
        try:
            data = [entry.to_dict() for entry in self.history]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Guardadas {len(data)} entradas de historial")
        except Exception as e:
            logger.error(f"Error guardando historial: {e}")
    
    def add_alert(
        self,
        level: str,
        variable: str,
        value: float,
        threshold: float,
        message: str,
        notified_users: Optional[List[str]] = None
    ) -> AlertHistoryEntry:
        """
        Agrega una alerta al historial.
        
        Args:
            level: Nivel de la alerta
            variable: Variable monitoreada
            value: Valor actual
            threshold: Umbral superado
            message: Mensaje de la alerta
            notified_users: Lista de usuarios notificados
        
        Returns:
            Entrada de historial creada
        """
        timestamp = datetime.now(timezone.utc)
        entry_id = f"alert_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        entry = AlertHistoryEntry(
            entry_id=entry_id,
            level=level,
            variable=variable,
            value=value,
            threshold=threshold,
            message=message,
            timestamp=timestamp.isoformat(),
            notified_users=notified_users or []
        )
        
        self.history.append(entry)
        self._save_history()
        
        logger.info(f"Alerta agregada al historial: {entry_id} ({level} - {variable})")
        return entry
    
    def get_history(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        level: Optional[str] = None,
        variable: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[AlertHistoryEntry]:
        """
        Obtiene historial de alertas con filtros opcionales.
        
        Args:
            from_date: Fecha inicial (inclusive)
            to_date: Fecha final (inclusive)
            level: Filtrar por nivel de alerta
            variable: Filtrar por variable
            limit: Número máximo de resultados
        
        Returns:
            Lista de entradas de historial
        """
        filtered = self.history.copy()
        
        # Filtrar por fecha
        if from_date:
            filtered = [
                entry for entry in filtered
                if datetime.fromisoformat(entry.timestamp) >= from_date
            ]
        
        if to_date:
            filtered = [
                entry for entry in filtered
                if datetime.fromisoformat(entry.timestamp) <= to_date
            ]
        
        # Filtrar por nivel
        if level:
            filtered = [
                entry for entry in filtered
                if entry.level == level
            ]
        
        # Filtrar por variable
        if variable:
            filtered = [
                entry for entry in filtered
                if entry.variable == variable
            ]
        
        # Ordenar por timestamp descendente (más recientes primero)
        filtered.sort(
            key=lambda x: datetime.fromisoformat(x.timestamp),
            reverse=True
        )
        
        # Limitar resultados
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_recent_alerts(self, hours: int = 24) -> List[AlertHistoryEntry]:
        """
        Obtiene alertas recientes.
        
        Args:
            hours: Número de horas hacia atrás
        
        Returns:
            Lista de alertas recientes
        """
        from_date = datetime.now(timezone.utc) - timedelta(hours=hours)
        return self.get_history(from_date=from_date)
    
    def get_statistics(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict:
        """
        Obtiene estadísticas del historial de alertas.
        
        Args:
            from_date: Fecha inicial
            to_date: Fecha final
        
        Returns:
            Diccionario con estadísticas
        """
        entries = self.get_history(from_date=from_date, to_date=to_date)
        
        if not entries:
            return {
                "total_alerts": 0,
                "by_level": {},
                "by_variable": {},
                "most_common_variable": None,
                "highest_level": None
            }
        
        # Contar por nivel
        by_level = {}
        for entry in entries:
            by_level[entry.level] = by_level.get(entry.level, 0) + 1
        
        # Contar por variable
        by_variable = {}
        for entry in entries:
            by_variable[entry.variable] = by_variable.get(entry.variable, 0) + 1
        
        # Variable más común
        most_common_variable = max(by_variable.items(), key=lambda x: x[1])[0]
        
        # Nivel más alto detectado
        level_order = {"baja": 1, "media": 2, "alta": 3, "critica": 4}
        highest_level = max(by_level.keys(), key=lambda x: level_order.get(x, 0))
        
        return {
            "total_alerts": len(entries),
            "by_level": by_level,
            "by_variable": by_variable,
            "most_common_variable": most_common_variable,
            "highest_level": highest_level,
            "date_range": {
                "from": entries[-1].timestamp if entries else None,
                "to": entries[0].timestamp if entries else None
            }
        }
    
    def cleanup_old_alerts(self):
        """Elimina alertas más antiguas que max_history_days."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_history_days)
        
        initial_count = len(self.history)
        self.history = [
            entry for entry in self.history
            if datetime.fromisoformat(entry.timestamp) >= cutoff_date
        ]
        
        removed_count = initial_count - len(self.history)
        if removed_count > 0:
            self._save_history()
            logger.info(f"Limpieza de historial: {removed_count} alertas eliminadas (más de {self.max_history_days} días)")
    
    def mark_as_notified(self, entry_id: str, user_id: str):
        """
        Marca que un usuario fue notificado de una alerta.
        
        Args:
            entry_id: ID de la entrada
            user_id: ID del usuario notificado
        """
        for entry in self.history:
            if entry.entry_id == entry_id:
                if user_id not in entry.notified_users:
                    entry.notified_users.append(user_id)
                    self._save_history()
                    logger.debug(f"Usuario {user_id} marcado como notificado para {entry_id}")
                break
    
    def resolve_alert(self, entry_id: str):
        """
        Marca una alerta como resuelta.
        
        Args:
            entry_id: ID de la entrada
        """
        for entry in self.history:
            if entry.entry_id == entry_id:
                entry.resolved_at = datetime.now(timezone.utc).isoformat()
                self._save_history()
                logger.info(f"Alerta resuelta: {entry_id}")
                break


# Singleton
_alert_history_service_instance = None

def get_alert_history_service() -> AlertHistoryService:
    """Obtiene la instancia singleton del servicio de historial."""
    global _alert_history_service_instance
    if _alert_history_service_instance is None:
        _alert_history_service_instance = AlertHistoryService()
    return _alert_history_service_instance
