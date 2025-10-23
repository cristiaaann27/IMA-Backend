"""Servicio de alertas para variables climáticas."""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from app.schemas.prediction import AlertLevel, TimeSeriesPoint
from app.core.logging import get_logger

logger = get_logger(__name__)

# Import lazy para evitar circular dependency
def get_alert_history_service():
    from app.services.alert_history_service import get_alert_history_service as _get
    return _get()


class WeatherAlert:
    """Representa una alerta climática."""
    
    def __init__(
        self,
        level: AlertLevel,
        variable: str,
        value: float,
        threshold: float,
        message: str,
        timestamp: datetime
    ):
        self.level = level
        self.variable = variable
        self.value = value
        self.threshold = threshold
        self.message = message
        self.timestamp = timestamp
    
    def to_dict(self) -> dict:
        """Convierte la alerta a diccionario."""
        # Determinar minutos hasta próxima actualización según nivel
        update_intervals = {
            "bajo": 1440,      # 24 horas
            "medio": 720,      # 12 horas
            "alto": 180,       # 3 horas
            "critico": 30      # 30 minutos
        }
        
        return {
            "level": self.level.value,
            "variable": self.variable,
            "value": round(self.value, 2),
            "threshold": round(self.threshold, 2),
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "detection_time": self.timestamp.strftime("%H:%M"),
            "next_update_minutes": update_intervals.get(self.level.value, 720),
            "color": self.level.color
        }


class AlertService:
    """Servicio para evaluar y generar alertas climáticas."""
    
    # Umbrales para variables climáticas
    THRESHOLDS = {
        "rh_2m_pct": {
            "media": 80.0,  # Humedad >= 80% es nivel MEDIO
            "alta": 90.0,   # Humedad >= 90% es nivel ALTO
        },
        "temp_2m_c": {
            "media_low": 10.0,   # Temperatura <= 10°C es nivel MEDIO
            "alta_low": 5.0,     # Temperatura <= 5°C es nivel ALTO
            "media_high": 30.0,  # Temperatura >= 30°C es nivel MEDIO
            "alta_high": 35.0,   # Temperatura >= 35°C es nivel ALTO
        },
        "wind_speed_2m_ms": {
            "media": 10.0,  # Viento >= 10 m/s es nivel MEDIO
            "alta": 15.0,   # Viento >= 15 m/s es nivel ALTO
        },
        "temp_delta": {
            "media": 3.0,   # Cambio de temperatura >= 3°C es nivel MEDIO
            "alta": 5.0,    # Cambio de temperatura >= 5°C es nivel ALTO
        },
        "rh_delta": {
            "media": 15.0,  # Cambio de humedad >= 15% es nivel MEDIO
            "alta": 25.0,   # Cambio de humedad >= 25% es nivel ALTO
        }
    }
    
    def __init__(self):
        self.active_alerts: List[WeatherAlert] = []
    
    def evaluate_conditions(
        self,
        current: TimeSeriesPoint,
        previous: Optional[List[TimeSeriesPoint]] = None
    ) -> List[WeatherAlert]:
        """
        Evalúa las condiciones climáticas y genera alertas.
        
        Args:
            current: Condiciones actuales
            previous: Condiciones previas (opcional, para calcular deltas)
        
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        timestamp = datetime.now(timezone.utc)
        
        # 1. Evaluar humedad relativa
        rh_alerts = self._evaluate_humidity(current.rh_2m_pct, timestamp)
        alerts.extend(rh_alerts)
        
        # 2. Evaluar temperatura
        temp_alerts = self._evaluate_temperature(current.temp_2m_c, timestamp)
        alerts.extend(temp_alerts)
        
        # 3. Evaluar velocidad del viento
        wind_alerts = self._evaluate_wind_speed(current.wind_speed_2m_ms, timestamp)
        alerts.extend(wind_alerts)
        
        # 4. Evaluar cambios (deltas) si hay datos previos
        if previous and len(previous) > 0:
            delta_alerts = self._evaluate_deltas(current, previous, timestamp)
            alerts.extend(delta_alerts)
        
        # 5. Evaluar condiciones combinadas
        combined_alerts = self._evaluate_combined_conditions(current, timestamp)
        alerts.extend(combined_alerts)
        
        # Actualizar alertas activas
        self.active_alerts = alerts
        
        # Guardar en historial
        if alerts:
            try:
                history_service = get_alert_history_service()
                for alert in alerts:
                    history_service.add_alert(
                        level=alert.level.value,
                        variable=alert.variable,
                        value=alert.value,
                        threshold=alert.threshold,
                        message=alert.message
                    )
            except Exception as e:
                logger.error(f"Error guardando alertas en historial: {e}")
        
        logger.info(f"Evaluación de alertas: {len(alerts)} alertas generadas")
        return alerts
    
    def _evaluate_humidity(self, rh: float, timestamp: datetime) -> List[WeatherAlert]:
        """Evalúa la humedad relativa."""
        alerts = []
        hora_deteccion = timestamp.strftime("%H:%M")
        
        if rh >= self.THRESHOLDS["rh_2m_pct"]["alta"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.ALTO,
                variable="Humedad Relativa",
                value=rh,
                threshold=self.THRESHOLDS["rh_2m_pct"]["alta"],
                message=f"🔴 Alerta La Dorada - NIVEL ALTO\n"
                       f"Se detectó humedad elevada ({rh:.1f}%), nivel de riesgo ALTO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 3-6 horas.\n"
                       f"Recomendación: Lluvias intensas detectadas. Evita transitar o realizar labores cerca de riberas o quebradas. Mantén encendido el celular y verifica rutas seguras hacia zonas altas.",
                timestamp=timestamp
            ))
        elif rh >= self.THRESHOLDS["rh_2m_pct"]["media"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.MEDIO,
                variable="Humedad Relativa",
                value=rh,
                threshold=self.THRESHOLDS["rh_2m_pct"]["media"],
                message=f"🟡 Alerta La Dorada - NIVEL MEDIO\n"
                       f"Se detectó humedad moderada ({rh:.1f}%), nivel de riesgo MEDIO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 12 horas.\n"
                       f"Recomendación: Se registran lluvias moderadas. Evita acumular residuos o materiales cerca de desagües. Permanece atento a actualizaciones del sistema.",
                timestamp=timestamp
            ))
        
        return alerts
    
    def _evaluate_temperature(self, temp: float, timestamp: datetime) -> List[WeatherAlert]:
        """Evalúa la temperatura."""
        alerts = []
        hora_deteccion = timestamp.strftime("%H:%M")
        
        # Temperaturas bajas
        if temp <= self.THRESHOLDS["temp_2m_c"]["alta_low"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.ALTO,
                variable="Temperatura",
                value=temp,
                threshold=self.THRESHOLDS["temp_2m_c"]["alta_low"],
                message=f"🔴 Alerta La Dorada - NIVEL ALTO\n"
                       f"Se detectó temperatura muy baja ({temp:.1f}°C), nivel de riesgo ALTO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 3-6 horas.\n"
                       f"Recomendación: Protege cultivos sensibles cubriéndolos. Mantente informado por los canales oficiales del IDEAM.",
                timestamp=timestamp
            ))
        elif temp <= self.THRESHOLDS["temp_2m_c"]["media_low"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.MEDIO,
                variable="Temperatura",
                value=temp,
                threshold=self.THRESHOLDS["temp_2m_c"]["media_low"],
                message=f"🟡 Alerta La Dorada - NIVEL MEDIO\n"
                       f"Se detectó temperatura baja ({temp:.1f}°C), nivel de riesgo MEDIO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 12 horas.\n"
                       f"Recomendación: Retrasa labores agrícolas en zonas bajas hasta que el terreno se estabilice.",
                timestamp=timestamp
            ))
        
        # Temperaturas altas
        if temp >= self.THRESHOLDS["temp_2m_c"]["alta_high"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.ALTO,
                variable="Temperatura",
                value=temp,
                threshold=self.THRESHOLDS["temp_2m_c"]["alta_high"],
                message=f"🔴 Alerta La Dorada - NIVEL ALTO\n"
                       f"Se detectó temperatura muy elevada ({temp:.1f}°C), nivel de riesgo ALTO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 3-6 horas.\n"
                       f"Recomendación: Recuerda hidratarte y protegerte del sol durante las actividades agrícolas. Prepara un kit de emergencia.",
                timestamp=timestamp
            ))
        elif temp >= self.THRESHOLDS["temp_2m_c"]["media_high"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.MEDIO,
                variable="Temperatura",
                value=temp,
                threshold=self.THRESHOLDS["temp_2m_c"]["media_high"],
                message=f"🟡 Alerta La Dorada - NIVEL MEDIO\n"
                       f"Se detectó temperatura elevada ({temp:.1f}°C), nivel de riesgo MEDIO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 12 horas.\n"
                       f"Recomendación: Recuerda hidratarte y protegerte del sol. Ideal para labores agrícolas con precaución.",
                timestamp=timestamp
            ))
        
        return alerts
    
    def _evaluate_wind_speed(self, wind_speed: float, timestamp: datetime) -> List[WeatherAlert]:
        """Evalúa la velocidad del viento."""
        alerts = []
        hora_deteccion = timestamp.strftime("%H:%M")
        
        if wind_speed >= self.THRESHOLDS["wind_speed_2m_ms"]["alta"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.ALTO,
                variable="Velocidad del Viento",
                value=wind_speed,
                threshold=self.THRESHOLDS["wind_speed_2m_ms"]["alta"],
                message=f"🔴 Alerta La Dorada - NIVEL ALTO\n"
                       f"Se detectó viento fuerte ({wind_speed:.1f} m/s), nivel de riesgo ALTO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 3-6 horas.\n"
                       f"Recomendación: Evita trabajar con maquinaria cerca de zonas ribereñas o inestables. Asegura estructuras y equipos.",
                timestamp=timestamp
            ))
        elif wind_speed >= self.THRESHOLDS["wind_speed_2m_ms"]["media"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.MEDIO,
                variable="Velocidad del Viento",
                value=wind_speed,
                threshold=self.THRESHOLDS["wind_speed_2m_ms"]["media"],
                message=f"🟡 Alerta La Dorada - NIVEL MEDIO\n"
                       f"Se detectó viento moderado ({wind_speed:.1f} m/s), nivel de riesgo MEDIO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 12 horas.\n"
                       f"Recomendación: Revisa el estado de techos y canaletas para prevenir filtraciones.",
                timestamp=timestamp
            ))
        
        return alerts
    
    def _evaluate_deltas(
        self,
        current: TimeSeriesPoint,
        previous: List[TimeSeriesPoint],
        timestamp: datetime
    ) -> List[WeatherAlert]:
        """Evalúa cambios rápidos en variables."""
        alerts = []
        hora_deteccion = timestamp.strftime("%H:%M")
        
        if len(previous) == 0:
            return alerts
        
        # Usar el punto más reciente para calcular deltas
        prev = previous[-1]
        
        # Delta de temperatura
        temp_delta = abs(current.temp_2m_c - prev.temp_2m_c)
        if temp_delta >= self.THRESHOLDS["temp_delta"]["alta"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.ALTO,
                variable="Cambio de Temperatura",
                value=temp_delta,
                threshold=self.THRESHOLDS["temp_delta"]["alta"],
                message=f"🔴 Alerta La Dorada - NIVEL ALTO\n"
                       f"Se detectó cambio brusco de temperatura ({temp_delta:.1f}°C), nivel de riesgo ALTO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 3-6 horas.\n"
                       f"Recomendación: Inestabilidad atmosférica. Mantente informado por los canales oficiales del IDEAM y el Comité Municipal de Gestión del Riesgo.",
                timestamp=timestamp
            ))
        elif temp_delta >= self.THRESHOLDS["temp_delta"]["media"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.MEDIO,
                variable="Cambio de Temperatura",
                value=temp_delta,
                threshold=self.THRESHOLDS["temp_delta"]["media"],
                message=f"🟡 Alerta La Dorada - NIVEL MEDIO\n"
                       f"Se detectó cambio notable de temperatura ({temp_delta:.1f}°C), nivel de riesgo MEDIO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 12 horas.\n"
                       f"Recomendación: Monitorea de cerca las condiciones climáticas. Comunica a los vecinos sobre las condiciones.",
                timestamp=timestamp
            ))
        
        # Delta de humedad
        rh_delta = abs(current.rh_2m_pct - prev.rh_2m_pct)
        if rh_delta >= self.THRESHOLDS["rh_delta"]["alta"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.ALTO,
                variable="Cambio de Humedad",
                value=rh_delta,
                threshold=self.THRESHOLDS["rh_delta"]["alta"],
                message=f"🔴 Alerta La Dorada - NIVEL ALTO\n"
                       f"Se detectó cambio brusco de humedad ({rh_delta:.1f}%), nivel de riesgo ALTO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 3-6 horas.\n"
                       f"Recomendación: Condiciones cambiantes. Ubica rutas seguras hacia zonas altas y asegúrate de que tu familia conozca el punto de encuentro.",
                timestamp=timestamp
            ))
        elif rh_delta >= self.THRESHOLDS["rh_delta"]["media"]:
            alerts.append(WeatherAlert(
                level=AlertLevel.MEDIO,
                variable="Cambio de Humedad",
                value=rh_delta,
                threshold=self.THRESHOLDS["rh_delta"]["media"],
                message=f"🟡 Alerta La Dorada - NIVEL MEDIO\n"
                       f"Se detectó cambio notable de humedad ({rh_delta:.1f}%), nivel de riesgo MEDIO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 12 horas.\n"
                       f"Recomendación: Se pronostican lluvias continuas en las próximas horas. Revisa el estado de los drenajes o canales.",
                timestamp=timestamp
            ))
        
        return alerts
    
    def _evaluate_combined_conditions(
        self,
        current: TimeSeriesPoint,
        timestamp: datetime
    ) -> List[WeatherAlert]:
        """Evalúa condiciones combinadas que indican riesgo."""
        alerts = []
        hora_deteccion = timestamp.strftime("%H:%M")
        
        # Condición: Alta humedad + Temperatura moderada + Viento bajo
        # (Condiciones ideales para precipitación)
        if (current.rh_2m_pct >= 85.0 and 
            15.0 <= current.temp_2m_c <= 25.0 and 
            current.wind_speed_2m_ms < 2.0):
            alerts.append(WeatherAlert(
                level=AlertLevel.ALTO,
                variable="Condiciones Combinadas",
                value=current.rh_2m_pct,
                threshold=85.0,
                message=f"🔴 Alerta La Dorada - NIVEL ALTO\n"
                       f"Se detectó condiciones óptimas para precipitación (RH={current.rh_2m_pct:.1f}%, T={current.temp_2m_c:.1f}°C, V={current.wind_speed_2m_ms:.1f}m/s), nivel de riesgo ALTO.\n"
                       f"Detectado desde las {hora_deteccion}. Próxima actualización en 3-6 horas.\n"
                       f"Recomendación: Lluvias intensas detectadas. Evita cruzar ríos, quebradas o zonas bajas. Prepara un kit de emergencia con documentos, linterna, radio y medicamentos.",
                timestamp=timestamp
            ))
        
        # Condición: Humedad muy alta + Viento muy bajo (calma)
        if current.rh_2m_pct >= 90.0 and current.wind_speed_2m_ms < 1.0:
            alerts.append(WeatherAlert(
                level=AlertLevel.CRITICO,
                variable="Condiciones Combinadas",
                value=current.rh_2m_pct,
                threshold=90.0,
                message=f"🔴 Alerta La Dorada - NIVEL CRÍTICO\n"
                       f"Se detectó calma con humedad extrema ({current.rh_2m_pct:.1f}%), nivel de riesgo CRÍTICO.\n"
                       f"Detectado desde las {hora_deteccion}. Actualizaciones cada 30 minutos.\n"
                       f"Recomendación: ¡Emergencia climática! Evacuación inmediata recomendada. Dirígete a los puntos seguros designados por la Alcaldía. Informa tu ubicación a las autoridades.",
                timestamp=timestamp
            ))
        
        return alerts
    
    def get_active_alerts(self) -> List[Dict]:
        """Retorna las alertas activas."""
        return [alert.to_dict() for alert in self.active_alerts]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Dict]:
        """Retorna alertas filtradas por nivel."""
        return [
            alert.to_dict() 
            for alert in self.active_alerts 
            if alert.level == level
        ]
    
    def has_alerts(self, min_level: AlertLevel = AlertLevel.MEDIO) -> bool:
        """Verifica si hay alertas activas de un nivel mínimo."""
        min_severity = min_level.severity
        return any(
            alert.level.severity >= min_severity 
            for alert in self.active_alerts
        )
    
    def clear_alerts(self):
        """Limpia todas las alertas activas."""
        self.active_alerts = []
        logger.info("Alertas limpiadas")


# Singleton
_alert_service_instance = None

def get_alert_service() -> AlertService:
    """Obtiene la instancia singleton del servicio de alertas."""
    global _alert_service_instance
    if _alert_service_instance is None:
        _alert_service_instance = AlertService()
    return _alert_service_instance
