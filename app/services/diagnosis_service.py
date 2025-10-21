"""Servicio de diagnóstico basado en reglas."""

from typing import List, Optional, Tuple
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.prediction import AlertLevel, TimeSeriesPoint

logger = get_logger(__name__)
settings = get_settings()


class DiagnosisService:
    """Servicio de diagnóstico meteorológico."""
    
    def diagnose(
        self,
        current: TimeSeriesPoint,
        previous: Optional[List[TimeSeriesPoint]] = None,
        predicted_precip: Optional[float] = None
    ) -> Tuple[AlertLevel, List[str], str]:
        """
        Realiza diagnóstico basado en reglas.
        
        Args:
            current: Condiciones actuales
            previous: Condiciones previas (para deltas)
            predicted_precip: Predicción de precipitación
        
        Returns:
            Tuple (nivel_alerta, reglas_activadas, recomendación)
        """
        triggered_rules = []
        alert_level = AlertLevel.BAJA
        
        # Calcular delta de temperatura si hay datos previos
        temp_delta_2h = None
        if previous and len(previous) >= 2:
            temp_delta_2h = current.temp_2m_c - previous[-2].temp_2m_c
        
        # Regla 1: RH alta + Temp descendente → ALTA
        if current.rh_2m_pct >= settings.rh_high and temp_delta_2h is not None:
            if temp_delta_2h <= settings.temp_drop_2h:
                alert_level = AlertLevel.ALTA
                triggered_rules.append(
                    f"RH_HIGH_TEMP_DROP: RH={current.rh_2m_pct:.1f}% >= {settings.rh_high}%, "
                    f"ΔTemp_2h={temp_delta_2h:.2f}°C <= {settings.temp_drop_2h}°C"
                )
        
        # Regla 2: RH media + Temp descendente → MEDIA (si aún no es ALTA)
        if alert_level == AlertLevel.BAJA:
            if current.rh_2m_pct >= settings.rh_medium and temp_delta_2h is not None:
                if temp_delta_2h <= settings.temp_drop_2h:
                    alert_level = AlertLevel.MEDIA
                    triggered_rules.append(
                        f"RH_MEDIUM_TEMP_DROP: RH={current.rh_2m_pct:.1f}% >= {settings.rh_medium}%, "
                        f"ΔTemp_2h={temp_delta_2h:.2f}°C <= {settings.temp_drop_2h}°C"
                    )
        
        # Regla 3: Viento calmo + RH alta → Refuerzo
        if current.wind_speed_2m_ms <= settings.wind_calm_ms and current.rh_2m_pct >= settings.rh_medium:
            triggered_rules.append(
                f"WIND_CALM_RH_HIGH: Viento={current.wind_speed_2m_ms:.2f} m/s <= {settings.wind_calm_ms}, "
                f"RH={current.rh_2m_pct:.1f}% >= {settings.rh_medium}%"
            )
            
            # Escalar alerta
            if alert_level == AlertLevel.BAJA:
                alert_level = AlertLevel.MEDIA
            elif alert_level == AlertLevel.MEDIA:
                alert_level = AlertLevel.ALTA
            elif alert_level == AlertLevel.ALTA:
                alert_level = AlertLevel.CRITICA
        
        # Regla 4: Predicción alta → Factor adicional
        if predicted_precip and predicted_precip >= settings.precip_event_mmhr * 2:
            triggered_rules.append(
                f"HIGH_PRECIP_PRED: Predicción={predicted_precip:.2f} mm/hr >= "
                f"{settings.precip_event_mmhr * 2:.2f} mm/hr"
            )
            
            if alert_level == AlertLevel.BAJA:
                alert_level = AlertLevel.MEDIA
            elif alert_level == AlertLevel.MEDIA:
                alert_level = AlertLevel.ALTA
        
        # Generar recomendación
        recommendation = self._get_recommendation(alert_level)
        
        logger.info(f"Diagnóstico: {alert_level.value} ({len(triggered_rules)} reglas)")
        
        return alert_level, triggered_rules, recommendation
    
    def _get_recommendation(self, level: AlertLevel) -> str:
        """Obtiene recomendación según nivel de alerta."""
        recommendations = {
            AlertLevel.CRITICA: (
                "[!] CRITICO: Condiciones altamente favorables para lluvia inminente. "
                "Activar protocolos de emergencia, asegurar drenajes, proteger equipos. "
                "Suspender operaciones al aire libre si es posible."
            ),
            AlertLevel.ALTA: (
                "[ALTA] ALERTA ALTA: Lluvia probable inminente. "
                "Asegurar drenajes, alertar a operaciones, proteger equipos expuestos. "
                "Preparar materiales de proteccion."
            ),
            AlertLevel.MEDIA: (
                "[MEDIA] ALERTA MEDIA: Condiciones favorables para lluvia en corto/mediano plazo. "
                "Monitorear de cerca, preparar protocolos de respuesta, "
                "informar al personal."
            ),
            AlertLevel.BAJA: (
                "[OK] BAJO: Sin senales fuertes de lluvia inminente. "
                "Continuar monitoreo rutinario, mantener protocolos estandar."
            )
        }
        
        return recommendations[level]

