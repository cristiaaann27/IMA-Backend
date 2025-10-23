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
        alert_level = AlertLevel.BAJO
        
        # Calcular delta de temperatura si hay datos previos
        temp_delta_2h = None
        if previous and len(previous) >= 2:
            temp_delta_2h = current.temp_2m_c - previous[-2].temp_2m_c
        
        # Regla 1: RH alta + Temp descendente → ALTO
        if current.rh_2m_pct >= settings.rh_high and temp_delta_2h is not None:
            if temp_delta_2h <= settings.temp_drop_2h:
                alert_level = AlertLevel.ALTO
                triggered_rules.append(
                    f"RH_HIGH_TEMP_DROP: RH={current.rh_2m_pct:.1f}% >= {settings.rh_high}%, "
                    f"ΔTemp_2h={temp_delta_2h:.2f}°C <= {settings.temp_drop_2h}°C"
                )
        
        # Regla 2: RH media + Temp descendente → MEDIO (si aún no es ALTO)
        if alert_level == AlertLevel.BAJO:
            if current.rh_2m_pct >= settings.rh_medium and temp_delta_2h is not None:
                if temp_delta_2h <= settings.temp_drop_2h:
                    alert_level = AlertLevel.MEDIO
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
            if alert_level == AlertLevel.BAJO:
                alert_level = AlertLevel.MEDIO
            elif alert_level == AlertLevel.MEDIO:
                alert_level = AlertLevel.ALTO
            elif alert_level == AlertLevel.ALTO:
                alert_level = AlertLevel.CRITICO
        
        # Regla 4: Predicción alta → Factor adicional
        if predicted_precip and predicted_precip >= settings.precip_event_mmhr * 2:
            triggered_rules.append(
                f"HIGH_PRECIP_PRED: Predicción={predicted_precip:.2f} mm/hr >= "
                f"{settings.precip_event_mmhr * 2:.2f} mm/hr"
            )
            
            if alert_level == AlertLevel.BAJO:
                alert_level = AlertLevel.MEDIO
            elif alert_level == AlertLevel.MEDIO:
                alert_level = AlertLevel.ALTO
        
        # Generar recomendación
        recommendation = self._get_recommendation(alert_level)
        
        logger.info(f"Diagnóstico: {alert_level.value} ({len(triggered_rules)} reglas)")
        
        return alert_level, triggered_rules, recommendation
    
    def _get_recommendation(self, level: AlertLevel) -> str:
        """Obtiene recomendación según nivel de alerta."""
        recommendations = {
            AlertLevel.CRITICO: (
                "🔴 NIVEL CRÍTICO — Emergencia\n"
                "¡Emergencia climática! Se recomienda evacuar zonas ribereñas y dirigirse a puntos seguros definidos por la Alcaldía. "
                "Contacta a Defensa Civil, Bomberos o Cruz Roja si detectas aumento repentino del nivel del agua. "
                "Desconecta equipos eléctricos y corta el suministro de energía si hay riesgo de inundación. "
                "Sigue las instrucciones de los organismos oficiales. Evita la desinformación y prioriza la seguridad de tu familia. "
                "El sistema activará mensajes automáticos cada 30 minutos con actualizaciones en tiempo real."
            ),
            AlertLevel.ALTO: (
                "🟠 NIVEL ALTO — Riesgo significativo\n"
                "Lluvias intensas detectadas. Evita transitar o realizar labores cerca de riberas o quebradas. "
                "Mantén encendido el celular y verifica rutas seguras hacia zonas altas. "
                "Protege cultivos sensibles cubriéndolos o drenando el exceso de agua. "
                "Sigue las recomendaciones del Consejo Municipal de Gestión del Riesgo (CMGRD). "
                "Prepara un kit de emergencia con documentos, linterna, radio y medicamentos. "
                "Mantente informado por los canales oficiales del IDEAM. Actualizaciones cada 3-6 horas."
            ),
            AlertLevel.MEDIO: (
                "🟡 NIVEL MEDIO — Riesgo moderado\n"
                "Se registran lluvias moderadas. Evita acumular residuos o materiales cerca de desagües. "
                "Retrasa labores agrícolas en zonas bajas hasta que el terreno se estabilice. "
                "Revisa el estado de techos y canaletas para prevenir filtraciones. "
                "Permanece atento a actualizaciones del sistema y reportes del IDEAM o Alcaldía. "
                "Comunica a los vecinos sobre las condiciones climáticas y mantén activos los grupos de alerta. "
                "Monitorea el caudal del río Magdalena con la app cada 3 horas. Actualizaciones cada 12 horas."
            ),
            AlertLevel.BAJO: (
                "🟢 NIVEL BAJO — Condiciones normales\n"
                "Condiciones meteorológicas estables. Aprovecha para revisar los canales de drenaje y mantener limpios los alrededores. "
                "Recuerda hidratarte y protegerte del sol durante las actividades agrícolas. "
                "Revisa el estado de los tanques y reservorios para conservar agua limpia en caso de sequía futura. "
                "Monitoreo constante activo. No se reportan alertas, pero se recomienda estar atentos a los próximos reportes. "
                "Ideal para labores agrícolas. Aprovecha para realizar mantenimiento preventivo en equipos o cultivos. "
                "Información diaria o semanal preventiva."
            )
        }
        
        return recommendations[level]

