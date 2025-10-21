"""
Servicio de diagnóstico inteligente con umbrales adaptativos.

Usa correlaciones reales entre variables para diagnóstico dinámico.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.prediction import AlertLevel, TimeSeriesPoint

logger = get_logger(__name__)
settings = get_settings()


class SmartDiagnosisService:
    """
    Servicio de diagnóstico inteligente con umbrales adaptativos.
    
    En lugar de umbrales fijos, calcula la probabilidad de precipitación
    basándose en:
    1. Correlaciones conocidas (RH vs precip, Temp vs precip)
    2. Condiciones actuales vs patrones históricos
    3. Velocidad de cambio de variables
    4. Combinación de factores de riesgo
    """
    
    # Correlaciones documentadas del análisis
    KNOWN_CORRELATIONS = {
        "precip_vs_rh": 0.32,      # +0.32: A mayor RH, más precipitación
        "precip_vs_temp": -0.21,   # -0.21: A menor temp, más precipitación
        "rh_vs_temp": -0.87,       # -0.87: Fuerte correlación inversa
    }
    
    def __init__(self):
        """Inicializa el servicio."""
        # Cargar estadísticas históricas si existen
        self.historical_stats = self._load_historical_stats()
    
    def _load_historical_stats(self) -> Dict:
        """
        Carga estadísticas históricas de los datos.
        
        En producción, esto vendría de la base de datos.
        Por ahora usa valores razonables de tus datos.
        """
        return {
            "rh_mean": 70.0,
            "rh_std": 20.0,
            "temp_mean": 20.0,
            "temp_std": 5.0,
            "wind_mean": 3.0,
            "wind_std": 2.0,
            "precip_mean": 4.15,  # Del análisis: 4.15 mm/hr
            "precip_std": 7.0,
        }
    
    def calculate_risk_score(
        self,
        current: TimeSeriesPoint,
        previous: Optional[List[TimeSeriesPoint]] = None,
        predicted_precip: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calcula score de riesgo basado en múltiples factores.
        
        Args:
            current: Condiciones actuales
            previous: Condiciones previas
            predicted_precip: Predicción del modelo
        
        Returns:
            Diccionario con scores de cada factor
        """
        scores = {}
        
        # 1. Score de Humedad Relativa (más alto = más riesgo)
        # Basado en correlación precip vs RH = +0.32
        rh_normalized = (current.rh_2m_pct - self.historical_stats["rh_mean"]) / self.historical_stats["rh_std"]
        scores["rh_score"] = np.clip(rh_normalized * 0.32, 0, 1)  # Usar correlación
        
        # 2. Score de Temperatura (más bajo = más riesgo)
        # Basado en correlación precip vs temp = -0.21
        temp_normalized = (current.temp_2m_c - self.historical_stats["temp_mean"]) / self.historical_stats["temp_std"]
        scores["temp_score"] = np.clip(-temp_normalized * 0.21, 0, 1)  # Correlación negativa
        
        # 3. Score de Viento Calmo (más bajo = más riesgo)
        # Viento bajo favorece acumulación de humedad
        wind_normalized = current.wind_speed_2m_ms / self.historical_stats["wind_mean"]
        scores["wind_score"] = np.clip(1.0 - wind_normalized, 0, 1)  # Invertir: bajo viento = alto score
        
        # 4. Score de Delta de Temperatura (enfriamiento rápido)
        if previous and isinstance(previous, list) and len(previous) >= 2:
            temp_delta_2h = current.temp_2m_c - previous[-2].temp_2m_c
            # Enfriamiento rápido = mayor riesgo
            scores["temp_delta_score"] = np.clip(-temp_delta_2h / 5.0, 0, 1)  # Normalizar por 5°C
        else:
            scores["temp_delta_score"] = 0.0
        
        # 5. Score del Modelo (si está disponible)
        if predicted_precip is not None:
            # Normalizar por media histórica
            precip_normalized = predicted_precip / max(self.historical_stats["precip_mean"], 1.0)
            scores["model_score"] = np.clip(precip_normalized, 0, 1)
        else:
            scores["model_score"] = 0.0
        
        # 6. Score Combinado (RH + Temp: correlación -0.87)
        # Cuando RH alto y Temp bajo = condiciones ideales para lluvia
        rh_high = (current.rh_2m_pct - 50) / 50  # Normalizar 50-100
        temp_low = (30 - current.temp_2m_c) / 30  # Normalizar (temp baja = score alto)
        scores["rh_temp_interaction"] = np.clip((rh_high + temp_low) / 2, 0, 1)
        
        return scores
    
    def diagnose_adaptive(
        self,
        current: TimeSeriesPoint,
        previous: Optional[List[TimeSeriesPoint]] = None,
        predicted_precip: Optional[float] = None
    ) -> Tuple[AlertLevel, List[str], str, Dict]:
        """
        Diagnóstico adaptativo basado en scores de riesgo.
        
        Args:
            current: Condiciones actuales
            previous: Condiciones previas
            predicted_precip: Predicción del modelo
        
        Returns:
            Tuple (nivel_alerta, reglas_activadas, recomendación, scores)
        """
        # Calcular scores de riesgo
        scores = self.calculate_risk_score(current, previous, predicted_precip)
        
        # Calcular score total (promedio ponderado)
        weights = {
            "rh_score": 0.25,           # 25% - Importante
            "temp_score": 0.15,         # 15% - Moderado
            "wind_score": 0.10,         # 10% - Menor
            "temp_delta_score": 0.15,   # 15% - Cambios rápidos
            "model_score": 0.20,        # 20% - Predicción del modelo
            "rh_temp_interaction": 0.15 # 15% - Sinergia RH-Temp
        }
        
        total_score = sum(scores[k] * weights[k] for k in weights.keys())
        
        # Determinar nivel de alerta basado en score total
        triggered_rules = []
        
        # Documentar qué factores contribuyeron
        for factor, score in scores.items():
            if score > 0.5:
                triggered_rules.append(
                    f"{factor.upper()}: {score:.2f} (contribución significativa)"
                )
        
        # Añadir valores actuales para contexto
        triggered_rules.append(
            f"CONDICIONES: RH={current.rh_2m_pct:.1f}%, "
            f"Temp={current.temp_2m_c:.1f}°C, "
            f"Viento={current.wind_speed_2m_ms:.1f}m/s"
        )
        
        if predicted_precip is not None:
            triggered_rules.append(
                f"PREDICCION_MODELO: {predicted_precip:.2f} mm/hr"
            )
        
        # Determinar nivel de alerta por score total
        if total_score >= 0.75:
            alert_level = AlertLevel.CRITICA
        elif total_score >= 0.55:
            alert_level = AlertLevel.ALTA
        elif total_score >= 0.35:
            alert_level = AlertLevel.MEDIA
        else:
            alert_level = AlertLevel.BAJA
        
        # Generar recomendación
        recommendation = self._get_recommendation_adaptive(alert_level, total_score, scores)
        
        logger.info(
            f"Diagnóstico adaptativo: Score={total_score:.3f} → {alert_level.value} "
            f"(RH={scores['rh_score']:.2f}, Temp={scores['temp_score']:.2f}, "
            f"Model={scores['model_score']:.2f})"
        )
        
        return alert_level, triggered_rules, recommendation, scores
    
    def _get_recommendation_adaptive(
        self,
        level: AlertLevel,
        total_score: float,
        scores: Dict[str, float]
    ) -> str:
        """
        Genera recomendación adaptativa con detalles específicos.
        
        Args:
            level: Nivel de alerta
            total_score: Score total de riesgo
            scores: Scores individuales
        
        Returns:
            Recomendación detallada
        """
        base_recommendations = {
            AlertLevel.CRITICA: (
                "[!] CRITICO: Condiciones altamente favorables para lluvia inminente. "
                "Activar protocolos de emergencia, asegurar drenajes, proteger equipos."
            ),
            AlertLevel.ALTA: (
                "[ALTA] ALERTA ALTA: Lluvia probable en corto plazo. "
                "Asegurar drenajes, alertar a operaciones, proteger equipos expuestos."
            ),
            AlertLevel.MEDIA: (
                "[MEDIA] ALERTA MEDIA: Condiciones favorables para desarrollo de lluvia. "
                "Monitorear de cerca, preparar protocolos de respuesta."
            ),
            AlertLevel.BAJA: (
                "[OK] BAJO: Sin senales fuertes de lluvia inminente. "
                "Continuar monitoreo rutinario."
            )
        }
        
        recommendation = base_recommendations[level]
        
        # Añadir detalles específicos según factores dominantes
        details = []
        
        if scores["rh_score"] > 0.6:
            details.append("Alta humedad relativa detectada (factor principal)")
        
        if scores["temp_delta_score"] > 0.6:
            details.append("Enfriamiento rapido detectado (favorece condensacion)")
        
        if scores["wind_score"] > 0.6:
            details.append("Viento calmo (favorece acumulacion de humedad)")
        
        if scores["model_score"] > 0.7:
            details.append(f"Modelo predice precipitacion significativa (score modelo: {int(scores['model_score']*100)}%)")
        
        if scores["rh_temp_interaction"] > 0.7:
            details.append("Combinacion RH alta + Temp baja (patron clasico pre-lluvia)")
        
        if details:
            recommendation += " Factores: " + "; ".join(details) + "."
        
        # Añadir score de confianza del diagnóstico general
        confidence = int(total_score * 100)
        recommendation += f" Confianza del diagnostico general: {confidence}%."
        
        return recommendation
    
    def get_dynamic_threshold(
        self,
        current: TimeSeriesPoint,
        previous: Optional[List[TimeSeriesPoint]] = None
    ) -> float:
        """
        Calcula umbral dinámico basado en condiciones actuales.
        
        Cuando las condiciones son propicias (RH alta, temp baja),
        el umbral es más bajo (más sensible).
        
        Args:
            current: Condiciones actuales
            previous: Condiciones previas
        
        Returns:
            Umbral dinámico en mm/hr
        """
        base_threshold = settings.precip_event_mmhr
        
        # Calcular scores
        scores = self.calculate_risk_score(current, previous, None)
        
        # Ajustar umbral según condiciones
        # Score alto → Umbral bajo (más sensible)
        total_score = (
            scores["rh_score"] * 0.3 +
            scores["temp_score"] * 0.2 +
            scores["wind_score"] * 0.1 +
            scores["temp_delta_score"] * 0.2 +
            scores["rh_temp_interaction"] * 0.2
        )
        
        # Ajuste: Score 0 → umbral x1.5, Score 1 → umbral x0.5
        adjustment = 1.5 - total_score
        dynamic_threshold = base_threshold * adjustment
        
        # Limitar a rango razonable
        dynamic_threshold = np.clip(dynamic_threshold, 0.3, 10.0)
        
        logger.info(
            f"Umbral dinámico: {dynamic_threshold:.2f} mm/hr "
            f"(base={base_threshold}, score={total_score:.2f})"
        )
        
        return float(dynamic_threshold)


def get_smart_diagnosis_service() -> SmartDiagnosisService:
    """Factory function para obtener servicio."""
    return SmartDiagnosisService()

