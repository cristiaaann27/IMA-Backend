"""Motor de reglas de diagnóstico meteorológico."""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from ..utils import Config


config = Config()


class AlertLevel(Enum):
    """Niveles de alerta ordenados por severidad."""
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"
    CRITICA = "critica"
    
    @property
    def severity(self) -> int:
        """Retorna el nivel de severidad numérico."""
        return {"baja": 0, "media": 1, "alta": 2, "critica": 3}[self.value]


@dataclass
class DiagnosticResult:
    """Resultado de diagnóstico para un registro."""
    timestamp: str
    alert_level: AlertLevel
    triggered_rules: List[str] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    precip_pred: float = 0.0
    precip_real: Optional[float] = None
    
    def __post_init__(self):
        """Validación post-inicialización."""
        if self.precip_pred < 0:
            self.precip_pred = 0.0


class DiagnosticEngine:
    """
    Motor de diagnóstico basado en reglas meteorológicas.
    
    Reglas basadas en análisis:
    1. RH alta (>=90%) + Temp descendente (Δ <= -0.5°C en 2h) → Alerta Alta
    2. RH media-alta (>=85%) + Temp descendente → Alerta Media
    3. Viento calmo (<=1.0 m/s) + RH alta → Refuerzo de alerta (+1 nivel)
    4. Predicción de precipitación > umbral → Factor adicional
    
    Correlaciones de referencia (documentadas):
    - precip vs rh: +0.32
    - precip vs temp: -0.21
    - rh vs temp: -0.87
    """
    
    def __init__(
        self,
        rh_high: Optional[float] = None,
        rh_medium: Optional[float] = None,
        temp_drop_2h: Optional[float] = None,
        wind_calm: Optional[float] = None,
        precip_threshold: Optional[float] = None
    ):
        """
        Inicializa el motor de diagnóstico.
        
        Args:
            rh_high: Umbral de RH alta (%). Default: config.rh_high
            rh_medium: Umbral de RH media-alta (%). Default: config.rh_medium
            temp_drop_2h: Umbral de descenso de temperatura en 2h (°C). Default: config.temp_drop_2h
            wind_calm: Umbral de viento calmo (m/s). Default: config.wind_calm_ms
            precip_threshold: Umbral de evento de precipitación (mm/hr). Default: config.precip_event_mmhr
        """
        self.rh_high = rh_high if rh_high is not None else config.rh_high
        self.rh_medium = rh_medium if rh_medium is not None else config.rh_medium
        self.temp_drop_2h = temp_drop_2h if temp_drop_2h is not None else config.temp_drop_2h
        self.wind_calm = wind_calm if wind_calm is not None else config.wind_calm_ms
        self.precip_threshold = precip_threshold if precip_threshold is not None else config.precip_event_mmhr
    
    def evaluate_single(self, row: pd.Series) -> DiagnosticResult:
        """
        Evalúa un registro individual aplicando reglas de diagnóstico.
        
        Args:
            row: Serie con datos meteorológicos y predicciones
        
        Returns:
            DiagnosticResult con nivel de alerta y reglas activadas
        """
        triggered_rules = []
        alert_level = AlertLevel.BAJA
        
        # Extraer y validar métricas
        metrics = self._extract_metrics(row)
        timestamp = self._format_timestamp(row.get("timestamp", "unknown"))
        precip_pred = row.get("precip_pred_mm_hr", 0.0)
        precip_real = row.get("precip_real_mm_hr", None)
        
        # Aplicar reglas de diagnóstico
        alert_level = self._apply_rules(metrics, precip_pred, triggered_rules)
        
        return DiagnosticResult(
            timestamp=timestamp,
            alert_level=alert_level,
            triggered_rules=triggered_rules,
            metrics=metrics,
            precip_pred=float(precip_pred),
            precip_real=float(precip_real) if precip_real is not None and not pd.isna(precip_real) else None
        )
    
    def _extract_metrics(self, row: pd.Series) -> Dict[str, Optional[float]]:
        """Extrae y valida métricas meteorológicas."""
        def safe_float(value) -> Optional[float]:
            return float(value) if not pd.isna(value) else None
        
        return {
            "rh_2m_pct": safe_float(row.get("rh_2m_pct", np.nan)),
            "temp_2m_c": safe_float(row.get("temp_2m_c", np.nan)),
            "temp_delta_2h": safe_float(row.get("temp_2m_c_delta_2h", np.nan)),
            "wind_speed_2m_ms": safe_float(row.get("wind_speed_2m_ms", np.nan)),
            "precip_pred_mm_hr": safe_float(row.get("precip_pred_mm_hr", 0.0))
        }
    
    def _format_timestamp(self, timestamp) -> str:
        """Formatea timestamp a string ISO."""
        if hasattr(timestamp, "isoformat"):
            return timestamp.isoformat()
        return str(timestamp)
    
    def _apply_rules(self, metrics: Dict, precip_pred: float, triggered_rules: List[str]) -> AlertLevel:
        """Aplica reglas de diagnóstico y retorna nivel de alerta."""
        alert_level = AlertLevel.BAJA
        
        rh = metrics.get("rh_2m_pct")
        temp_delta_2h = metrics.get("temp_delta_2h")
        wind_speed = metrics.get("wind_speed_2m_ms")
        
        # Regla 1: RH alta + Temp descendente → Alerta Alta
        if rh is not None and temp_delta_2h is not None:
            if rh >= self.rh_high and temp_delta_2h <= self.temp_drop_2h:
                alert_level = AlertLevel.ALTA
                triggered_rules.append(
                    f"RH_HIGH_TEMP_DROP: RH={rh:.1f}% >= {self.rh_high}%, "
                    f"ΔTemp_2h={temp_delta_2h:.2f}°C <= {self.temp_drop_2h}°C"
                )
        
        # Regla 2: RH media-alta + Temp descendente → Alerta Media
        if alert_level == AlertLevel.BAJA and rh is not None and temp_delta_2h is not None:
            if rh >= self.rh_medium and temp_delta_2h <= self.temp_drop_2h:
                alert_level = AlertLevel.MEDIA
                triggered_rules.append(
                    f"RH_MEDIUM_TEMP_DROP: RH={rh:.1f}% >= {self.rh_medium}%, "
                    f"ΔTemp_2h={temp_delta_2h:.2f}°C <= {self.temp_drop_2h}°C"
                )
        
        # Regla 3: Viento calmo + RH alta → Refuerzo de alerta
        if wind_speed is not None and rh is not None:
            if wind_speed <= self.wind_calm and rh >= self.rh_medium:
                triggered_rules.append(
                    f"WIND_CALM_RH_HIGH: Viento={wind_speed:.2f} m/s <= {self.wind_calm}, "
                    f"RH={rh:.1f}% >= {self.rh_medium}%"
                )
                alert_level = self._escalate_alert(alert_level)
        
        # Regla 4: Predicción alta de precipitación
        if precip_pred >= self.precip_threshold * 2:
            triggered_rules.append(
                f"HIGH_PRECIP_PRED: Predicción={precip_pred:.2f} mm/hr >= "
                f"{self.precip_threshold * 2:.2f} mm/hr"
            )
            if alert_level.severity < AlertLevel.ALTA.severity:
                alert_level = self._escalate_alert(alert_level)
        
        return alert_level
    
    def _escalate_alert(self, current_level: AlertLevel) -> AlertLevel:
        """Escala el nivel de alerta al siguiente nivel."""
        escalation = {
            AlertLevel.BAJA: AlertLevel.MEDIA,
            AlertLevel.MEDIA: AlertLevel.ALTA,
            AlertLevel.ALTA: AlertLevel.CRITICA,
            AlertLevel.CRITICA: AlertLevel.CRITICA
        }
        return escalation[current_level]
    
    def evaluate_dataframe(self, df: pd.DataFrame) -> List[DiagnosticResult]:
        """
        Evalúa un DataFrame completo.
        
        Args:
            df: DataFrame con datos y predicciones
        
        Returns:
            Lista de DiagnosticResult
        """
        results = []
        
        for idx, row in df.iterrows():
            result = self.evaluate_single(row)
            results.append(result)
        
        return results
    
    def summarize_results(self, results: List[DiagnosticResult]) -> Dict:
        """
        Resume los resultados de diagnóstico.
        
        Args:
            results: Lista de DiagnosticResult
        
        Returns:
            Diccionario con resumen estadístico
        """
        if not results:
            return self._empty_summary()
        
        total = len(results)
        alert_counts = self._count_alerts(results)
        precip_stats = self._calculate_precip_stats(results)
        top_rules = self._get_top_rules(results)
        
        return {
            "total_registros": total,
            "alertas_por_nivel": {
                "baja": alert_counts[AlertLevel.BAJA],
                "media": alert_counts[AlertLevel.MEDIA],
                "alta": alert_counts[AlertLevel.ALTA],
                "critica": alert_counts[AlertLevel.CRITICA]
            },
            "alertas_porcentaje": {
                "baja": (alert_counts[AlertLevel.BAJA] / total * 100),
                "media": (alert_counts[AlertLevel.MEDIA] / total * 100),
                "alta": (alert_counts[AlertLevel.ALTA] / total * 100),
                "critica": (alert_counts[AlertLevel.CRITICA] / total * 100)
            },
            "precipitacion_predicha": precip_stats,
            "reglas_mas_frecuentes": top_rules
        }
    
    def _empty_summary(self) -> Dict:
        """Retorna resumen vacío."""
        return {
            "total_registros": 0,
            "alertas_por_nivel": {"baja": 0, "media": 0, "alta": 0, "critica": 0},
            "alertas_porcentaje": {"baja": 0, "media": 0, "alta": 0, "critica": 0},
            "precipitacion_predicha": {"media": 0, "max": 0, "eventos_lluvia": 0},
            "reglas_mas_frecuentes": []
        }
    
    def _count_alerts(self, results: List[DiagnosticResult]) -> Dict[AlertLevel, int]:
        """Cuenta alertas por nivel."""
        counts = {level: 0 for level in AlertLevel}
        for result in results:
            counts[result.alert_level] += 1
        return counts
    
    def _calculate_precip_stats(self, results: List[DiagnosticResult]) -> Dict:
        """Calcula estadísticas de precipitación."""
        precip_preds = [r.precip_pred for r in results]
        return {
            "media": float(np.mean(precip_preds)),
            "max": float(np.max(precip_preds)),
            "eventos_lluvia": sum(1 for p in precip_preds if p >= self.precip_threshold)
        }
    
    def _get_top_rules(self, results: List[DiagnosticResult], top_n: int = 5) -> List:
        """Obtiene las reglas más frecuentemente activadas."""
        rule_counts = {}
        for result in results:
            for rule in result.triggered_rules:
                rule_name = rule.split(":")[0]
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        return sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]


