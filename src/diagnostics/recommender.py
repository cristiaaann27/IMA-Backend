"""Generador de recomendaciones basado en diagnósticos."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

from ..utils import Config, setup_logger, load_dataframe, ensure_dir
from .rules import DiagnosticEngine, AlertLevel, DiagnosticResult


config = Config()
logger = setup_logger(
    "diagnostics.recommender",
    log_file=config.reports_dir / "diagnostics.log",
    level=config.log_level,
    format_type=config.log_format
)


class Recommender:
    """Generador de recomendaciones operacionales basadas en niveles de alerta."""
    
    # Recomendaciones por nivel de alerta
    RECOMMENDATIONS: Dict[AlertLevel, Dict] = {
        AlertLevel.CRITICA: {
            "nivel": "CRÍTICO",
            "descripcion": "Condiciones altamente favorables para lluvia inminente con factores agravantes",
            "acciones": [
                "[!] ACCION INMEDIATA: Lluvia muy probable en corto plazo",
                "Activar protocolos de emergencia para lluvia",
                "Asegurar todos los drenajes y sistemas de evacuación",
                "Alertar a todo el personal operativo",
                "Proteger equipos expuestos y materiales sensibles",
                "Suspender operaciones al aire libre de ser posible",
                "Mantener comunicación constante con supervisores",
                "Preparar equipos de respuesta ante inundaciones"
            ],
            "prioridad": 4
        },
        AlertLevel.ALTA: {
            "nivel": "ALTO",
            "descripcion": "Lluvia probable en el corto plazo",
            "acciones": [
                "[ALTA] Lluvia probable inminente",
                "Asegurar drenajes y verificar sistemas de evacuación",
                "Alertar a operaciones y personal de campo",
                "Revisar y proteger equipos expuestos",
                "Preparar materiales de protección (lonas, cubiertas)",
                "Monitorear continuamente las condiciones",
                "Tener personal de respuesta en alerta",
                "Considerar posponer actividades críticas al exterior"
            ],
            "prioridad": 3
        },
        AlertLevel.MEDIA: {
            "nivel": "MEDIO",
            "descripcion": "Ambiente propenso a desarrollo de lluvia en corto/mediano plazo",
            "acciones": [
                "[MEDIA] Condiciones favorables para lluvia",
                "Monitorear de cerca las condiciones meteorológicas",
                "Preparar protocolos de respuesta ante lluvia",
                "Verificar estado de equipos de protección",
                "Informar al personal sobre posible lluvia",
                "Revisar planes de contingencia",
                "Mantener comunicación con áreas operativas",
                "Estar preparado para escalar a nivel de alerta superior"
            ],
            "prioridad": 2
        },
        AlertLevel.BAJA: {
            "nivel": "BAJO",
            "descripcion": "Sin señales fuertes de lluvia inminente",
            "acciones": [
                "[OK] Condiciones estables, sin senales fuertes de lluvia",
                "Continuar monitoreo rutinario",
                "Mantener protocolos estándar de operación",
                "Verificar periódicamente actualizaciones meteorológicas",
                "Asegurar que equipos de respuesta estén disponibles"
            ],
            "prioridad": 1
        }
    }
    
    @classmethod
    def get_recommendation(cls, alert_level: AlertLevel) -> Dict:
        """
        Obtiene recomendación para un nivel de alerta.
        
        Args:
            alert_level: Nivel de alerta
        
        Returns:
            Diccionario con recomendación
        """
        return cls.RECOMMENDATIONS[alert_level].copy()
    
    @classmethod
    def generate_report(
        cls,
        results: List[DiagnosticResult],
        summary: Dict,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Genera reporte completo de diagnóstico y recomendaciones.
        
        Args:
            results: Lista de DiagnosticResult
            summary: Resumen de diagnósticos
            output_path: Ruta de salida (opcional)
        
        Returns:
            Diccionario con reporte completo
        """
        logger.info("Generando reporte de diagnóstico y recomendaciones")
        
        if not results:
            logger.warning("No hay resultados para generar reporte")
            return cls._empty_report()
        
        # Obtener registros más críticos
        top_critical = cls._get_top_critical(results, top_n=10)
        
        # Recomendaciones por nivel
        recommendations_by_level = cls._build_recommendations_by_level(summary)
        
        # Determinar nivel de alerta general
        overall_alert = cls._determine_overall_alert(summary)
        overall_recommendation = cls.get_recommendation(overall_alert)
        
        # Construir reporte
        report = cls._build_report_dict(
            results, summary, overall_alert, 
            overall_recommendation, recommendations_by_level, top_critical
        )
        
        # Guardar reporte
        output_path = cls._save_report(report, output_path)
        
        # Log resumen
        cls._log_summary(summary, overall_alert)
        
        return report
    
    @classmethod
    def _empty_report(cls) -> Dict:
        """Retorna un reporte vacío."""
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_registros": 0,
                "periodo_analizado": {"inicio": None, "fin": None}
            },
            "resumen_alertas": {"baja": 0, "media": 0, "alta": 0, "critica": 0},
            "resumen_porcentajes": {"baja": 0, "media": 0, "alta": 0, "critica": 0},
            "precipitacion": {"media": 0, "max": 0, "eventos_lluvia": 0},
            "reglas_frecuentes": [],
            "nivel_alerta_general": {
                "nivel": "baja",
                "descripcion": "Sin datos para analizar",
                "acciones_recomendadas": []
            },
            "recomendaciones_por_nivel": {},
            "top_10_criticos": []
        }
    
    @classmethod
    def _get_top_critical(cls, results: List[DiagnosticResult], top_n: int = 10) -> List[DiagnosticResult]:
        """Obtiene los registros más críticos."""
        critical_results = [r for r in results if r.alert_level != AlertLevel.BAJA]
        critical_results.sort(key=lambda x: x.alert_level.severity, reverse=True)
        return critical_results[:top_n]
    
    @classmethod
    def _build_recommendations_by_level(cls, summary: Dict) -> Dict:
        """Construye recomendaciones por nivel de alerta."""
        recommendations = {}
        for level in AlertLevel:
            count = summary["alertas_por_nivel"][level.value]
            if count > 0:
                recommendations[level.value] = {
                    "count": count,
                    "percentage": summary["alertas_porcentaje"][level.value],
                    "recommendation": cls.get_recommendation(level)
                }
        return recommendations
    
    @classmethod
    def _determine_overall_alert(cls, summary: Dict, threshold_pct: float = 5.0) -> AlertLevel:
        """Determina el nivel de alerta general basado en porcentajes."""
        for level in [AlertLevel.CRITICA, AlertLevel.ALTA, AlertLevel.MEDIA]:
            if summary["alertas_porcentaje"][level.value] > threshold_pct:
                return level
        return AlertLevel.BAJA
    
    @classmethod
    def _build_report_dict(
        cls,
        results: List[DiagnosticResult],
        summary: Dict,
        overall_alert: AlertLevel,
        overall_recommendation: Dict,
        recommendations_by_level: Dict,
        top_critical: List[DiagnosticResult]
    ) -> Dict:
        """Construye el diccionario del reporte."""
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_registros": summary["total_registros"],
                "periodo_analizado": {
                    "inicio": results[0].timestamp,
                    "fin": results[-1].timestamp
                }
            },
            "resumen_alertas": summary["alertas_por_nivel"],
            "resumen_porcentajes": summary["alertas_porcentaje"],
            "precipitacion": summary["precipitacion_predicha"],
            "reglas_frecuentes": summary["reglas_mas_frecuentes"],
            "nivel_alerta_general": {
                "nivel": overall_alert.value,
                "descripcion": overall_recommendation["descripcion"],
                "acciones_recomendadas": overall_recommendation["acciones"]
            },
            "recomendaciones_por_nivel": recommendations_by_level,
            "top_10_criticos": [
                {
                    "timestamp": r.timestamp,
                    "nivel_alerta": r.alert_level.value,
                    "precip_pred": r.precip_pred,
                    "reglas_activadas": r.triggered_rules,
                    "metricas": r.metrics
                }
                for r in top_critical
            ]
        }
    
    @classmethod
    def _save_report(cls, report: Dict, output_path: Optional[Path] = None) -> Path:
        """Guarda el reporte en formato JSON."""
        if output_path is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = config.reports_dir / f"diagnosis_{timestamp_str}.json"
        
        ensure_dir(output_path.parent)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reporte guardado en {output_path}")
        return output_path
    
    @classmethod
    def _log_summary(cls, summary: Dict, overall_alert: AlertLevel):
        """Registra resumen en logs."""
        logger.info("="*60)
        logger.info("RESUMEN DE DIAGNÓSTICO")
        logger.info("="*60)
        logger.info(f"Nivel de Alerta General: {overall_alert.value.upper()}")
        logger.info(f"Total de registros: {summary['total_registros']}")
        logger.info("Distribución de alertas:")
        for level, count in summary["alertas_por_nivel"].items():
            pct = summary["alertas_porcentaje"][level]
            logger.info(f"  {level.upper()}: {count} ({pct:.1f}%)")
        
        logger.info(f"Precipitación media predicha: {summary['precipitacion_predicha']['media']:.2f} mm/hr")
        logger.info(f"Eventos de lluvia: {summary['precipitacion_predicha']['eventos_lluvia']}")
        logger.info("="*60)


def diagnose(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Ejecuta diagnóstico completo sobre predicciones.
    
    Args:
        input_path: Ruta de archivo de predicciones (si None, usa el más reciente)
        output_path: Ruta de salida para reporte (si None, genera automáticamente)
    
    Returns:
        Diccionario con reporte completo
    """
    logger.info("="*60)
    logger.info("INICIANDO DIAGNÓSTICO")
    logger.info("="*60)
    
    # Cargar y preparar datos
    df = _load_prediction_data(input_path)
    df = _merge_with_curated_data(df)
    
    # Ejecutar diagnóstico
    engine = DiagnosticEngine()
    results = engine.evaluate_dataframe(df)
    logger.info(f"Evaluados {len(results)} registros")
    
    # Generar resumen y reporte
    summary = engine.summarize_results(results)
    report = Recommender.generate_report(results, summary, output_path)
    
    logger.info("="*60)
    logger.info("DIAGNÓSTICO COMPLETADO")
    logger.info("="*60)
    
    return report


def _load_prediction_data(input_path: Optional[Path] = None) -> pd.DataFrame:
    """Carga datos de predicciones."""
    if input_path is None:
        pred_files = sorted(config.predictions_dir.glob("pred_*.parquet"))
        if not pred_files:
            logger.error("No se encontraron archivos de predicciones")
            raise FileNotFoundError("No hay predicciones disponibles")
        input_path = pred_files[-1]
    
    logger.info(f"Cargando predicciones desde {input_path}")
    return load_dataframe(input_path)


def _merge_with_curated_data(df: pd.DataFrame) -> pd.DataFrame:
    """Combina predicciones con datos curados para obtener métricas meteorológicas."""
    curated_path = config.curated_data_dir / "curated_latest.parquet"
    
    if not curated_path.exists():
        logger.warning(f"No se encontró {curated_path}, usando solo datos de predicciones")
        return df
    
    df_curated = load_dataframe(curated_path)
    
    if "timestamp" not in df.columns or "timestamp" not in df_curated.columns:
        logger.warning("No se puede hacer merge: falta columna timestamp")
        return df
    
    df_merged = pd.merge(df, df_curated, on="timestamp", how="left", suffixes=("", "_curated"))
    logger.info(f"Datos combinados con curated: {len(df_merged)} registros")
    
    return df_merged


