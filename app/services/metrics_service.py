"""Servicio de métricas."""

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from app.core.logging import get_logger
from app.services.model_service import get_model_service

logger = get_logger(__name__)


class MetricsService:
    """Servicio singleton de métricas."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.start_time = time.time()
        self.total_predictions = 0
        self.total_forecasts = 0
        self.total_diagnoses = 0
        self.prediction_latencies: List[float] = []
        self.forecast_latencies: List[float] = []
        self.total_errors = 0
        self.errors_by_type: Dict[str, int] = defaultdict(int)
        
        self._initialized = True
        logger.info("MetricsService inicializado")
    
    def record_prediction(self, latency_ms: float):
        """Registra una predicción."""
        self.total_predictions += 1
        self.prediction_latencies.append(latency_ms)
        
        # Mantener solo últimas 1000
        if len(self.prediction_latencies) > 1000:
            self.prediction_latencies = self.prediction_latencies[-1000:]
    
    def record_forecast(self, latency_ms: float):
        """Registra un pronóstico."""
        self.total_forecasts += 1
        self.forecast_latencies.append(latency_ms)
        
        if len(self.forecast_latencies) > 1000:
            self.forecast_latencies = self.forecast_latencies[-1000:]
    
    def record_diagnosis(self):
        """Registra un diagnóstico."""
        self.total_diagnoses += 1
    
    def record_error(self, error_type: str):
        """Registra un error."""
        self.total_errors += 1
        self.errors_by_type[error_type] += 1
    
    def get_uptime(self) -> float:
        """Retorna uptime en segundos."""
        return time.time() - self.start_time
    
    def _get_model_metrics(self) -> dict:
        """Obtiene métricas de ambos modelos (LSTM y XGBoost)."""
        import json
        from pathlib import Path
        
        # Inicializar métricas vacías
        model_metrics = {
            "lstm": {
                "mae": None,
                "rmse": None,
                "r2": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "mape": None
            },
            "xgboost": {
                "mae": None,
                "rmse": None,
                "r2": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "mape": None
            }
        }
        
        try:
            # Leer métricas del LSTM desde evaluation_results.json
            eval_results_path = Path("reports/evaluation_results.json")
            if eval_results_path.exists():
                with open(eval_results_path, 'r') as f:
                    eval_data = json.load(f)
                
                # Extraer métricas de test del LSTM
                test_metrics = eval_data.get("splits", {}).get("test", {})
                regression_metrics = test_metrics.get("regression", {})
                classification_metrics = test_metrics.get("classification", {})
                
                model_metrics["lstm"].update({
                    "mae": regression_metrics.get("MAE"),
                    "rmse": regression_metrics.get("RMSE"),
                    "r2": regression_metrics.get("R2"),
                    "mape": regression_metrics.get("MAPE"),
                    "precision": classification_metrics.get("precision"),
                    "recall": classification_metrics.get("recall"),
                    "f1_score": classification_metrics.get("f1_score")
                })
                
                logger.info("✓ Métricas LSTM cargadas desde evaluation_results.json")
            
            # Leer métricas del XGBoost desde xgboost_metadata.json
            xgb_metadata_path = Path("models/xgboost_metadata.json")
            if xgb_metadata_path.exists():
                with open(xgb_metadata_path, 'r') as f:
                    xgb_data = json.load(f)
                
                # Extraer métricas del XGBoost
                xgb_metrics = xgb_data.get("metrics", {})
                
                model_metrics["xgboost"].update({
                    "mae": xgb_metrics.get("mae"),
                    "rmse": xgb_metrics.get("rmse"),
                    "r2": xgb_metrics.get("r2"),
                    "mape": xgb_metrics.get("mape"),
                    "precision": xgb_metrics.get("precision"),
                    "recall": xgb_metrics.get("recall"),
                    "f1_score": xgb_metrics.get("f1_score")
                })
                
                logger.info("✓ Métricas XGBoost cargadas desde xgboost_metadata.json")
                
        except Exception as e:
            logger.warning(f"Error cargando métricas de modelos: {e}")
        
        return model_metrics
    
    def get_metrics(self) -> dict:
        """Retorna todas las métricas."""
        import numpy as np
        import json
        from pathlib import Path
        
        # Métricas del modelo (desde metadata)
        model_metrics = self._get_model_metrics()
        
        # Métricas online
        avg_pred_latency = (
            np.mean(self.prediction_latencies)
            if self.prediction_latencies else 0.0
        )
        avg_forecast_latency = (
            np.mean(self.forecast_latencies)
            if self.forecast_latencies else 0.0
        )
        
        p95_pred = (
            np.percentile(self.prediction_latencies, 95)
            if self.prediction_latencies else 0.0
        )
        p99_pred = (
            np.percentile(self.prediction_latencies, 99)
            if self.prediction_latencies else 0.0
        )
        
        # Tasa de error
        total_requests = self.total_predictions + self.total_forecasts + self.total_diagnoses
        error_rate = (self.total_errors / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "timestamp": datetime.now(timezone.utc),
            "model_metrics": model_metrics,
            "online_metrics": {
                "total_predictions": self.total_predictions,
                "total_forecasts": self.total_forecasts,
                "total_diagnoses": self.total_diagnoses,
                "avg_prediction_latency_ms": float(avg_pred_latency),
                "avg_forecast_latency_ms": float(avg_forecast_latency),
                "p95_prediction_latency_ms": float(p95_pred),
                "p99_prediction_latency_ms": float(p99_pred)
            },
            "error_metrics": {
                "total_errors": self.total_errors,
                "error_rate": float(error_rate),
                "errors_by_type": dict(self.errors_by_type)
            },
            "uptime_seconds": self.get_uptime()
        }


def get_metrics_service() -> MetricsService:
    """Factory function para obtener MetricsService singleton."""
    return MetricsService()

