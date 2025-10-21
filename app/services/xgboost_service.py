"""Servicio para predicción con XGBoost."""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xgboost as xgb
import pickle

from app.core.config import get_settings
from app.core.exceptions import ModelNotLoadedException, PredictionException

logger = logging.getLogger(__name__)
settings = get_settings()


class XGBoostService:
    def __init__(self):
        self.model: xgb.XGBRegressor = None
        self.scaler = None
        self.feature_names: List[str] = None
        self.loaded = False
    
    def load_model(self):
        try:
            logger.info("Cargando modelo XGBoost...")
            
            # Cargar modelo
            model_path = settings.project_root / 'models' / 'xgboost_latest.json'
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo XGBoost no encontrado en {model_path}")
            
            self.model = xgb.XGBRegressor()
            self.model.load_model(str(model_path))
            
            # Cargar scaler (mismo que LSTM)
            scaler_path = settings.scaler_full_path
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Cargar feature names
            import json
            metadata_path = settings.project_root / 'models' / 'xgboost_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
            
            self.loaded = True
            logger.info("✓ Modelo XGBoost cargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelo XGBoost: {e}")
            raise ModelNotLoadedException(f"No se pudo cargar modelo XGBoost: {e}")
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:

        if not self.loaded:
            raise ModelNotLoadedException("Modelo XGBoost no está cargado")
        
        try:
            # Escalar features si el scaler está disponible
            if self.scaler is not None:
                # El scaler puede tener múltiples scalers (X, y)
                if isinstance(self.scaler, dict) and "scaler_X" in self.scaler:
                    # Usar el scaler de features del LSTM
                    features_scaled = self.scaler["scaler_X"].transform(features.reshape(1, -1))
                else:
                    # Usar el scaler directamente
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                # Sin escalamiento (comportamiento original)
                features_scaled = features.reshape(1, -1)
            
            # Predicción
            precip_pred = float(self.model.predict(features_scaled)[0])
            
            # Des-escalar la predicción si es necesario
            if self.scaler is not None and isinstance(self.scaler, dict) and "scaler_y" in self.scaler:
                # Des-escalar usando el scaler de salida del LSTM
                precip_pred = self.scaler["scaler_y"].inverse_transform([[precip_pred]])[0, 0]
            
            # Asegurar que no sea negativo
            precip_pred = max(0.0, precip_pred)
            
            # Probabilidad de evento usando sigmoide (consistente con LSTM)
            rain_prob = self._calculate_rain_probability(precip_pred)
            
            return precip_pred, rain_prob
            
        except Exception as e:
            logger.error(f"Error en predicción XGBoost: {e}")
            raise PredictionException(f"Error en predicción: {e}")
    
    def is_loaded(self) -> bool:
        return self.loaded
    
    def _calculate_rain_probability(self, prediction_mm_hr: float) -> float:

        threshold = settings.precip_event_mmhr
        
        # Sigmoide: prob = 1 / (1 + exp(-k * (x - threshold)))
        k = 2.0  # Factor de escala
        prob = 1.0 / (1.0 + np.exp(-k * (prediction_mm_hr - threshold)))
        
        return float(np.clip(prob, 0.0, 1.0))


# Instancia singleton
_xgboost_service = None


def get_xgboost_service() -> XGBoostService:
    global _xgboost_service
    if _xgboost_service is None:
        _xgboost_service = XGBoostService()
    return _xgboost_service
