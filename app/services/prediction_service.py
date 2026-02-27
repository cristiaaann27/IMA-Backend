"""Servicio de predicción."""

import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch

from app.core.config import get_settings
from app.core.exceptions import PredictionException, ValidationException
from app.core.logging import get_logger
from app.schemas.prediction import TimeSeriesPoint
from app.services.model_service import get_model_service
from app.services.feature_utils import prepare_features, calculate_rain_probability

logger = get_logger(__name__)
settings = get_settings()


class PredictionService:
    """Servicio de predicción con LSTM."""
    
    def __init__(self):
        self.model_service = get_model_service()
    
    def predict(self, lookback_data: List[TimeSeriesPoint]) -> Tuple[float, float]:
        """
        Realiza predicción t+1.
        
        Args:
            lookback_data: Ventana de observación
        
        Returns:
            Tuple (predicción_mm_hr, probabilidad_evento)
        
        Raises:
            PredictionException: Si falla la predicción
        """
        start_time = time.time()
        
        try:
            # Preparar features
            X = self._prepare_features(lookback_data)
            
            # Escalar
            scaler_X = self.model_service.get_scalers()["scaler_X"]
            scaler_y = self.model_service.get_scalers()["scaler_y"]
            
            X_scaled = scaler_X.transform(X)
            
            # Crear secuencia (reshape para LSTM)
            lookback = self.model_service.get_lookback()
            
            if len(X_scaled) < lookback:
                # Pad si es necesario
                pad_size = lookback - len(X_scaled)
                X_padded = np.vstack([
                    np.zeros((pad_size, X_scaled.shape[1])),
                    X_scaled
                ])
            else:
                X_padded = X_scaled[-lookback:]
            
            X_seq = torch.FloatTensor(X_padded).unsqueeze(0)  # (1, lookback, features)
            
            # Predecir
            model = self.model_service.get_model()
            model.eval()
            
            with torch.no_grad():
                y_pred_scaled = model(X_seq).numpy()  # (1, horizon)
            
            # Des-escalar: el modelo puede producir horizon > 1 pasos
            # Para /predict (t+1) tomamos solo el primer paso
            if y_pred_scaled.ndim == 2 and y_pred_scaled.shape[1] > 1:
                # Multi-step: tomar primer paso para t+1
                first_step_scaled = y_pred_scaled[:, 0:1]
                y_pred = scaler_y.inverse_transform(first_step_scaled)[0, 0]
            else:
                y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
            y_pred = max(0.0, y_pred)  # No negativos
            
            # Calcular probabilidad de evento
            rain_prob = self._calculate_rain_probability(y_pred)
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Predicción: {y_pred:.2f} mm/hr (prob: {rain_prob:.2f}, latency: {elapsed:.1f}ms)")
            
            return y_pred, rain_prob
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise PredictionException(str(e))
    
    def predict_multistep(self, lookback_data: List[TimeSeriesPoint]) -> list:
        """
        Predicción multi-paso nativa (hasta horizon pasos).
        
        Retorna lista de tuplas (predicción_mm_hr, probabilidad_evento)
        para cada paso del horizonte.
        """
        try:
            X = self._prepare_features(lookback_data)
            
            scaler_X = self.model_service.get_scalers()["scaler_X"]
            scaler_y = self.model_service.get_scalers()["scaler_y"]
            
            X_scaled = scaler_X.transform(X)
            lookback = self.model_service.get_lookback()
            
            if len(X_scaled) < lookback:
                pad_size = lookback - len(X_scaled)
                X_padded = np.vstack([
                    np.zeros((pad_size, X_scaled.shape[1])),
                    X_scaled
                ])
            else:
                X_padded = X_scaled[-lookback:]
            
            X_seq = torch.FloatTensor(X_padded).unsqueeze(0)
            
            model = self.model_service.get_model()
            model.eval()
            
            with torch.no_grad():
                y_pred_scaled = model(X_seq).numpy()  # (1, horizon)
            
            # Des-escalar todos los pasos
            results = []
            n_steps = y_pred_scaled.shape[1] if y_pred_scaled.ndim == 2 else 1
            
            for step in range(n_steps):
                if y_pred_scaled.ndim == 2:
                    step_scaled = y_pred_scaled[:, step:step+1]
                else:
                    step_scaled = y_pred_scaled
                y_val = scaler_y.inverse_transform(step_scaled)[0, 0]
                y_val = max(0.0, y_val)
                rain_prob = self._calculate_rain_probability(y_val)
                results.append((y_val, rain_prob))
            
            return results
            
        except Exception as e:
            logger.error(f"Error en predicción multi-step: {e}")
            raise PredictionException(str(e))
    
    def _prepare_features(self, lookback_data: List[TimeSeriesPoint]) -> np.ndarray:
        """Delega a feature_utils.prepare_features (módulo compartido)."""
        return prepare_features(lookback_data)
    
    def _calculate_rain_probability(self, prediction_mm_hr: float) -> float:
        """Delega a feature_utils.calculate_rain_probability (módulo compartido)."""
        return calculate_rain_probability(prediction_mm_hr)

