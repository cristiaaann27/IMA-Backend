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
                y_pred_scaled = model(X_seq).numpy()
            
            # Des-escalar
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
    
    def _prepare_features(self, lookback_data: List[TimeSeriesPoint]) -> np.ndarray:
        """
        Prepara features a partir de datos de entrada.
        
        Genera las 33 features esperadas por el modelo LSTM.
        """
        if not lookback_data:
            return np.array([])
        
        n_samples = len(lookback_data)
        n_expected_features = 33  # Del modelo entrenado según metadata
        
        # Crear array expandido
        expanded = np.zeros((n_samples, n_expected_features))
        
        for i, point in enumerate(lookback_data):
            # Features base (0-1)
            expanded[i, 0] = point.rh_2m_pct
            expanded[i, 1] = point.temp_2m_c
            
            # Features temporales desde timestamp real (2-9)
            ts = point.timestamp
            hour = ts.hour
            month = ts.month
            day_of_week = ts.weekday()
            day_of_year = ts.timetuple().tm_yday
            
            expanded[i, 2] = hour
            expanded[i, 3] = month
            expanded[i, 4] = day_of_week
            expanded[i, 5] = day_of_year
            
            # Codificación cíclica de hora y mes (6-9)
            expanded[i, 6] = np.sin(2 * np.pi * hour / 24)
            expanded[i, 7] = np.cos(2 * np.pi * hour / 24)
            expanded[i, 8] = np.sin(2 * np.pi * month / 12)
            expanded[i, 9] = np.cos(2 * np.pi * month / 12)
            
            # Lags de precipitación (10-14) - asumiendo 0 si no hay datos históricos
            expanded[i, 10] = 0  # precip_mm_hr_lag_1
            expanded[i, 11] = 0  # precip_mm_hr_lag_2
            expanded[i, 12] = 0  # precip_mm_hr_lag_3
            expanded[i, 13] = 0  # precip_mm_hr_lag_6
            expanded[i, 14] = 0  # precip_mm_hr_lag_12
            
            # Rolling statistics (15-18) - usando RH como aproximación
            expanded[i, 15] = point.rh_2m_pct  # precip_mm_hr_rolling_mean_3
            expanded[i, 16] = point.rh_2m_pct * 0.1  # precip_mm_hr_rolling_std_3
            expanded[i, 17] = point.rh_2m_pct  # precip_mm_hr_rolling_mean_6
            expanded[i, 18] = point.rh_2m_pct * 0.1  # precip_mm_hr_rolling_std_6
            
            # Deltas de precipitación (19-20) - asumiendo 0
            expanded[i, 19] = 0  # precip_mm_hr_delta_1h
            expanded[i, 20] = 0  # precip_mm_hr_delta_3h
            
            # Feature de interacción RH-Temp (21)
            expanded[i, 21] = point.rh_2m_pct * point.temp_2m_c
            
            # Deltas de temperatura (22-23) - calculados si hay datos previos
            if i >= 2:
                expanded[i, 22] = point.temp_2m_c - lookback_data[i-2].temp_2m_c  # delta_2h
            else:
                expanded[i, 22] = 0
            
            if i >= 6:
                expanded[i, 23] = point.temp_2m_c - lookback_data[i-6].temp_2m_c  # delta_6h
            else:
                expanded[i, 23] = 0
            
            # Features adicionales (24-32) - wind_speed, wind_dir y derivados
            expanded[i, 24] = point.wind_speed_2m_ms
            expanded[i, 25] = point.wind_dir_2m_deg
            
            # Wind components (26-27)
            wind_rad = np.deg2rad(point.wind_dir_2m_deg)
            expanded[i, 26] = point.wind_speed_2m_ms * np.sin(wind_rad)  # u_component
            expanded[i, 27] = point.wind_speed_2m_ms * np.cos(wind_rad)  # v_component
            
            # Deltas de viento (28-29)
            if i >= 1:
                expanded[i, 28] = point.wind_speed_2m_ms - lookback_data[i-1].wind_speed_2m_ms
            else:
                expanded[i, 28] = 0
            
            if i >= 3:
                expanded[i, 29] = point.wind_speed_2m_ms - lookback_data[i-3].wind_speed_2m_ms
            else:
                expanded[i, 29] = 0
            
            # Deltas de RH (30-31)
            if i >= 1:
                expanded[i, 30] = point.rh_2m_pct - lookback_data[i-1].rh_2m_pct
            else:
                expanded[i, 30] = 0
            
            if i >= 3:
                expanded[i, 31] = point.rh_2m_pct - lookback_data[i-3].rh_2m_pct
            else:
                expanded[i, 31] = 0
            
            # Interacción adicional wind-temp (32)
            expanded[i, 32] = point.wind_speed_2m_ms * point.temp_2m_c
        
        return expanded
    
    def _calculate_rain_probability(self, prediction_mm_hr: float) -> float:
        """
        Calcula probabilidad de evento de lluvia.
        
        Usa función sigmoide escalada por umbral.
        """
        threshold = settings.precip_event_mmhr
        
        # Sigmoide: prob = 1 / (1 + exp(-k * (x - threshold)))
        k = 2.0  # Factor de escala
        prob = 1.0 / (1.0 + np.exp(-k * (prediction_mm_hr - threshold)))
        
        return float(np.clip(prob, 0.0, 1.0))

