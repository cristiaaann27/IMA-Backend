"""
Funciones compartidas de feature engineering y probabilidad de lluvia.

Este módulo centraliza la lógica que antes estaba duplicada en:
- app/services/prediction_service.py (_prepare_features, _calculate_rain_probability)
- app/routers/api_v1.py (prepare_xgboost_features)
- app/services/xgboost_service.py (_calculate_rain_probability)
"""

from typing import List

import numpy as np

from app.core.config import get_settings
from app.schemas.prediction import TimeSeriesPoint

settings = get_settings()


def prepare_features(lookback_data: List[TimeSeriesPoint]) -> np.ndarray:
    """
    Prepara las 33 features esperadas por los modelos a partir de TimeSeriesPoint.

    Uso compartido entre LSTM y XGBoost en inferencia.

    Args:
        lookback_data: Lista de puntos de serie temporal.

    Returns:
        Array (n_samples, 33) con las features generadas.
    """
    if not lookback_data:
        return np.array([])

    n_samples = len(lookback_data)
    n_expected_features = 33

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

        # Lags de precipitación (10-14)
        expanded[i, 10] = 0  # precip_mm_hr_lag_1
        expanded[i, 11] = 0  # precip_mm_hr_lag_2
        expanded[i, 12] = 0  # precip_mm_hr_lag_3
        expanded[i, 13] = 0  # precip_mm_hr_lag_6
        expanded[i, 14] = 0  # precip_mm_hr_lag_12

        # Rolling statistics (15-18)
        expanded[i, 15] = point.rh_2m_pct       # precip_mm_hr_rolling_mean_3
        expanded[i, 16] = point.rh_2m_pct * 0.1  # precip_mm_hr_rolling_std_3
        expanded[i, 17] = point.rh_2m_pct       # precip_mm_hr_rolling_mean_6
        expanded[i, 18] = point.rh_2m_pct * 0.1  # precip_mm_hr_rolling_std_6

        # Deltas de precipitación (19-20)
        expanded[i, 19] = 0  # precip_mm_hr_delta_1h
        expanded[i, 20] = 0  # precip_mm_hr_delta_3h

        # Feature de interacción RH-Temp (21)
        expanded[i, 21] = point.rh_2m_pct * point.temp_2m_c

        # Deltas de temperatura (22-23)
        if i >= 2:
            expanded[i, 22] = point.temp_2m_c - lookback_data[i - 2].temp_2m_c
        else:
            expanded[i, 22] = 0

        if i >= 6:
            expanded[i, 23] = point.temp_2m_c - lookback_data[i - 6].temp_2m_c
        else:
            expanded[i, 23] = 0

        # Features adicionales (24-32) - wind_speed, wind_dir y derivados
        expanded[i, 24] = point.wind_speed_2m_ms
        expanded[i, 25] = point.wind_dir_2m_deg

        # Wind components (26-27)
        wind_rad = np.deg2rad(point.wind_dir_2m_deg)
        expanded[i, 26] = point.wind_speed_2m_ms * np.sin(wind_rad)
        expanded[i, 27] = point.wind_speed_2m_ms * np.cos(wind_rad)

        # Deltas de viento (28-29)
        if i >= 1:
            expanded[i, 28] = point.wind_speed_2m_ms - lookback_data[i - 1].wind_speed_2m_ms
        else:
            expanded[i, 28] = 0

        if i >= 3:
            expanded[i, 29] = point.wind_speed_2m_ms - lookback_data[i - 3].wind_speed_2m_ms
        else:
            expanded[i, 29] = 0

        # Deltas de RH (30-31)
        if i >= 1:
            expanded[i, 30] = point.rh_2m_pct - lookback_data[i - 1].rh_2m_pct
        else:
            expanded[i, 30] = 0

        if i >= 3:
            expanded[i, 31] = point.rh_2m_pct - lookback_data[i - 3].rh_2m_pct
        else:
            expanded[i, 31] = 0

        # Interacción adicional wind-temp (32)
        expanded[i, 32] = point.wind_speed_2m_ms * point.temp_2m_c

    return expanded


def calculate_rain_probability(prediction_mm_hr: float) -> float:
    """
    Calcula probabilidad de evento de lluvia mediante sigmoide escalada.

    Args:
        prediction_mm_hr: Predicción de precipitación en mm/hr.

    Returns:
        Probabilidad en [0, 1].
    """
    threshold = settings.precip_event_mmhr
    k = 2.0
    prob = 1.0 / (1.0 + np.exp(-k * (prediction_mm_hr - threshold)))
    return float(np.clip(prob, 0.0, 1.0))
