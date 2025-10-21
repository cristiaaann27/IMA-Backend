"""Ingeniería de características modular para predicción de precipitación."""

import pandas as pd
import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod

from ..utils import Config, setup_logger


config = Config()
logger = setup_logger(
    "features.engineering",
    log_file=config.reports_dir / "features.log",
    level=config.log_level,
    format_type=config.log_format
)


class BaseFeatureCreator(ABC):
    """Clase base para creadores de features."""
    
    @abstractmethod
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features en el DataFrame."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Retorna nombres de features creados."""
        pass


class TemporalFeatureCreator(BaseFeatureCreator):
    """Creador de features temporales."""
    
    def __init__(self, include_cyclical: bool = True):
        """
        Inicializa el creador.
        
        Args:
            include_cyclical: Incluir features cíclicos (sin/cos)
        """
        self.include_cyclical = include_cyclical
        self._feature_names = []
    
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características temporales.
        
        Args:
            df: DataFrame con columna 'timestamp'
        
        Returns:
            DataFrame con features temporales
        """
        df = df.copy()
        self._feature_names = []
        
        # Features temporales básicos
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        self._feature_names.extend([
            "hour_of_day", "month", "day_of_week", 
            "day_of_year", "is_weekend"
        ])
        
        # Features cíclicos
        if self.include_cyclical:
            # Hora (24h)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
            
            # Mes (12 meses)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            
            # Día de la semana (7 días)
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            
            self._feature_names.extend([
                "hour_sin", "hour_cos", "month_sin", "month_cos",
                "dow_sin", "dow_cos"
            ])
        
        logger.info(f"✅ Creados {len(self._feature_names)} features temporales")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Retorna nombres de features creados."""
        return self._feature_names


class LagFeatureCreator(BaseFeatureCreator):
    """Creador de features de lag y rolling."""
    
    def __init__(
        self,
        target_col: str = "precip_mm_hr",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        include_deltas: bool = True
    ):
        """
        Inicializa el creador.
        
        Args:
            target_col: Columna objetivo
            lags: Lista de lags (ej: [1, 2, 3, 6, 12])
            rolling_windows: Ventanas móviles (ej: [3, 6, 12])
            include_deltas: Incluir features de cambio
        """
        self.target_col = target_col
        self.lags = lags or config.lags
        self.rolling_windows = rolling_windows or config.rolling_windows
        self.include_deltas = include_deltas
        self._feature_names = []
    
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de lag y rolling.
        
        Args:
            df: DataFrame con datos
        
        Returns:
            DataFrame con lag features
        """
        df = df.copy()
        self._feature_names = []
        
        if self.target_col not in df.columns:
            raise ValueError(f"Columna '{self.target_col}' no encontrada")
        
        # Lags
        for lag in self.lags:
            col_name = f"{self.target_col}_lag_{lag}"
            df[col_name] = df[self.target_col].shift(lag)
            self._feature_names.append(col_name)
        
        # Rolling statistics
        for window in self.rolling_windows:
            # Media móvil
            col_mean = f"{self.target_col}_rolling_mean_{window}"
            df[col_mean] = df[self.target_col].rolling(
                window=window, min_periods=1
            ).mean()
            self._feature_names.append(col_mean)
            
            # Desviación estándar móvil
            col_std = f"{self.target_col}_rolling_std_{window}"
            df[col_std] = df[self.target_col].rolling(
                window=window, min_periods=1
            ).std()
            self._feature_names.append(col_std)
            
            # Máximo y mínimo móvil
            col_max = f"{self.target_col}_rolling_max_{window}"
            df[col_max] = df[self.target_col].rolling(
                window=window, min_periods=1
            ).max()
            self._feature_names.append(col_max)
            
            col_min = f"{self.target_col}_rolling_min_{window}"
            df[col_min] = df[self.target_col].rolling(
                window=window, min_periods=1
            ).min()
            self._feature_names.append(col_min)
        
        # Deltas (cambios)
        if self.include_deltas:
            df[f"{self.target_col}_delta_1h"] = df[self.target_col].diff(1)
            df[f"{self.target_col}_delta_3h"] = df[self.target_col].diff(3)
            df[f"{self.target_col}_delta_6h"] = df[self.target_col].diff(6)
            
            self._feature_names.extend([
                f"{self.target_col}_delta_1h",
                f"{self.target_col}_delta_3h",
                f"{self.target_col}_delta_6h"
            ])
        
        logger.info(
            f"✅ Creados {len(self._feature_names)} lag/rolling features "
            f"({len(self.lags)} lags, {len(self.rolling_windows)} ventanas)"
        )
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Retorna nombres de features creados."""
        return self._feature_names


class InteractionFeatureCreator(BaseFeatureCreator):
    """Creador de features de interacción entre variables."""
    
    def __init__(self):
        """Inicializa el creador."""
        self._feature_names = []
    
    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de interacción.
        
        Args:
            df: DataFrame con variables
        
        Returns:
            DataFrame con features de interacción
        """
        df = df.copy()
        self._feature_names = []
        
        # Interacción RH * Temp
        if "rh_2m_pct" in df.columns and "temp_2m_c" in df.columns:
            df["rh_temp_interaction"] = df["rh_2m_pct"] * df["temp_2m_c"]
            self._feature_names.append("rh_temp_interaction")
            
            # Punto de rocío aproximado
            df["dewpoint_approx"] = df["temp_2m_c"] - ((100 - df["rh_2m_pct"]) / 5)
            self._feature_names.append("dewpoint_approx")
        
        # Índice de calma (viento bajo + humedad alta)
        if "wind_speed_2m_ms" in df.columns and "rh_2m_pct" in df.columns:
            df["calm_humid_index"] = (
                (df["rh_2m_pct"] / 100) * (1 / (df["wind_speed_2m_ms"] + 0.1))
            )
            self._feature_names.append("calm_humid_index")
        
        # Deltas de temperatura
        if "temp_2m_c" in df.columns:
            df["temp_2m_c_delta_2h"] = df["temp_2m_c"].diff(2)
            df["temp_2m_c_delta_6h"] = df["temp_2m_c"].diff(6)
            self._feature_names.extend([
                "temp_2m_c_delta_2h",
                "temp_2m_c_delta_6h"
            ])
        
        # Componentes de viento (si hay dirección)
        if "wind_speed_2m_ms" in df.columns and "wind_dir_2m_deg" in df.columns:
            # Convertir a radianes
            wind_rad = np.deg2rad(df["wind_dir_2m_deg"])
            
            # Componentes U (este-oeste) y V (norte-sur)
            df["wind_u_2m"] = -df["wind_speed_2m_ms"] * np.sin(wind_rad)
            df["wind_v_2m"] = -df["wind_speed_2m_ms"] * np.cos(wind_rad)
            
            self._feature_names.extend(["wind_u_2m", "wind_v_2m"])
        
        logger.info(f"✅ Creados {len(self._feature_names)} features de interacción")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Retorna nombres de features creados."""
        return self._feature_names


class FeaturePipeline:
    """Pipeline de ingeniería de características."""
    
    def __init__(
        self,
        creators: Optional[List[BaseFeatureCreator]] = None,
        drop_initial_nans: bool = True
    ):
        """
        Inicializa el pipeline.
        
        Args:
            creators: Lista de creadores de features
            drop_initial_nans: Eliminar filas iniciales con NaN en lags
        """
        self.creators = creators or self._default_creators()
        self.drop_initial_nans = drop_initial_nans
    
    def _default_creators(self) -> List[BaseFeatureCreator]:
        """Creadores por defecto."""
        return [
            TemporalFeatureCreator(include_cyclical=True),
            LagFeatureCreator(),
            InteractionFeatureCreator()
        ]
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todos los creadores de features.
        
        Args:
            df: DataFrame de entrada
        
        Returns:
            DataFrame con features creados
        """
        logger.info("="*60)
        logger.info("INGENIERÍA DE CARACTERÍSTICAS")
        logger.info("="*60)
        
        initial_rows = len(df)
        initial_cols = len(df.columns)
        
        # Aplicar cada creador
        for creator in self.creators:
            df = creator.create(df)
            logger.debug(f"  Aplicado: {creator.__class__.__name__}")
        
        # Eliminar filas iniciales con NaN
        if self.drop_initial_nans:
            df = self._drop_initial_nans(df)
        
        final_rows = len(df)
        final_cols = len(df.columns)
        
        logger.info(f"✅ Features creados:")
        logger.info(f"  • Registros: {initial_rows} → {final_rows}")
        logger.info(f"  • Columnas: {initial_cols} → {final_cols}")
        logger.info(f"  • Nuevas features: {final_cols - initial_cols}")
        
        logger.info("="*60)
        
        return df
    
    def _drop_initial_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas iniciales con NaN en lags."""
        # Encontrar el lag máximo
        max_lag = 0
        for creator in self.creators:
            if isinstance(creator, LagFeatureCreator):
                max_lag = max(max_lag, max(creator.lags))
        
        if max_lag > 0:
            df_clean = df.iloc[max_lag:].copy()
            dropped = len(df) - len(df_clean)
            logger.info(f"  🗑️ Eliminadas {dropped} filas iniciales con NaN")
            return df_clean
        
        return df
    
    def get_all_feature_names(self) -> List[str]:
        """Retorna todos los nombres de features creados."""
        all_names = []
        for creator in self.creators:
            all_names.extend(creator.get_feature_names())
        return all_names
