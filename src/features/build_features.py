"""Construcción de features para predicción de precipitación.

Este módulo mantiene retrocompatibilidad con el código antiguo.
Para nuevos desarrollos, usar feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

from ..utils import Config, setup_logger, load_dataframe, save_dataframe
from .feature_engineering import (
    FeaturePipeline,
    TemporalFeatureCreator,
    LagFeatureCreator,
    InteractionFeatureCreator
)
from ..etl import DataLoader


config = Config()
logger = setup_logger(
    "features.build",
    log_file=config.reports_dir / "features.log",
    level=config.log_level,
    format_type=config.log_format
)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características temporales (hora del día, mes).
    
    LEGACY: Usa TemporalFeatureCreator del módulo feature_engineering.py
    """
    creator = TemporalFeatureCreator(include_cyclical=True)
    return creator.create(df)


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "precip_mm_hr",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Crea features de lag y rolling para la variable objetivo.
    
    LEGACY: Usa LagFeatureCreator del módulo feature_engineering.py
    """
    creator = LagFeatureCreator(
        target_col=target_col,
        lags=lags,
        rolling_windows=rolling_windows,
        include_deltas=True
    )
    return creator.create(df)


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features de interacción entre variables meteorológicas.
    
    LEGACY: Usa InteractionFeatureCreator del módulo feature_engineering.py
    """
    creator = InteractionFeatureCreator()
    return creator.create(df)


def filter_wind_variables(df: pd.DataFrame, keep_wind_10m: bool = False) -> pd.DataFrame:
    """
    Filtra variables de viento según configuración.
    
    Dado que wind_* 2m y wind_* 10m están altamente correlacionadas,
    por defecto solo se mantienen las de 2m.
    
    Args:
        df: DataFrame con variables
        keep_wind_10m: Si True, mantiene variables de 10m
    
    Returns:
        DataFrame filtrado
    """
    if not keep_wind_10m:
        cols_to_drop = [col for col in df.columns if "10m" in col]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Eliminadas variables de viento a 10m: {cols_to_drop}")
    
    return df


def build_features(
    input_path: Optional[Path] = None,
    keep_wind_10m: bool = False,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    use_new_pipeline: bool = True
) -> pd.DataFrame:
    """
    Pipeline completo de construcción de features.
    
    Args:
        input_path: Ruta del archivo processed (si None, usa latest)
        keep_wind_10m: Mantener variables de viento a 10m
        lags: Lista de lags personalizados
        rolling_windows: Ventanas móviles personalizadas
        use_new_pipeline: Usar nuevo pipeline modular (recomendado)
    
    Returns:
        DataFrame con features construidos
    """
    logger.info("="*60)
    logger.info("INICIANDO CONSTRUCCIÓN DE FEATURES")
    logger.info("="*60)
    
    # Cargar datos processed
    if input_path is None:
        input_path = config.processed_data_dir / "processed_latest.parquet"
    
    if not input_path.exists():
        logger.error(f"Archivo no encontrado: {input_path}")
        raise FileNotFoundError(f"No existe {input_path}")
    
    logger.info(f"Cargando datos desde {input_path}")
    df = load_dataframe(input_path)
    
    initial_rows = len(df)
    logger.info(f"Datos cargados: {initial_rows} registros, {len(df.columns)} columnas")
    
    # Asegurar que timestamp es columna
    if "timestamp" not in df.columns:
        df = df.reset_index()
    
    # Usar nuevo pipeline modular
    if use_new_pipeline:
        logger.info("Usando pipeline modular de features")
        
        # Filtrar variables de viento si es necesario
        if not keep_wind_10m:
            df = filter_wind_variables(df, keep_wind_10m)
        
        # Crear pipeline con configuración personalizada
        creators = [
            TemporalFeatureCreator(include_cyclical=True),
            LagFeatureCreator(
                target_col="precip_mm_hr",
                lags=lags,
                rolling_windows=rolling_windows
            ),
            InteractionFeatureCreator()
        ]
        
        pipeline = FeaturePipeline(creators=creators, drop_initial_nans=True)
        df_clean = pipeline.transform(df)
    
    else:
        # Usar pipeline antiguo (retrocompatibilidad)
        logger.info("Usando pipeline legacy de features")
        
        # Asegurar que timestamp es índice y está ordenado
        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()
        
        # Asegurar frecuencia horaria
        original_freq = len(df)
        df = df.asfreq("H")
        
        if len(df) > original_freq:
            logger.warning(
                f"Se completó la frecuencia horaria: {original_freq} → {len(df)} registros"
            )
        
        df = df.reset_index()
        
        # Aplicar transformaciones legacy
        df = filter_wind_variables(df, keep_wind_10m)
        df = create_temporal_features(df)
        df = create_lag_features(df, "precip_mm_hr", lags, rolling_windows)
        df = create_interaction_features(df)
        
        # Eliminar filas con NaN
        max_lag = max(config.lags) if lags is None else max(lags)
        df_clean = df.iloc[max_lag:].copy()
        dropped = len(df) - len(df_clean)
        logger.info(f"Eliminadas {dropped} filas iniciales con NaN en lags")
    
    # Información final
    logger.info(f"Features construidos: {len(df_clean)} registros, {len(df_clean.columns)} columnas")
    
    # Estadísticas de valores nulos
    null_counts = df_clean.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning("Valores nulos por columna:")
        for col, count in null_counts[null_counts > 0].items():
            pct = (count / len(df_clean)) * 100
            logger.warning(f"  {col}: {count} ({pct:.2f}%)")
    
    # Guardar en curated usando DataLoader
    loader = DataLoader()
    paths = loader.save_curated(df_clean, create_version=True)
    logger.info(f"Features guardados en: {paths['latest']}")
    
    logger.info("="*60)
    logger.info("CONSTRUCCIÓN DE FEATURES COMPLETADA")
    logger.info("="*60)
    
    return df_clean


