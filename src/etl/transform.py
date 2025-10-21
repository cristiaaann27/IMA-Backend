"""Módulo de transformación de datos meteorológicos (Transform)."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from ..utils import Config, setup_logger
from .validators import DataValidator


config = Config()
logger = setup_logger(
    "etl.transform",
    log_file=config.reports_dir / "etl.log",
    level=config.log_level,
    format_type=config.log_format
)


class DataMerger:
    """Consolidador de múltiples variables en un único DataFrame."""
    
    @staticmethod
    def merge_variables(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge de múltiples DataFrames de variables por timestamp.
        
        Args:
            dfs: Diccionario {nombre_variable: DataFrame}
        
        Returns:
            DataFrame consolidado
        """
        logger.info(f"🔗 Consolidando {len(dfs)} variables")
        
        if not dfs:
            raise ValueError("No hay DataFrames para consolidar")
        
        df_merged = None
        
        for var_name, df in dfs.items():
            if df_merged is None:
                df_merged = df.copy()
                logger.debug(f"  Base: {var_name} ({len(df)} registros)")
            else:
                df_merged = pd.merge(
                    df_merged,
                    df,
                    on="timestamp",
                    how="outer"
                )
                logger.debug(f"  + {var_name} ({len(df)} registros)")
        
        df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(
            f"✅ Consolidado: {len(df_merged)} registros, "
            f"{len(df_merged.columns) - 1} variables"
        )
        logger.info(
            f"📅 Rango: {df_merged['timestamp'].min()} → {df_merged['timestamp'].max()}"
        )
        
        return df_merged


class DataCleaner:
    """Limpiador y validador de datos."""
    
    def __init__(self, validator: DataValidator = None):
        """
        Inicializa el limpiador.
        
        Args:
            validator: Validador personalizado (opcional)
        """
        self.validator = validator or DataValidator()
    
    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza y validación completa de datos.
        
        Args:
            df: DataFrame a limpiar
        
        Returns:
            DataFrame limpio y validado
        """
        logger.info("🧹 Iniciando limpieza y validación")
        
        initial_rows = len(df)
        
        # 1. Eliminar duplicados
        df = self._remove_duplicates(df)
        
        # 2. Validar rangos
        df = self._validate_ranges(df)
        
        # 3. Limpiar outliers
        df = self._clean_outliers(df)
        
        # 4. Validar consistencia temporal
        self._check_temporal_consistency(df)
        
        final_rows = len(df)
        logger.info(
            f"✅ Limpieza completada: {initial_rows} → {final_rows} registros "
            f"({initial_rows - final_rows} eliminados)"
        )
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina timestamps duplicados."""
        duplicates = df["timestamp"].duplicated().sum()
        
        if duplicates > 0:
            logger.warning(f"⚠️ Eliminando {duplicates} timestamps duplicados")
            df = df.drop_duplicates(subset=["timestamp"], keep="first")
        
        return df
    
    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida rangos de variables."""
        validation_result = self.validator.validate_dataframe(df)
        
        for error in validation_result.errors:
            logger.error(f"❌ {error}")
        
        for warning in validation_result.warnings:
            logger.warning(f"⚠️ {warning}")
        
        if not validation_result.valid:
            raise ValueError("Falló la validación de datos")
        
        return df
    
    def _clean_outliers(self, df: pd.DataFrame, method: str = "clip") -> pd.DataFrame:
        """Limpia outliers extremos."""
        df_clean, modified = self.validator.clean_outliers(df, method=method)
        
        if modified > 0:
            logger.info(f"🔧 Corregidos {modified} valores fuera de rango")
        
        return df_clean
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> None:
        """Verifica consistencia temporal."""
        temporal_result = self.validator.check_temporal_consistency(df)
        
        for warning in temporal_result.warnings:
            logger.warning(f"⏰ {warning}")


class DataTransformer:
    """Transformador principal de datos."""
    
    def __init__(self):
        """Inicializa el transformador."""
        self.merger = DataMerger()
        self.cleaner = DataCleaner()
    
    def transform(
        self,
        extracted_data: Dict[str, pd.DataFrame],
        fill_gaps: bool = True,
        freq: str = "H"
    ) -> pd.DataFrame:
        """
        Pipeline completo de transformación.
        
        Args:
            extracted_data: Datos extraídos {variable: DataFrame}
            fill_gaps: Completar gaps temporales
            freq: Frecuencia temporal (default: 'H' para horaria)
        
        Returns:
            DataFrame transformado y limpio
        """
        logger.info("="*60)
        logger.info("TRANSFORMACIÓN DE DATOS (TRANSFORM)")
        logger.info("="*60)
        
        # 1. Merge de variables
        df = self.merger.merge_variables(extracted_data)
        
        # 2. Completar gaps temporales si es necesario
        if fill_gaps:
            df = self._fill_temporal_gaps(df, freq)
        
        # 3. Limpieza y validación
        df = self.cleaner.clean_and_validate(df)
        
        # 4. Reporte de calidad
        self._quality_report(df)
        
        logger.info("="*60)
        
        return df
    
    def _fill_temporal_gaps(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Completa gaps temporales con frecuencia especificada.
        
        Args:
            df: DataFrame con timestamp
            freq: Frecuencia ('H', 'D', etc.)
        
        Returns:
            DataFrame con frecuencia completa
        """
        original_len = len(df)
        
        # Establecer timestamp como índice
        df = df.set_index("timestamp").sort_index()
        
        # Completar frecuencia
        df = df.asfreq(freq)
        
        # Resetear índice
        df = df.reset_index()
        
        if len(df) > original_len:
            added = len(df) - original_len
            logger.warning(
                f"⚠️ Completada frecuencia {freq}: {original_len} → {len(df)} "
                f"({added} registros añadidos con NaN)"
            )
        
        return df
    
    def _quality_report(self, df: pd.DataFrame) -> None:
        """Genera reporte de calidad de datos."""
        logger.info("📊 Reporte de calidad:")
        logger.info(f"  • Registros totales: {len(df)}")
        logger.info(f"  • Variables: {len(df.columns) - 1}")
        logger.info(f"  • Columnas: {list(df.columns)}")
        
        # Valores nulos
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning("  • Valores nulos:")
            for col, count in null_counts[null_counts > 0].items():
                pct = (count / len(df)) * 100
                logger.warning(f"    - {col}: {count} ({pct:.2f}%)")
        else:
            logger.info("  • Sin valores nulos ✅")
        
        # Estadísticas básicas
        logger.info("  • Estadísticas:")
        for col in df.columns:
            if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col]):
                logger.info(
                    f"    - {col}: "
                    f"min={df[col].min():.2f}, "
                    f"max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}"
                )


class FeatureEngineer:
    """Ingeniero de características (opcional en ETL)."""
    
    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características básicas temporales.
        
        Args:
            df: DataFrame con timestamp
        
        Returns:
            DataFrame con features básicos
        """
        df = df.copy()
        
        # Asegurar que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Features temporales básicos
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        logger.info("✅ Features temporales básicos añadidos")
        
        return df
    
    @staticmethod
    def filter_columns(
        df: pd.DataFrame,
        exclude_patterns: List[str] = None
    ) -> pd.DataFrame:
        """
        Filtra columnas según patrones.
        
        Args:
            df: DataFrame
            exclude_patterns: Patrones a excluir (ej: ['10m'])
        
        Returns:
            DataFrame filtrado
        """
        if exclude_patterns is None:
            return df
        
        cols_to_drop = []
        for pattern in exclude_patterns:
            cols_to_drop.extend([col for col in df.columns if pattern in col])
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"🗑️ Eliminadas columnas: {cols_to_drop}")
        
        return df
