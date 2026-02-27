"""Clase base abstracta para modelos de predicción de precipitación."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import json

import numpy as np
import pandas as pd

from ..utils import Config, setup_logger, ensure_dir


config = Config()
logger = setup_logger(
    "modeling.base",
    log_file=config.reports_dir / "modeling.log",
    level=config.log_level,
    format_type=config.log_format
)


class BaseModel(ABC):
    """Clase base abstracta para modelos de predicción."""
    
    def __init__(self, model_name: str):
        """
        Inicializa el modelo base.
        
        Args:
            model_name: Nombre del modelo (ej: 'lstm', 'xgboost')
        """
        self.model_name = model_name
        self.model = None
        self.metadata = {}
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Dict:
        """
        Entrena el modelo.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación
            y_val: Target de validación
            **kwargs: Hiperparámetros adicionales
            
        Returns:
            Diccionario con historia/métricas de entrenamiento
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones.
        
        Args:
            X: Features de entrada
            
        Returns:
            Predicciones
        """
        pass
    
    @abstractmethod
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """
        Guarda el modelo y metadata.
        
        Args:
            output_dir: Directorio de salida
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        pass
    
    @abstractmethod
    def load(self, model_path: Path, metadata_path: Optional[Path] = None):
        """
        Carga el modelo y metadata.
        
        Args:
            model_path: Ruta del modelo
            metadata_path: Ruta de metadata (opcional)
        """
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 threshold: float = 0.5) -> Dict[str, float]:
        """
        Evalúa el modelo con métricas estándar.
        
        Args:
            X_test: Features de test
            y_test: Target de test
            threshold: Umbral para clasificación binaria
            
        Returns:
            Diccionario con métricas
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            precision_recall_fscore_support,
            fbeta_score
        )
        
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Predicciones
        y_pred = self.predict(X_test)
        
        # Asegurar que son 1D
        y_test_flat = y_test.ravel()
        y_pred_flat = y_pred.ravel()
        
        # Métricas de regresión
        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
        r2 = r2_score(y_test_flat, y_pred_flat)
        
        # MAPE
        mask = y_test_flat != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test_flat[mask] - y_pred_flat[mask]) / y_test_flat[mask])) * 100
        else:
            mape = np.nan
        
        # Clasificación binaria (evento de lluvia)
        y_test_binary = (y_test_flat >= threshold).astype(int)
        y_pred_binary = (y_pred_flat >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_binary, y_pred_binary, average='binary', zero_division=0
        )
        
        # F2-score (beta=2, pondera recall más que precision)
        f2 = fbeta_score(y_test_binary, y_pred_binary, beta=2, zero_division=0)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape) if not np.isnan(mape) else None,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'f2_score': float(f2)
        }
        
        logger.info(f"Evaluación {self.model_name}:")
        logger.info(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, F2: {f2:.4f}")
        
        return metrics
    
    def _save_metadata(self, output_dir: Path, additional_data: Dict = None) -> Path:
        """
        Guarda metadata del modelo.
        
        Args:
            output_dir: Directorio de salida
            additional_data: Datos adicionales a incluir
            
        Returns:
            Ruta del archivo de metadata
        """
        ensure_dir(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        metadata = {
            'model_name': self.model_name,
            'timestamp': timestamp,
            'is_trained': self.is_trained
        }
        
        if additional_data:
            metadata.update(additional_data)
        
        self.metadata = metadata
        
        metadata_path = output_dir / f'{self.model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata guardada en {metadata_path}")
        
        return metadata_path
    
    def _load_metadata(self, metadata_path: Path) -> Dict:
        """
        Carga metadata del modelo.
        
        Args:
            metadata_path: Ruta del archivo de metadata
            
        Returns:
            Diccionario con metadata
        """
        if not metadata_path.exists():
            logger.warning(f"Metadata no encontrada: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.metadata = metadata
        logger.info(f"Metadata cargada desde {metadata_path}")
        
        return metadata


class DataPreparator:
    """Clase auxiliar para preparación de datos común a todos los modelos."""
    
    @staticmethod
    def prepare_temporal_split(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ) -> Tuple[np.ndarray, ...]:
        """
        Prepara split temporal de datos.
        
        Args:
            df: DataFrame con datos
            target_col: Columna objetivo
            feature_cols: Columnas de features (si None, usa todas excepto timestamp y target)
            train_split: Proporción de train
            val_split: Proporción de validación
            test_split: Proporción de test
            
        Returns:
            Tupla (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)
        """
        # Seleccionar features
        if feature_cols is None:
            exclude_cols = ["timestamp", target_col]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Extraer X e y
        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)
        
        # Eliminar filas con NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).ravel())
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Datos válidos: {len(X)} registros")
        logger.info(f"Features seleccionados: {len(feature_cols)} columnas")
        
        # Split temporal
        n = len(X)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Split temporal: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    
    @staticmethod
    def validate_features(df: pd.DataFrame, required_features: list) -> bool:
        """
        Valida que el DataFrame contenga las features requeridas.
        
        Args:
            df: DataFrame a validar
            required_features: Lista de features requeridas
            
        Returns:
            True si todas las features están presentes
            
        Raises:
            ValueError: Si faltan features
        """
        missing = [col for col in required_features if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        return True
