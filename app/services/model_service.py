"""Servicio de gestión del modelo LSTM."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from app.core.config import get_settings
from app.core.exceptions import ModelNotLoadedException
from app.core.logging import get_logger
from app.repositories.model_repository import ModelRepository, get_model_repository
from src.modeling.train_lstm import LSTMModel


logger = get_logger(__name__)
settings = get_settings()


class ModelService:
    """Servicio singleton para gestión del modelo."""
    
    _instance = None
    _model: Optional[LSTMModel] = None
    _scalers: Optional[Dict[str, Any]] = None
    _metadata: Optional[Dict[str, Any]] = None
    _loaded_at: Optional[datetime] = None
    _repository: Optional[ModelRepository] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa el servicio."""
        if self._repository is None:
            self._repository = get_model_repository()
            logger.info("ModelService inicializado")
    
    def load_model(self) -> None:
        """
        Carga modelo, scalers y metadata.
        
        Raises:
            ModelNotLoadedException: Si falla la carga
        """
        try:
            logger.info("Cargando modelo...")
            
            # Cargar metadata primero para obtener arquitectura
            self._metadata = self._repository.load_metadata()
            
            arch = self._metadata["model_architecture"]
            
            # Crear modelo con arquitectura correcta
            data_config = self._metadata.get("data", {})
            horizon = data_config.get("horizon", 6)
            self._model = LSTMModel(
                input_size=arch["input_size"],
                hidden_size=arch["hidden_size"],
                num_layers=arch["num_layers"],
                dropout=arch["dropout"],
                output_size=horizon
            )
            
            # Cargar state dict
            state_dict = self._repository.load_model_state()
            self._model.load_state_dict(state_dict)
            self._model.eval()  # Modo evaluación
            
            # Cargar scalers
            self._scalers = self._repository.load_scaler()
            
            self._loaded_at = datetime.now(timezone.utc)
            
            logger.info(f"Modelo cargado exitosamente")
            logger.info(f"  Arquitectura: {arch}")
            logger.info(f"  Features: {self._metadata['data']['n_features']}")
            logger.info(f"  Lookback: {self._metadata['data']['lookback']}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise ModelNotLoadedException(f"Error cargando modelo: {str(e)}")
    
    def reload_model(self) -> None:
        """Recarga el modelo sin reiniciar el servicio."""
        logger.info("Recargando modelo...")
        self._model = None
        self._scalers = None
        self._metadata = None
        self.load_model()
        logger.info("Modelo recargado exitosamente")
    
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado."""
        return (
            self._model is not None and
            self._scalers is not None and
            self._metadata is not None
        )
    
    def get_model(self) -> LSTMModel:
        """
        Obtiene instancia del modelo.
        
        Returns:
            Modelo LSTM
        
        Raises:
            ModelNotLoadedException: Si modelo no está cargado
        """
        if not self.is_loaded():
            raise ModelNotLoadedException("Modelo no cargado")
        return self._model
    
    def get_scalers(self) -> Dict[str, Any]:
        """
        Obtiene scalers.
        
        Returns:
            Diccionario con scaler_X y scaler_y
        
        Raises:
            ModelNotLoadedException: Si modelo no está cargado
        """
        if not self.is_loaded():
            raise ModelNotLoadedException("Modelo no cargado")
        return self._scalers
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Obtiene metadata del modelo.
        
        Returns:
            Metadata completa
        
        Raises:
            ModelNotLoadedException: Si modelo no está cargado
        """
        if not self.is_loaded():
            raise ModelNotLoadedException("Modelo no cargado")
        return self._metadata
    
    def get_loaded_at(self) -> Optional[datetime]:
        """Retorna timestamp de carga del modelo."""
        return self._loaded_at
    
    def get_feature_names(self) -> list:
        """
        Obtiene nombres de features esperados.
        
        Returns:
            Lista de nombres de features
        """
        if not self.is_loaded():
            raise ModelNotLoadedException("Modelo no cargado")
        return self._metadata["data"]["feature_cols"]
    
    def get_lookback(self) -> int:
        """Obtiene ventana lookback esperada."""
        if not self.is_loaded():
            raise ModelNotLoadedException("Modelo no cargado")
        return self._metadata["data"]["lookback"]
    
    def get_lstm_info(self) -> dict:
        """
        Obtiene información completa del modelo LSTM.
        
        Returns:
            Diccionario con arquitectura, training, data y métricas del LSTM
        """
        try:
            # Leer metadata del LSTM
            metadata_path = Path("models/lstm_metadata.json")
            eval_results_path = Path("reports/evaluation_results.json")
            
            lstm_info = {}
            
            # Leer metadata básica
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                lstm_info.update({
                    "architecture": metadata.get("model_architecture", {}),
                    "training": metadata.get("training", {}),
                    "data": metadata.get("data", {}),
                    "timestamp": metadata.get("timestamp", "unknown")
                })
            
            # Leer métricas de evaluación
            if eval_results_path.exists():
                with open(eval_results_path, 'r') as f:
                    eval_data = json.load(f)
                
                test_metrics = eval_data.get("splits", {}).get("test", {})
                regression_metrics = test_metrics.get("regression", {})
                classification_metrics = test_metrics.get("classification", {})
                
                lstm_info["metrics"] = {
                    "mae": regression_metrics.get("MAE"),
                    "rmse": regression_metrics.get("RMSE"),
                    "r2": regression_metrics.get("R2"),
                    "mape": regression_metrics.get("MAPE"),
                    "precision": classification_metrics.get("precision"),
                    "recall": classification_metrics.get("recall"),
                    "f1_score": classification_metrics.get("f1_score"),
                    "f2_score": classification_metrics.get("f2_score")
                }
            
            return lstm_info
            
        except Exception as e:
            logger.warning(f"Error cargando información LSTM: {e}")
            return {}
    
    def get_xgboost_info(self) -> dict:
        """
        Obtiene información completa del modelo XGBoost.
        
        Returns:
            Diccionario con metadata y métricas del XGBoost
        """
        try:
            # Leer metadata del XGBoost
            xgb_metadata_path = Path("models/xgboost_metadata.json")
            
            if xgb_metadata_path.exists():
                with open(xgb_metadata_path, 'r') as f:
                    xgb_data = json.load(f)
                
                # Extraer información relevante
                xgb_info = {
                    "model_type": xgb_data.get("model_type", "xgboost"),
                    "timestamp": xgb_data.get("timestamp", "unknown"),
                    "n_features": xgb_data.get("n_features", 24),
                    "metrics": xgb_data.get("metrics", {}),
                    "hyperparameters": {
                        key: value for key, value in xgb_data.get("hyperparameters", {}).items()
                        if key in ["learning_rate", "max_depth", "n_estimators", "subsample", "colsample_bytree", "gamma", "min_child_weight"]
                    }
                }
                
                return xgb_info
            
            return {}
            
        except Exception as e:
            logger.warning(f"Error cargando información XGBoost: {e}")
            return {}


def get_model_service() -> ModelService:
    """Factory function para obtener ModelService singleton."""
    return ModelService()

