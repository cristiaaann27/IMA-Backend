import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from app.core.config import get_settings
from app.core.exceptions import ResourceNotFoundException
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


class ModelRepository(ABC):
    
    @abstractmethod
    def load_model_state(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def load_scaler(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def load_metadata(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def save_prediction(self, prediction_data: Dict[str, Any], filename: str) -> str:
        pass
    
    @abstractmethod
    def model_exists(self) -> bool:
        pass


class LocalModelRepository(ModelRepository):
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        predictions_dir: Optional[Path] = None
    ):

        self.model_path = model_path or settings.model_full_path
        self.scaler_path = scaler_path or settings.scaler_full_path
        self.metadata_path = metadata_path or settings.metadata_full_path
        self.predictions_dir = predictions_dir or settings.predictions_dir
        
        logger.info(f"LocalModelRepository inicializado:")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Scaler: {self.scaler_path}")
        logger.info(f"  Metadata: {self.metadata_path}")
    
    def load_model_state(self) -> Dict[str, Any]:

        if not self.model_path.exists():
            raise ResourceNotFoundException(
                f"Modelo no encontrado en {self.model_path}"
            )
        
        try:
            state_dict = torch.load(
                self.model_path,
                map_location="cpu" 
            )
            logger.info(f"Modelo cargado desde {self.model_path}")
            return state_dict
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise ResourceNotFoundException(
                f"Error cargando modelo: {str(e)}"
            )
    
    def load_scaler(self) -> Dict[str, Any]:

        if not self.scaler_path.exists():
            raise ResourceNotFoundException(
                f"Scaler no encontrado en {self.scaler_path}"
            )
        
        try:
            with open(self.scaler_path, "rb") as f:
                scalers = pickle.load(f)
            
            logger.info(f"Scalers cargados desde {self.scaler_path}")
            return scalers
        except Exception as e:
            logger.error(f"Error cargando scalers: {e}")
            raise ResourceNotFoundException(
                f"Error cargando scalers: {str(e)}"
            )
    
    def load_metadata(self) -> Dict[str, Any]:

        if not self.metadata_path.exists():
            raise ResourceNotFoundException(
                f"Metadata no encontrada en {self.metadata_path}"
            )
        
        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            
            logger.info(f"Metadata cargada desde {self.metadata_path}")
            return metadata
        except Exception as e:
            logger.error(f"Error cargando metadata: {e}")
            raise ResourceNotFoundException(
                f"Error cargando metadata: {str(e)}"
            )
    
    def save_prediction(self, prediction_data: Dict[str, Any], filename: str) -> str:
 
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = self.predictions_dir / filename
        
        try:
            with open(filepath, "w") as f:
                json.dump(prediction_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Predicción guardada en {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error guardando predicción: {e}")
            raise
    
    def model_exists(self) -> bool:

        return (
            self.model_path.exists() and
            self.scaler_path.exists() and
            self.metadata_path.exists()
        )


def get_model_repository() -> ModelRepository:

    backend = settings.model_backend.lower()
    
    if backend == "local":
        return LocalModelRepository()
    else:
        raise ValueError(f"Backend no soportado: {backend}")

