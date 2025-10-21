"""Gestión de configuración desde .env"""

import os
from pathlib import Path
from typing import Any, List
from dotenv import load_dotenv


class Config:
    """Clase para gestionar la configuración del proyecto."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Cargar .env desde la raíz del proyecto
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Cargar ejemplo si no existe .env
            load_dotenv(project_root / "configs" / ".env.example")
        
        self.project_root = project_root
        self._initialized = True
    
    def get(self, key: str, default: Any = None, cast_type: type = str) -> Any:
        """Obtiene una variable de configuración con cast opcional."""
        value = os.getenv(key, default)
        
        if value is None:
            return default
        
        if cast_type == bool:
            return value.lower() in ("true", "1", "yes")
        elif cast_type == int:
            return int(value)
        elif cast_type == float:
            return float(value)
        elif cast_type == list:
            # Si el valor es una lista, devolverla directamente
            if isinstance(value, list):
                return value
            # Si es string, dividir por comas
            return [x.strip() for x in str(value).split(",") if x.strip()]
        
        return str(value)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Obtiene un entero de configuración."""
        return self.get(key, default, int)
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Obtiene un float de configuración."""
        return self.get(key, default, float)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Obtiene un booleano de configuración."""
        return self.get(key, default, bool)
    
    def get_list(self, key: str, default: List[str] = None) -> List[str]:
        """Obtiene una lista de configuración."""
        if default is None:
            default = []
        
        # Si ya es una lista, devolverla directamente
        if isinstance(default, list):
            default_value = default
        else:
            default_value = []
        
        return self.get(key, default_value, list)
    
    def get_path(self, key: str, default: str = "") -> Path:
        """Obtiene una ruta de configuración relativa a la raíz del proyecto."""
        value = self.get(key, default)
        return self.project_root / value
    
    # Propiedades de acceso rápido
    @property
    def rh_high(self) -> float:
        return self.get_float("RH_HIGH", 90.0)
    
    @property
    def rh_medium(self) -> float:
        return self.get_float("RH_MEDIUM", 85.0)
    
    @property
    def temp_drop_2h(self) -> float:
        return self.get_float("TEMP_DROP_2H", -0.5)
    
    @property
    def wind_calm_ms(self) -> float:
        return self.get_float("WIND_CALM_MS", 1.0)
    
    @property
    def precip_event_mmhr(self) -> float:
        return self.get_float("PRECIP_EVENT_MMHR", 0.5)
    
    @property
    def skip_rows(self) -> int:
        return self.get_int("SKIP_ROWS", 9)
    
    @property
    def delimiter(self) -> str:
        return self.get("DELIMITER", ";")
    
    @property
    def raw_data_dir(self) -> Path:
        return self.get_path("RAW_DATA_DIR", "data/raw")
    
    @property
    def processed_data_dir(self) -> Path:
        return self.get_path("PROCESSED_DATA_DIR", "data/processed")
    
    @property
    def curated_data_dir(self) -> Path:
        return self.get_path("CURATED_DATA_DIR", "data/curated")
    
    @property
    def predictions_dir(self) -> Path:
        return self.get_path("PREDICTIONS_DIR", "data/predictions")
    
    @property
    def models_dir(self) -> Path:
        return self.get_path("MODELS_DIR", "models")
    
    @property
    def reports_dir(self) -> Path:
        return self.get_path("REPORTS_DIR", "reports")
    
    @property
    def hidden_size(self) -> int:
        return self.get_int("HIDDEN_SIZE", 64)
    
    @property
    def num_layers(self) -> int:
        return self.get_int("NUM_LAYERS", 2)
    
    @property
    def dropout(self) -> float:
        return self.get_float("DROPOUT", 0.2)
    
    @property
    def learning_rate(self) -> float:
        return self.get_float("LEARNING_RATE", 0.001)
    
    @property
    def epochs(self) -> int:
        return self.get_int("EPOCHS", 50)
    
    @property
    def batch_size(self) -> int:
        return self.get_int("BATCH_SIZE", 64)
    
    @property
    def lookback(self) -> int:
        return self.get_int("LOOKBACK", 24)
    
    @property
    def horizon(self) -> int:
        return self.get_int("HORIZON", 1)
    
    @property
    def train_split(self) -> float:
        return self.get_float("TRAIN_SPLIT", 0.7)
    
    @property
    def val_split(self) -> float:
        return self.get_float("VAL_SPLIT", 0.15)
    
    @property
    def test_split(self) -> float:
        return self.get_float("TEST_SPLIT", 0.15)
    
    @property
    def keep_wind_10m(self) -> bool:
        return self.get_bool("KEEP_WIND_10M", False)
    
    @property
    def lags(self) -> List[int]:
        lags_str = self.get_list("LAGS", ["1", "2", "3", "6", "12"])
        return [int(x) for x in lags_str]
    
    @property
    def rolling_windows(self) -> List[int]:
        windows_str = self.get_list("ROLLING_WINDOWS", ["3", "6"])
        return [int(x) for x in windows_str]
    
    @property
    def log_level(self) -> str:
        return self.get("LOG_LEVEL", "INFO")
    
    @property
    def log_format(self) -> str:
        return self.get("LOG_FORMAT", "json")


