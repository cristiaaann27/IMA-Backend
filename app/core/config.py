from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    api_title: str = "Precipitation Prediction API"
    api_version: str = "1.0.0"
    api_description: str = "API para predicción de precipitación con LSTM"
    api_prefix: str = "/api/v1"
    debug: bool = Field(default=False, env="DEBUG")
    
    api_key: str = Field(default="dev-key-change-in-production", env="API_KEY")
    api_key_header: str = "X-API-Key"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def curated_data_dir(self) -> Path:
        return self.project_root / "data" / "curated"
    
    @property
    def predictions_dir(self) -> Path:
        return self.project_root / "data" / "predictions"
    
    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"
    
    model_path: str = Field(default="models/lstm_latest.pt", env="MODEL_PATH")
    scaler_path: str = Field(default="models/scaler.pkl", env="SCALER_PATH")
    metadata_path: str = Field(default="models/lstm_metadata.json", env="METADATA_PATH")
    model_backend: str = Field(default="local", env="MODEL_BACKEND")
    
    lookback: int = Field(default=24, env="LOOKBACK")
    horizon: int = Field(default=1, env="HORIZON")
    
    rh_high: float = Field(default=90.0, env="RH_HIGH")
    rh_medium: float = Field(default=85.0, env="RH_MEDIUM")
    temp_drop_2h: float = Field(default=-0.5, env="TEMP_DROP_2H")
    wind_calm_ms: float = Field(default=1.0, env="WIND_CALM_MS")
    precip_event_mmhr: float = Field(default=0.5, env="PRECIP_EVENT_MMHR")
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="reports/api.log", env="LOG_FILE")
    
    max_forecast_horizon: int = Field(default=24, env="MAX_FORECAST_HORIZON")
    prediction_timeout_seconds: float = Field(default=5.0, env="PREDICTION_TIMEOUT")
    
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_namespace: str = Field(default="PrecipitationAPI", env="METRICS_NAMESPACE")
    
    @property
    def model_full_path(self) -> Path:
        return self.project_root / self.model_path
    
    @property
    def scaler_full_path(self) -> Path:
        return self.project_root / self.scaler_path
    
    @property
    def metadata_full_path(self) -> Path:
        return self.project_root / self.metadata_path
    
    @property
    def log_full_path(self) -> Path:
        return self.project_root / self.log_file


@lru_cache()
def get_settings() -> Settings:

    return Settings()

