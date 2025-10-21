from typing import List, Optional

from pydantic import BaseModel, Field


class ModelArchitecture(BaseModel):
    input_size: int = Field(description="Tamaño de entrada")
    hidden_size: int = Field(description="Tamaño de capa oculta")
    num_layers: int = Field(description="Número de capas LSTM")
    dropout: float = Field(description="Tasa de dropout")


class TrainingInfo(BaseModel):
    epochs: int = Field(description="Épocas entrenadas")
    best_epoch: int = Field(description="Mejor época")
    best_val_loss: float = Field(description="Mejor loss de validación")
    final_train_loss: float = Field(description="Loss final de entrenamiento")
    learning_rate: float = Field(description="Learning rate")
    batch_size: int = Field(description="Batch size")


class ModelMetrics(BaseModel):
    mae: Optional[float] = Field(default=None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(default=None, description="Root Mean Squared Error")
    mape: Optional[float] = Field(default=None, description="Mean Absolute Percentage Error")
    r2: Optional[float] = Field(default=None, description="R² Score")
    precision: Optional[float] = Field(default=None, description="Precision (evento de lluvia)")
    recall: Optional[float] = Field(default=None, description="Recall (evento de lluvia)")
    f1_score: Optional[float] = Field(default=None, description="F1-Score (evento de lluvia)")


class DataInfo(BaseModel):
    n_features: int = Field(description="Número de features")
    feature_cols: List[str] = Field(description="Nombres de features")
    lookback: int = Field(description="Ventana lookback (horas)")
    horizon: int = Field(description="Horizonte de predicción (horas)")
    train_samples: int = Field(description="Samples de entrenamiento")
    val_samples: int = Field(description="Samples de validación")
    test_samples: int = Field(description="Samples de test")


class ModelInfo(BaseModel):
    timestamp: str = Field(description="Timestamp del modelo")
    lstm: dict = Field(description="Información del modelo LSTM")
    xgboost: dict = Field(description="Información del modelo XGBoost")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "20250115_120000",
                "lstm": {
                    "architecture": {
                        "input_size": 24,
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.2
                    },
                    "training": {
                        "epochs": 24,
                        "best_epoch": 14,
                        "best_val_loss": 0.1512,
                        "final_train_loss": 0.0643,
                        "learning_rate": 0.001,
                        "batch_size": 64
                    },
                    "data": {
                        "n_features": 24,
                        "feature_cols": ["rh_2m_pct", "temp_2m_c", "..."],
                        "lookback": 24,
                        "horizon": 1,
                        "train_samples": 151839,
                        "val_samples": 32518,
                        "test_samples": 32519
                    },
                    "metrics": {
                        "mae": 0.9292,
                        "rmse": 3.4836,
                        "mape": 74.82,
                        "r2": 0.8612,
                        "precision": 0.9583,
                        "recall": 0.9658,
                        "f1_score": 0.9620
                    }
                },
                "xgboost": {
                    "model_type": "xgboost",
                    "metrics": {
                        "mae": 0.2493,
                        "rmse": 2.8976,
                        "r2": 0.9039,
                        "mape": 7354587.54,
                        "precision": 0.9922,
                        "recall": 0.9939,
                        "f1_score": 0.9930,
                        "auc": 0.9993
                    },
                    "n_features": 24,
                    "hyperparameters": {
                        "learning_rate": 0.1,
                        "max_depth": 6,
                        "n_estimators": 200,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8
                    }
                }
            }
        }
    }

