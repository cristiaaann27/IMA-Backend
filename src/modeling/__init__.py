"""Módulo de modelado para predicción de precipitación."""

# Clase base
from .base import BaseModel, DataPreparator

# Modelos específicos
from .train_lstm import (
    LSTMModel,
    LSTMModelWrapper,
    train_model as train_lstm,
    prepare_data as prepare_lstm_data,
    create_sequences
)
from .train_xgboost import (
    XGBoostModelWrapper,
    train_xgboost_model,
    prepare_data as prepare_xgboost_data
)

# Evaluación
from .evaluate import evaluate_model, calculate_regression_metrics, calculate_classification_metrics

# Predicción unificada
from .predictor import UnifiedPredictor, HybridPredictor, predict, ModelType

__all__ = [
    # Clase base
    "BaseModel",
    "DataPreparator",
    
    # LSTM
    "LSTMModel",
    "LSTMModelWrapper",
    "train_lstm",
    "prepare_lstm_data",
    "create_sequences",
    
    # XGBoost
    "XGBoostModelWrapper",
    "train_xgboost_model",
    "prepare_xgboost_data",
    
    # Evaluación
    "evaluate_model",
    "calculate_regression_metrics",
    "calculate_classification_metrics",
    
    # Predicción
    "UnifiedPredictor",
    "HybridPredictor",
    "predict",
    "ModelType"
]


