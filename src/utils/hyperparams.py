"""Carga centralizada de hiperparámetros desde configs/hyperparameters.json."""

import json
from pathlib import Path
from typing import Dict, Any


_HYPERPARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "hyperparameters.json"
_cache: Dict[str, Any] = {}


def load_hyperparameters(force_reload: bool = False) -> Dict[str, Any]:
    """
    Carga hiperparámetros desde configs/hyperparameters.json.

    Cachea el resultado en memoria para no leer disco en cada llamada.

    Args:
        force_reload: Si True, recarga desde disco ignorando cache.

    Returns:
        Diccionario con secciones 'lstm', 'xgboost', 'training', 'ensemble'.
    """
    global _cache
    if _cache and not force_reload:
        return _cache

    if _HYPERPARAMS_PATH.exists():
        with open(_HYPERPARAMS_PATH, "r") as f:
            _cache = json.load(f)
    else:
        _cache = {}

    return _cache


def get_lstm_defaults() -> Dict[str, Any]:
    """Retorna defaults de LSTM desde la configuración centralizada."""
    return load_hyperparameters().get("lstm", {})


def get_xgboost_defaults() -> Dict[str, Any]:
    """Retorna defaults de XGBoost desde la configuración centralizada."""
    return load_hyperparameters().get("xgboost", {})


def get_training_defaults() -> Dict[str, Any]:
    """Retorna defaults de training (splits, target) desde la configuración centralizada."""
    return load_hyperparameters().get("training", {})


def get_ensemble_defaults() -> Dict[str, Any]:
    """Retorna defaults de ensemble (pesos) desde la configuración centralizada."""
    return load_hyperparameters().get("ensemble", {})
