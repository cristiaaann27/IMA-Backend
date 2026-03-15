"""Carga centralizada de hiperparámetros desde configs/hyperparameters.json y configs/best_hyperparams.json (Optuna)."""

import json
from pathlib import Path
from typing import Dict, Any


_HYPERPARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "hyperparameters.json"
_BEST_HYPERPARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "best_hyperparams.json"
_cache: Dict[str, Any] = {}
_best_cache: Dict[str, Any] = {}


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


def get_best_hyperparams(force_reload: bool = False) -> Dict[str, Any]:
    """
    Carga mejores hiperparámetros generados por Optuna desde configs/best_hyperparams.json.

    Returns:
        Diccionario con los mejores hiperparámetros, o {} si no existe.
    """
    global _best_cache
    if _best_cache and not force_reload:
        return _best_cache

    if _BEST_HYPERPARAMS_PATH.exists():
        with open(_BEST_HYPERPARAMS_PATH, "r") as f:
            _best_cache = json.load(f)
    else:
        _best_cache = {}

    return _best_cache


def get_effective_lstm_params() -> Dict[str, Any]:
    """
    Retorna hiperparámetros LSTM efectivos: best_hyperparams.json (Optuna) > hyperparameters.json (defaults).
    """
    defaults = get_lstm_defaults()
    best = get_best_hyperparams()

    # Mapeo de keys de Optuna a keys de hyperparameters.json
    mapping = {
        "lstm_hidden_size": "hidden_size",
        "lstm_num_layers": "num_layers",
        "lstm_dropout": "dropout",
        "lstm_lr": "learning_rate",
        "lstm_batch_size": "batch_size",
    }

    for optuna_key, param_key in mapping.items():
        if optuna_key in best:
            defaults[param_key] = best[optuna_key]

    return defaults


def get_effective_xgboost_params() -> Dict[str, Any]:
    """
    Retorna hiperparámetros XGBoost efectivos: best_hyperparams.json (Optuna) > hyperparameters.json (defaults).
    """
    defaults = get_xgboost_defaults()
    best = get_best_hyperparams()

    mapping = {
        "xgb_n_estimators": "n_estimators",
        "xgb_max_depth": "max_depth",
        "xgb_lr": "learning_rate",
        "xgb_subsample": "subsample",
        "xgb_colsample": "colsample_bytree",
        "xgb_min_child_weight": "min_child_weight",
        "xgb_gamma": "gamma",
        "xgb_reg_alpha": "reg_alpha",
        "xgb_reg_lambda": "reg_lambda",
    }

    for optuna_key, param_key in mapping.items():
        if optuna_key in best:
            defaults[param_key] = best[optuna_key]

    return defaults
