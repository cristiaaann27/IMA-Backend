"""
Búsqueda de hiperparámetros con Optuna para LSTM y XGBoost.

Usa pruning para descartar trials poco prometedores.
Guarda los mejores hiperparámetros en configs/best_hyperparams.json.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
from optuna.exceptions import TrialPruned

from ..utils import (
    Config,
    setup_logger,
    load_dataframe,
    set_global_seed,
    sanitize_scaled,
)
from .train_lstm import prepare_data, LSTMModel, TimeSeriesDataset, create_sequences
from .train_xgboost import train_xgboost_model, evaluate_model as evaluate_xgboost

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

config = Config()
logger = setup_logger(
    "modeling.optuna_tune",
    log_file=config.reports_dir / "optuna_tune.log",
    level=config.log_level,
    format_type=config.log_format,
)

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "configs" / "best_hyperparams.json"


def _train_lstm_trial(
    trial: optuna.Trial,
    data_dict: Dict,
    device: str,
    seed: int = 42,
) -> float:
    """Entrena LSTM con hiperparámetros sugeridos por Optuna. Retorna val RMSE."""

    hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("lstm_num_layers", 1, 3)
    dropout = trial.suggest_float("lstm_dropout", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("lstm_batch_size", [32, 64, 128])

    set_global_seed(seed)

    train_dataset = TimeSeriesDataset(data_dict["X_train"], data_dict["y_train"])
    val_dataset = TimeSeriesDataset(data_dict["X_val"], data_dict["y_val"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = data_dict["n_features"]
    horizon = data_dict["y_train"].shape[1] if data_dict["y_train"].ndim >= 2 else 1

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=horizon,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 30
    patience = 7
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_losses.append(criterion(model(X_b), y_b).item())

        val_loss = float(np.mean(val_losses))

        # Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return float(np.sqrt(best_val_loss))


def _train_xgb_trial(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Entrena XGBoost con hiperparámetros sugeridos. Retorna val RMSE."""

    params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0, step=0.1),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 0.5, step=0.05),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 1.0, step=0.1),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.5, 2.0, step=0.1),
        "random_state": 42,
        "early_stopping_rounds": 10,
    }

    model = train_xgboost_model(X_train, y_train, X_val, y_val, params)
    y_pred = model.predict(X_val)
    rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
    return rmse


def objective(
    trial: optuna.Trial,
    data_dict: Dict,
    X_train_flat: np.ndarray,
    y_train_flat: np.ndarray,
    X_val_flat: np.ndarray,
    y_val_flat: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Objetivo combinado: minimiza promedio de val RMSE de LSTM y XGBoost.
    """
    lstm_rmse = _train_lstm_trial(trial, data_dict, device)
    xgb_rmse = _train_xgb_trial(trial, X_train_flat, y_train_flat, X_val_flat, y_val_flat)

    combined = 0.5 * lstm_rmse + 0.5 * xgb_rmse
    logger.info(
        f"Trial {trial.number}: LSTM_RMSE={lstm_rmse:.4f}, XGB_RMSE={xgb_rmse:.4f}, combined={combined:.4f}"
    )
    return combined


def run_tuning(
    n_trials: int = 30,
    timeout: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Ejecuta la búsqueda de hiperparámetros y guarda los mejores en JSON.

    Args:
        n_trials: Número de trials de Optuna.
        timeout: Timeout en segundos (opcional).
        seed: Semilla global.

    Returns:
        Diccionario con los mejores hiperparámetros.
    """
    logger.info("=" * 60)
    logger.info("BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA")
    logger.info("=" * 60)

    set_global_seed(seed)

    # Cargar datos
    curated_path = config.curated_data_dir / "curated_latest.parquet"
    if not curated_path.exists():
        raise FileNotFoundError(f"No se encontró {curated_path}. Ejecuta 'features' primero.")

    df = load_dataframe(curated_path)
    data_dict = prepare_data(df)

    # Preparar datos planos para XGBoost (sin secuencias)
    exclude_cols = ["timestamp", "precip_mm_hr"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_all = df[feature_cols].values
    y_all = df["precip_mm_hr"].values

    valid = ~(np.isnan(X_all).any(axis=1) | np.isnan(y_all))
    X_all, y_all = X_all[valid], y_all[valid]

    n = len(X_all)
    t_end = int(n * 0.7)
    v_end = int(n * 0.85)

    X_train_flat, y_train_flat = X_all[:t_end], y_all[:t_end]
    X_val_flat, y_val_flat = X_all[t_end:v_end], y_all[t_end:v_end]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="ima_hypertune",
    )

    study.optimize(
        lambda trial: objective(
            trial, data_dict, X_train_flat, y_train_flat, X_val_flat, y_val_flat, device
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    best = study.best_params
    best["best_value"] = study.best_value
    logger.info(f"Mejor trial: {study.best_trial.number}, valor: {study.best_value:.4f}")
    logger.info(f"Mejores hiperparámetros: {best}")

    # Guardar
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"Hiperparámetros guardados en {OUTPUT_PATH}")

    return best
