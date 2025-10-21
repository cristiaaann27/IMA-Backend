"""Evaluación de modelo LSTM."""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
    auc
)

from ..utils import Config, setup_logger, ensure_dir
from .train_lstm import LSTMModel, TimeSeriesDataset


config = Config()
logger = setup_logger(
    "modeling.evaluate",
    log_file=config.reports_dir / "evaluation.log",
    level=config.log_level,
    format_type=config.log_format
)


def load_model_artifacts(
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None
) -> Dict:
    """
    Carga modelo, scalers y metadata (función legacy para LSTM).
    
    Args:
        model_path: Ruta del modelo (.pt)
        scaler_path: Ruta de scalers (.pkl)
        metadata_path: Ruta de metadata (.json)
    
    Returns:
        Diccionario con artefactos
    """
    if model_path is None:
        model_path = config.models_dir / "lstm_latest.pt"
    
    if scaler_path is None:
        scaler_path = config.models_dir / "scaler.pkl"
    
    if metadata_path is None:
        # Intentar primero con el nuevo nombre
        metadata_path = config.models_dir / "lstm_metadata.json"
        if not metadata_path.exists():
            # Fallback al nombre antiguo
            metadata_path = config.models_dir / "metadata.json"
    
    # Cargar metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Cargar scalers
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    
    # Cargar modelo
    model_arch = metadata["model_architecture"]
    model = LSTMModel(
        input_size=model_arch["input_size"],
        hidden_size=model_arch["hidden_size"],
        num_layers=model_arch["num_layers"],
        dropout=model_arch.get("dropout", 0.2)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    logger.info(f"Modelo cargado desde {model_path}")
    
    return {
        "model": model,
        "scaler_X": scalers["scaler_X"],
        "scaler_y": scalers["scaler_y"],
        "metadata": metadata
    }


def predict_sequences(
    model: LSTMModel,
    X: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """
    Genera predicciones para secuencias.
    
    Args:
        model: Modelo LSTM
        X: Secuencias de entrada (n_samples, lookback, n_features)
        device: Device
    
    Returns:
        Predicciones (n_samples, horizon)
    """
    model.eval()
    dataset = TimeSeriesDataset(X, np.zeros((len(X), 1)))  # y dummy
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas de regresión.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
    
    Returns:
        Diccionario con métricas
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (evitando división por cero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape)
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Calcula métricas de clasificación (evento de lluvia).
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        threshold: Umbral para considerar evento de lluvia
    
    Returns:
        Diccionario con métricas
    """
    # Convertir a clasificación binaria
    y_true_binary = (y_true >= threshold).astype(int).ravel()
    y_pred_binary = (y_pred >= threshold).astype(int).ravel()
    
    # Métricas
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        average="binary",
        zero_division=0
    )
    
    # Matriz de confusión
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    # Curva PR
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true_binary,
            y_pred.ravel()
        )
        pr_auc = auc(recall_curve, precision_curve)
        pr_curve_data = {
            "precision_curve": precision_curve.tolist(),
            "recall_curve": recall_curve.tolist()
        }
    except Exception as e:
        logger.warning(f"Error calculando curva PR: {e}")
        pr_auc = 0.0
        pr_curve_data = {
            "precision_curve": [],
            "recall_curve": []
        }
    
    # Calcular support (número de casos positivos)
    n_positive = int(y_true_binary.sum())
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "support": n_positive,
        "confusion_matrix": cm.tolist(),
        "pr_auc": float(pr_auc),
        **pr_curve_data
    }


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    n_samples: int = 200
) -> Path:
    """
    Genera gráfico de predicciones vs valores reales.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        split_name: Nombre del split (train/val/test)
        n_samples: Número de muestras a graficar
    
    Returns:
        Path del gráfico guardado
    """
    ensure_dir(config.reports_dir)
    
    # Asegurar que son arrays 1D
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Serie temporal (últimas n_samples)
    n_plot = min(n_samples, len(y_true))
    indices = np.arange(n_plot)
    axes[0].plot(indices, y_true[-n_plot:], label="Real", alpha=0.7)
    axes[0].plot(indices, y_pred[-n_plot:], label="Predicción", alpha=0.7)
    axes[0].set_xlabel("Tiempo")
    axes[0].set_ylabel("Precipitación (mm/hr)")
    axes[0].set_title(f"Predicciones vs Reales ({split_name})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Línea perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfecto")
    
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Predicción")
    axes[1].set_title(f"Scatter Plot ({split_name})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = config.reports_dir / f"predictions_{split_name}.png"
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    logger.info(f"Gráfico guardado en {output_path}")
    
    return output_path


def plot_training_history(history: Dict) -> Path:
    """
    Genera gráfico de historia de entrenamiento.
    
    Args:
        history: Diccionario con train_loss y val_loss
    
    Returns:
        Path del gráfico guardado
    """
    ensure_dir(config.reports_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker='o', markersize=3)
    ax.plot(epochs, history["val_loss"], label="Val Loss", marker='o', markersize=3)
    
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Historia de Entrenamiento")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = config.reports_dir / "training_history.png"
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    logger.info(f"Historial de entrenamiento guardado en {output_path}")
    
    return output_path


def plot_pr_curve(pr_data: Dict, split_name: str) -> Path:
    """
    Genera curva Precision-Recall.
    
    Args:
        pr_data: Datos de precision-recall curve
        split_name: Nombre del split
    
    Returns:
        Path del gráfico guardado
    """
    ensure_dir(config.reports_dir)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(
        pr_data["recall_curve"],
        pr_data["precision_curve"],
        label=f'PR AUC = {pr_data["pr_auc"]:.3f}'
    )
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curva Precision-Recall ({split_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = config.reports_dir / f"pr_curve_{split_name}.png"
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    logger.info(f"Curva PR guardada en {output_path}")
    
    return output_path


def evaluate_model(data_dict: Dict, device: str = "cpu", train_ratio: float = 0.7) -> Dict:
    """
    Evalúa el modelo con splits configurables.
    
    Args:
        data_dict: Diccionario con datos
        device: Device para predicciones
        train_ratio: Ratio de datos para entrenamiento (default: 0.7 = 70%)
    
    Returns:
        Diccionario con métricas
    """
    logger.info("="*60)
    logger.info("EVALUACIÓN DEL MODELO LSTM")
    logger.info("="*60)
    
    # Cargar artefactos
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    scaler_y = artifacts["scaler_y"]
    metadata = artifacts["metadata"]
    
    # Umbral para evento de lluvia
    rain_threshold = config.precip_event_mmhr
    
    # Combinar datos de validación y test para re-split 70/30
    X_val = data_dict.get("X_val")
    X_test = data_dict.get("X_test")
    y_val = data_dict.get("y_val")
    y_test = data_dict.get("y_test")
    
    # Concatenar val + test
    X_combined = np.concatenate([X_val, X_test], axis=0)
    y_combined = np.concatenate([y_val, y_test], axis=0)
    
    # Nuevo split 70/30
    split_idx = int(len(X_combined) * train_ratio)
    X_eval_train = X_combined[:split_idx]
    X_eval_test = X_combined[split_idx:]
    y_eval_train_scaled = y_combined[:split_idx]
    y_eval_test_scaled = y_combined[split_idx:]
    
    logger.info(f"Split configurado: {int(train_ratio*100)}% train / {int((1-train_ratio)*100)}% test")
    logger.info(f"  Train samples: {len(X_eval_train)}")
    logger.info(f"  Test samples: {len(X_eval_test)}")
    
    results = {
        "metadata": metadata,
        "rain_threshold": rain_threshold,
        "split_ratio": {"train": train_ratio, "test": 1 - train_ratio},
        "splits": {}
    }
    
    # Evaluar solo en test (70% no se usa, solo para referencia del split)
    logger.info("\nEvaluando en conjunto de TEST (30%)...")
    
    # Predicciones
    y_pred_scaled = predict_sequences(model, X_eval_test, device)
    
    # Reshape si es necesario
    if y_eval_test_scaled.ndim == 3:
        y_eval_test_scaled = y_eval_test_scaled.reshape(-1, y_eval_test_scaled.shape[-1])
    if y_pred_scaled.ndim == 3:
        y_pred_scaled = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
    
    # Des-escalar
    y_true = scaler_y.inverse_transform(y_eval_test_scaled).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_pred = np.maximum(y_pred, 0)
    
    # Estadísticas descriptivas
    logger.info("\nEstadísticas descriptivas:")
    logger.info(f"  Real    - Media: {y_true.mean():.4f}, Std: {y_true.std():.4f}, Max: {y_true.max():.4f}")
    logger.info(f"  Predicho - Media: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}, Max: {y_pred.max():.4f}")
    
    # Métricas de regresión
    regression_metrics = calculate_regression_metrics(y_true, y_pred)
    logger.info("\nMétricas de regresión:")
    logger.info(f"  MAE:  {regression_metrics['MAE']:.4f} mm/hr")
    logger.info(f"  RMSE: {regression_metrics['RMSE']:.4f} mm/hr")
    logger.info(f"  R²:   {regression_metrics['R2']:.4f}")
    
    # Métricas de clasificación
    classification_metrics = calculate_classification_metrics(
        y_true, y_pred, threshold=rain_threshold
    )
    
    # Calcular métricas adicionales
    y_true_binary = (y_true >= rain_threshold).astype(int)
    y_pred_binary = (y_pred >= rain_threshold).astype(int)
    
    true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    true_negatives = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    logger.info(f"\nMétricas de clasificación (umbral: {rain_threshold} mm/hr):")
    logger.info(f"  Precision: {classification_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {classification_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {classification_metrics['f1_score']:.4f}")
    logger.info(f"\nMatriz de confusión:")
    logger.info(f"  TP: {true_positives}, TN: {true_negatives}")
    logger.info(f"  FP: {false_positives}, FN: {false_negatives}")
    logger.info(f"  Eventos reales: {np.sum(y_true_binary)} ({np.sum(y_true_binary)/len(y_true_binary)*100:.1f}%)")
    logger.info(f"  Eventos predichos: {np.sum(y_pred_binary)} ({np.sum(y_pred_binary)/len(y_pred_binary)*100:.1f}%)")
    
    # Guardar resultados
    results["splits"]["test"] = {
        "regression": regression_metrics,
        "classification": {
            "precision": classification_metrics['precision'],
            "recall": classification_metrics['recall'],
            "f1_score": classification_metrics['f1_score'],
            "support": classification_metrics['support'],
            "confusion_matrix": classification_metrics['confusion_matrix']
        },
        "statistics": {
            "y_true_mean": float(y_true.mean()),
            "y_true_std": float(y_true.std()),
            "y_pred_mean": float(y_pred.mean()),
            "y_pred_std": float(y_pred.std()),
            "n_samples": len(y_true)
        }
    }
    
    # Guardar resultados en JSON
    results_path = config.reports_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResultados guardados en {results_path}")
    logger.info("="*60)
    
    return results


