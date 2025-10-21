"""Entrenamiento de modelo LSTM para predicción de precipitación."""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from ..utils import Config, setup_logger, ensure_dir
from .base import BaseModel, DataPreparator


config = Config()
logger = setup_logger(
    "modeling.train_lstm",
    log_file=config.reports_dir / "training_lstm.log",
    level=config.log_level,
    format_type=config.log_format
)


class TimeSeriesDataset(Dataset):
    """Dataset para series temporales con ventana lookback."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        # Asegurar que y tenga la forma correcta (n_samples, 1)
        if y.ndim == 1:
            self.y = torch.FloatTensor(y).unsqueeze(1)
        elif y.ndim == 2 and y.shape[1] == 1:
            # Ya tiene la forma correcta (n_samples, 1)
            self.y = torch.FloatTensor(y)
        else:
            # Para otros casos, usar reshape
            self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """Modelo LSTM para predicción de series temporales."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Tomar última salida temporal
        last_out = lstm_out[:, -1, :]
        
        # Dropout y capa fully connected
        out = self.dropout(last_out)
        out = self.fc(out)
        
        return out


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 24,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea secuencias de ventana deslizante para LSTM.
    
    Args:
        X: Features (n_samples, n_features)
        y: Target (n_samples,)
        lookback: Ventana de observación (pasos hacia atrás)
        horizon: Horizonte de predicción (pasos hacia adelante)
    
    Returns:
        X_seq: (n_sequences, lookback, n_features)
        y_seq: (n_sequences, horizon)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - lookback - horizon + 1):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y[i + lookback:i + lookback + horizon])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Asegurar que y_seq tenga la forma correcta
    if y_seq.ndim == 1:
        y_seq = y_seq.reshape(-1, 1)
    
    return X_seq, y_seq


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "precip_mm_hr",
    feature_cols: Optional[list] = None,
    lookback: int = 24,
    horizon: int = 1,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Dict:
    """
    Prepara datos para entrenamiento de LSTM.
    
    Args:
        df: DataFrame con features
        target_col: Columna objetivo
        feature_cols: Columnas de features (si None, usa todas excepto timestamp y target)
        lookback: Ventana de observación
        horizon: Horizonte de predicción
        train_split: Proporción de train
        val_split: Proporción de validación
        test_split: Proporción de test
    
    Returns:
        Diccionario con splits y scaler
    """
    logger.info(f"Preparando datos para LSTM (lookback={lookback}, horizon={horizon})")
    
    # Seleccionar features
    if feature_cols is None:
        # Excluir timestamp y target, usar el resto
        exclude_cols = ["timestamp", target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Features seleccionados: {len(feature_cols)} columnas")
    logger.debug(f"Features: {feature_cols}")
    
    # Extraer X e y
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # Eliminar filas con NaN
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).ravel())
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Datos válidos: {len(X)} registros")
    
    # Split temporal (train/val/test)
    n = len(X)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Split temporal: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Escalado
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)
    
    # Crear secuencias
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback, horizon)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, lookback, horizon)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback, horizon)
    
    logger.info(
        f"Secuencias creadas: train={len(X_train_seq)}, "
        f"val={len(X_val_seq)}, test={len(X_test_seq)}"
    )
    
    return {
        "X_train": X_train_seq,
        "y_train": y_train_seq,
        "X_val": X_val_seq,
        "y_val": y_val_seq,
        "X_test": X_test_seq,
        "y_test": y_test_seq,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "n_features": X_train_seq.shape[2]
    }


def train_model(
    data_dict: Dict,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 64,
    early_stopping_patience: int = 10,
    device: str = None
) -> Tuple[LSTMModel, Dict]:
    """
    Entrena modelo LSTM.
    
    Args:
        data_dict: Diccionario con datos preparados
        hidden_size: Tamaño de capa oculta
        num_layers: Número de capas LSTM
        dropout: Tasa de dropout
        learning_rate: Learning rate
        epochs: Número de épocas
        batch_size: Tamaño de batch
        early_stopping_patience: Paciencia para early stopping
        device: Device ('cuda' o 'cpu')
    
    Returns:
        Modelo entrenado y diccionario con historia
    """
    logger.info("="*60)
    logger.info("INICIANDO ENTRENAMIENTO LSTM")
    logger.info("="*60)
    
    # Determinar device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Usando device: {device}")
    
    # Crear datasets y dataloaders
    train_dataset = TimeSeriesDataset(data_dict["X_train"], data_dict["y_train"])
    val_dataset = TimeSeriesDataset(data_dict["X_val"], data_dict["y_val"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    input_size = data_dict["n_features"]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    logger.info(f"Modelo creado: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
    logger.info(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters())}")
    
    # Loss y optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Historia de entrenamiento
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf")
    }
    
    # Early stopping
    patience_counter = 0
    best_model_state = None
    
    # Entrenamiento
    for epoch in range(epochs):
        # Fase de entrenamiento
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Fase de validación
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Guardar historia
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        # Log cada 5 épocas
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )
        
        # Early stopping
        if avg_val_loss < history["best_val_loss"]:
            history["best_val_loss"] = avg_val_loss
            history["best_epoch"] = epoch + 1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping en época {epoch + 1}")
            break
    
    # Restaurar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Mejor modelo de época {history['best_epoch']} restaurado")
    
    logger.info("="*60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info(f"Mejor Val Loss: {history['best_val_loss']:.6f} (época {history['best_epoch']})")
    logger.info("="*60)
    
    return model, history


class LSTMModelWrapper(BaseModel):
    """Wrapper para modelo LSTM que implementa la interfaz BaseModel."""
    
    def __init__(self):
        super().__init__("lstm")
        self.scaler_X = None
        self.scaler_y = None
        self.lookback = 24
        self.horizon = 1
        self.feature_cols = []
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        device: str = None
    ) -> Dict:
        """Entrena el modelo LSTM."""
        # Llamar a la función original de entrenamiento
        self.model, history = train_model(
            {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
             "n_features": X_train.shape[2]},
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            device=device
        )
        self.is_trained = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predicciones."""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        
        dataset = TimeSeriesDataset(X, np.zeros((len(X), 1)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                y_pred = self.model(X_batch)
                predictions.append(y_pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Guarda modelo, scalers y metadata."""
        ensure_dir(output_dir)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar modelo PyTorch
        model_path = output_dir / f"lstm_{timestamp_str}.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # También guardar como "latest"
        latest_model_path = output_dir / "lstm_latest.pt"
        torch.save(self.model.state_dict(), latest_model_path)
        
        # Guardar scalers si existen
        if self.scaler_X is not None and self.scaler_y is not None:
            scaler_path = output_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump({
                    "scaler_X": self.scaler_X,
                    "scaler_y": self.scaler_y
                }, f)
            logger.info(f"Scalers guardados en {scaler_path}")
        
        # Guardar metadata usando método de la clase base
        metadata_path = self._save_metadata(output_dir, {
            "model_architecture": {
                "input_size": self.model.lstm.input_size if hasattr(self.model, 'lstm') else None,
                "hidden_size": self.model.hidden_size if hasattr(self.model, 'hidden_size') else None,
                "num_layers": self.model.num_layers if hasattr(self.model, 'num_layers') else None
            },
            "data": {
                "feature_cols": self.feature_cols,
                "lookback": self.lookback,
                "horizon": self.horizon
            }
        })
        
        logger.info(f"Modelo guardado en {model_path}")
        
        return {
            "model": model_path,
            "scaler": scaler_path if self.scaler_X is not None else None,
            "metadata": metadata_path
        }
    
    def load(self, model_path: Path, metadata_path: Optional[Path] = None):
        """Carga modelo y metadata."""
        # Cargar metadata
        if metadata_path is None:
            metadata_path = model_path.parent / "lstm_metadata.json"
        
        if metadata_path.exists():
            metadata = self._load_metadata(metadata_path)
            arch = metadata.get("model_architecture", {})
            
            # Crear modelo con arquitectura guardada
            self.model = LSTMModel(
                input_size=arch.get("input_size", 10),
                hidden_size=arch.get("hidden_size", 64),
                num_layers=arch.get("num_layers", 2)
            )
            
            # Cargar pesos
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            
            # Cargar configuración de datos
            data_config = metadata.get("data", {})
            self.feature_cols = data_config.get("feature_cols", [])
            self.lookback = data_config.get("lookback", 24)
            self.horizon = data_config.get("horizon", 1)
            
            self.is_trained = True
            logger.info(f"Modelo LSTM cargado desde {model_path}")
        else:
            logger.warning(f"Metadata no encontrada en {metadata_path}")


def save_model(
    model: LSTMModel,
    data_dict: Dict,
    history: Dict,
    hyperparams: Dict
) -> Dict[str, Path]:
    """
    Guarda modelo, scalers y metadata (función legacy).
    
    Args:
        model: Modelo entrenado
        data_dict: Diccionario con datos y scalers
        history: Historia de entrenamiento
        hyperparams: Hiperparámetros
    
    Returns:
        Diccionario con rutas de archivos guardados
    """
    ensure_dir(config.models_dir)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar modelo PyTorch
    model_path = config.models_dir / f"lstm_{timestamp_str}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Modelo guardado en {model_path}")
    
    # También guardar como "latest"
    latest_model_path = config.models_dir / "lstm_latest.pt"
    torch.save(model.state_dict(), latest_model_path)
    
    # Guardar scalers
    scaler_path = config.models_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "scaler_X": data_dict["scaler_X"],
            "scaler_y": data_dict["scaler_y"]
        }, f)
    logger.info(f"Scalers guardados en {scaler_path}")
    
    # Guardar metadata
    metadata = {
        "timestamp": timestamp_str,
        "model_name": "lstm",
        "model_architecture": {
            "input_size": data_dict["n_features"],
            "hidden_size": hyperparams["hidden_size"],
            "num_layers": hyperparams["num_layers"],
            "dropout": hyperparams["dropout"]
        },
        "training": {
            "epochs": len(history["train_loss"]),
            "best_epoch": history["best_epoch"],
            "best_val_loss": float(history["best_val_loss"]),
            "final_train_loss": float(history["train_loss"][-1]),
            "learning_rate": hyperparams["learning_rate"],
            "batch_size": hyperparams["batch_size"]
        },
        "data": {
            "n_features": data_dict["n_features"],
            "feature_cols": data_dict["feature_cols"],
            "lookback": hyperparams.get("lookback", 24),
            "horizon": hyperparams.get("horizon", 1),
            "train_samples": len(data_dict["X_train"]),
            "val_samples": len(data_dict["X_val"]),
            "test_samples": len(data_dict["X_test"])
        },
        "history": {
            "train_loss": [float(x) for x in history["train_loss"]],
            "val_loss": [float(x) for x in history["val_loss"]]
        }
    }
    
    metadata_path = config.models_dir / "lstm_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata guardada en {metadata_path}")
    
    return {
        "model": model_path,
        "scaler": scaler_path,
        "metadata": metadata_path
    }


