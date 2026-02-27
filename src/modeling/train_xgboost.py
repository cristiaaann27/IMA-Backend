"""
Entrenamiento de modelo XGBoost para predicción de precipitación.
Complementa al modelo LSTM existente.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
    fbeta_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import Config, setup_logger, ensure_dir
from .base import BaseModel, DataPreparator


config = Config()
logger = setup_logger(
    'modeling.train_xgboost',
    log_file=config.reports_dir / 'training_xgboost.log',
    level=config.log_level,
    format_type=config.log_format
)


def load_curated_data(curated_path: Path) -> pd.DataFrame:
    """Carga datos curados."""
    logger.info(f"Cargando datos desde {curated_path}")
    df = pd.read_parquet(curated_path)
    logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    return df


def prepare_data(df: pd.DataFrame, target_col: str = 'precip_mm_hr') -> Tuple:
    """
    Prepara datos para XGBoost.
    
    Args:
        df: DataFrame con features
        target_col: Columna objetivo
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Preparando datos para XGBoost...")
    
    # Eliminar timestamp y target
    feature_cols = [col for col in df.columns if col not in ['timestamp', target_col]]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, shuffle=False  # 0.176 * 0.85 ≈ 0.15
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict = None
) -> xgb.XGBRegressor:
    """
    Entrena modelo XGBoost.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        params: Hiperparámetros personalizados
        
    Returns:
        Modelo entrenado
    """
    logger.info("Entrenando modelo XGBoost...")
    
    # Parámetros por defecto
    default_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1
    }
    
    if params:
        default_params.update(params)
    
    # Crear modelo
    model = xgb.XGBRegressor(**default_params)
    
    # Entrenar con early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10,
        early_stopping_rounds=10
    )
    
    logger.info("✓ Modelo XGBoost entrenado")
    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Evalúa modelo XGBoost.
    
    Args:
        model: Modelo entrenado
        X_test, y_test: Datos de test
        threshold: Umbral para clasificación binaria
        
    Returns:
        Diccionario con métricas
    """
    logger.info("Evaluando modelo XGBoost...")
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas de regresión
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    # Clasificación binaria (evento de lluvia)
    y_test_binary = (y_test >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_binary, y_pred_binary, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_test_binary, y_pred)
    except:
        auc = None
    
    # F2-score (beta=2, pondera recall más que precision)
    f2 = fbeta_score(y_test_binary, y_pred_binary, beta=2, zero_division=0)
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'f2_score': float(f2),
        'auc': float(auc) if auc else None
    }
    
    logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, F1: {f1:.4f}, F2: {f2:.4f}")
    
    return metrics


def plot_feature_importance(model: xgb.XGBRegressor, feature_names: list, output_path: Path):
    """Grafica importancia de features."""
    logger.info("Generando gráfico de feature importance...")
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[-20:]  # Top 20
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title('Top 20 Features - XGBoost')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"✓ Gráfico guardado en {output_path}")


class XGBoostModelWrapper(BaseModel):
    """Wrapper para modelo XGBoost que implementa la interfaz BaseModel."""
    
    def __init__(self):
        super().__init__("xgboost")
        self.feature_names = []
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict = None
    ) -> Dict:
        """Entrena el modelo XGBoost."""
        self.model = train_xgboost_model(X_train, y_train, X_val, y_val, params)
        self.is_trained = True
        return {"model": self.model}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predicciones."""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        return self.model.predict(X)
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Guarda modelo y metadata."""
        ensure_dir(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar modelo
        model_path = output_dir / f'xgboost_{timestamp}.json'
        self.model.save_model(str(model_path))
        
        # Guardar también como latest
        latest_path = output_dir / 'xgboost_latest.json'
        self.model.save_model(str(latest_path))
        
        # Guardar metadata usando método de la clase base
        metadata_path = self._save_metadata(output_dir, {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'hyperparameters': self.model.get_params()
        })
        
        logger.info(f"✓ Modelo guardado en {model_path}")
        
        return {
            "model": model_path,
            "metadata": metadata_path
        }
    
    def load(self, model_path: Path, metadata_path: Optional[Path] = None):
        """Carga modelo y metadata."""
        # Cargar modelo
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        
        # Cargar metadata
        if metadata_path is None:
            metadata_path = model_path.parent / 'xgboost_metadata.json'
        
        if metadata_path.exists():
            metadata = self._load_metadata(metadata_path)
            self.feature_names = metadata.get('feature_names', [])
        
        self.is_trained = True
        logger.info(f"Modelo XGBoost cargado desde {model_path}")


def save_model(model: xgb.XGBRegressor, metrics: Dict, feature_names: list, output_dir: Path):
    """Guarda modelo y metadata (función legacy)."""
    ensure_dir(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar modelo
    model_path = output_dir / f'xgboost_{timestamp}.json'
    model.save_model(str(model_path))
    
    # Guardar también como latest
    latest_path = output_dir / 'xgboost_latest.json'
    model.save_model(str(latest_path))
    
    # Guardar metadata
    metadata = {
        'timestamp': timestamp,
        'model_name': 'xgboost',
        'model_type': 'xgboost',
        'metrics': metrics,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'hyperparameters': model.get_params()
    }
    
    metadata_path = output_dir / 'xgboost_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Modelo guardado en {model_path}")
    logger.info(f"✓ Metadata guardado en {metadata_path}")


def main():
    """Pipeline completo de entrenamiento XGBoost."""
    logger.info("="*60)
    logger.info("ENTRENAMIENTO DE MODELO XGBOOST")
    logger.info("="*60)
    
    # Paths
    curated_path = Path('data/curated/curated_latest.parquet')
    models_dir = Path('models')
    reports_dir = Path('reports')
    
    # 1. Cargar datos
    df = load_curated_data(curated_path)
    
    # 2. Preparar datos
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(df)
    
    # 3. Entrenar modelo
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # 4. Evaluar
    metrics = evaluate_model(model, X_test, y_test)
    
    # 5. Feature importance
    plot_feature_importance(model, feature_names, reports_dir / 'xgboost_feature_importance.png')
    
    # 6. Guardar modelo
    save_model(model, metrics, feature_names, models_dir)
    
    logger.info("="*60)
    logger.info("✓ ENTRENAMIENTO COMPLETADO")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
