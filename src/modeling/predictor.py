"""Módulo unificado de predicción para todos los modelos."""

from pathlib import Path
from typing import Optional, Union, Dict
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from ..utils import Config, setup_logger, load_dataframe, save_dataframe, ensure_dir


config = Config()
logger = setup_logger(
    "modeling.predictor",
    log_file=config.reports_dir / "predictions.log",
    level=config.log_level,
    format_type=config.log_format
)


class ModelType(Enum):
    """Tipos de modelos disponibles."""
    LSTM = "lstm"
    XGBOOST = "xgboost"
    HYBRID = "hybrid"


class UnifiedPredictor:
    """Predictor unificado que maneja múltiples tipos de modelos."""
    
    def __init__(self, model_type: Union[ModelType, str]):
        """
        Inicializa el predictor.
        
        Args:
            model_type: Tipo de modelo ('lstm' o 'xgboost')
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        self.model_type = model_type
        self.model_wrapper = None
        
    def load_model(self, model_dir: Optional[Path] = None):
        """
        Carga el modelo especificado.
        
        Args:
            model_dir: Directorio donde está el modelo (si None, usa config.models_dir)
        """
        if model_dir is None:
            model_dir = config.models_dir
        
        if self.model_type == ModelType.LSTM:
            from .train_lstm import LSTMModelWrapper
            self.model_wrapper = LSTMModelWrapper()
            model_path = model_dir / "lstm_latest.pt"
            self.model_wrapper.load(model_path)
            
        elif self.model_type == ModelType.XGBOOST:
            from .train_xgboost import XGBoostModelWrapper
            self.model_wrapper = XGBoostModelWrapper()
            model_path = model_dir / "xgboost_latest.json"
            self.model_wrapper.load(model_path)
        
        logger.info(f"Modelo {self.model_type.value} cargado exitosamente")
    
    def predict(
        self,
        input_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Genera predicciones usando el modelo cargado.
        
        Args:
            input_path: Ruta de datos curados (si None, usa latest)
            output_dir: Directorio de salida (si None, usa config.predictions_dir)
            save_results: Si True, guarda las predicciones
            
        Returns:
            DataFrame con predicciones
        """
        logger.info("="*60)
        logger.info(f"GENERANDO PREDICCIONES CON {self.model_type.value.upper()}")
        logger.info("="*60)
        
        if self.model_wrapper is None:
            raise ValueError("Debe cargar un modelo primero usando load_model()")
        
        # Cargar datos curados
        if input_path is None:
            input_path = config.curated_data_dir / "curated_latest.parquet"
        
        if not input_path.exists():
            raise FileNotFoundError(f"No existe {input_path}")
        
        logger.info(f"Cargando datos desde {input_path}")
        df = load_dataframe(input_path)
        
        # Preparar features según el tipo de modelo
        if self.model_type == ModelType.LSTM:
            pred_df = self._predict_lstm(df)
        elif self.model_type == ModelType.XGBOOST:
            pred_df = self._predict_xgboost(df)
        
        # Guardar predicciones si se solicita
        if save_results:
            if output_dir is None:
                output_dir = config.predictions_dir
            
            ensure_dir(output_dir)
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            pred_path = output_dir / f"pred_{self.model_type.value}_{timestamp_str}.parquet"
            save_dataframe(pred_df, pred_path, format="parquet")
            logger.info(f"Predicciones guardadas en {pred_path}")
            
            # Guardar también como CSV para fácil consumo
            csv_path = config.reports_dir / f"last_prediction_{self.model_type.value}.csv"
            save_dataframe(pred_df, csv_path, format="csv")
            logger.info(f"Predicciones exportadas a {csv_path}")
        
        # Estadísticas
        self._log_prediction_stats(pred_df)
        
        logger.info("="*60)
        logger.info("PREDICCIONES COMPLETADAS")
        logger.info("="*60)
        
        return pred_df
    
    def _predict_lstm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera predicciones usando modelo LSTM."""
        import pickle
        
        feature_cols = self.model_wrapper.feature_cols
        lookback = self.model_wrapper.lookback
        
        # Verificar features
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")
        
        # Cargar scalers
        scaler_path = config.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scalers no encontrados: {scaler_path}")
        
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        
        scaler_X = scalers["scaler_X"]
        scaler_y = scalers["scaler_y"]
        
        # Extraer timestamp
        timestamps = df["timestamp"].values if "timestamp" in df.columns else None
        
        # Preparar features
        X = df[feature_cols].values
        
        # Manejar NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        
        if timestamps is not None:
            timestamps_valid = timestamps[valid_mask]
        
        logger.info(f"Datos válidos: {len(X_valid)} de {len(X)} registros")
        
        # Escalar
        X_scaled = scaler_X.transform(X_valid)
        
        # Crear secuencias
        X_sequences = []
        sequence_indices = []
        
        for i in range(len(X_scaled) - lookback + 1):
            X_sequences.append(X_scaled[i:i + lookback])
            sequence_indices.append(i + lookback - 1)
        
        X_sequences = np.array(X_sequences)
        logger.info(f"Secuencias creadas: {len(X_sequences)}")
        
        # Predecir
        y_pred_scaled = self.model_wrapper.predict(X_sequences)
        
        # Des-escalar
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_pred = np.maximum(y_pred, 0)  # No negativos
        
        # Crear DataFrame
        pred_df = pd.DataFrame({
            "precip_pred_mm_hr": y_pred.ravel()
        })
        
        # Añadir timestamp
        if timestamps is not None:
            pred_df["timestamp"] = timestamps_valid[sequence_indices]
        
        # Añadir valor real si existe
        if "precip_mm_hr" in df.columns:
            real_values = df.loc[valid_mask, "precip_mm_hr"].values
            pred_df["precip_real_mm_hr"] = real_values[sequence_indices]
            pred_df["error"] = pred_df["precip_real_mm_hr"] - pred_df["precip_pred_mm_hr"]
            pred_df["abs_error"] = np.abs(pred_df["error"])
        
        return pred_df
    
    def _predict_xgboost(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera predicciones usando modelo XGBoost."""
        feature_cols = self.model_wrapper.feature_names
        
        # Verificar features
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")
        
        # Extraer timestamp
        timestamps = df["timestamp"].values if "timestamp" in df.columns else None
        
        # Preparar features
        X = df[feature_cols].values
        
        # Manejar NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        
        if timestamps is not None:
            timestamps_valid = timestamps[valid_mask]
        
        logger.info(f"Datos válidos: {len(X_valid)} de {len(X)} registros")
        
        # Predecir
        y_pred = self.model_wrapper.predict(X_valid)
        y_pred = np.maximum(y_pred, 0)  # No negativos
        
        # Crear DataFrame
        pred_df = pd.DataFrame({
            "precip_pred_mm_hr": y_pred
        })
        
        # Añadir timestamp
        if timestamps is not None:
            pred_df["timestamp"] = timestamps_valid
        
        # Añadir valor real si existe
        if "precip_mm_hr" in df.columns:
            pred_df["precip_real_mm_hr"] = df["precip_mm_hr"].values[valid_mask]
            pred_df["error"] = pred_df["precip_real_mm_hr"] - pred_df["precip_pred_mm_hr"]
            pred_df["abs_error"] = np.abs(pred_df["error"])
        
        # Añadir clasificación binaria
        threshold = config.precip_event_mmhr
        pred_df["rain_event_pred"] = (y_pred >= threshold).astype(int)
        
        return pred_df
    
    def _log_prediction_stats(self, pred_df: pd.DataFrame):
        """Registra estadísticas de las predicciones."""
        logger.info(f"Predicciones generadas: {len(pred_df)}")
        logger.info(f"Precipitación predicha media: {pred_df['precip_pred_mm_hr'].mean():.4f} mm/hr")
        logger.info(f"Precipitación predicha máxima: {pred_df['precip_pred_mm_hr'].max():.4f} mm/hr")
        
        rain_events = (pred_df["precip_pred_mm_hr"] >= config.precip_event_mmhr).sum()
        rain_pct = (rain_events / len(pred_df)) * 100
        logger.info(f"Eventos de lluvia predichos: {rain_events} ({rain_pct:.2f}%)")
        
        # Si hay valores reales, calcular métricas
        if "precip_real_mm_hr" in pred_df.columns:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(pred_df["precip_real_mm_hr"], pred_df["precip_pred_mm_hr"])
            rmse = np.sqrt(mean_squared_error(pred_df["precip_real_mm_hr"], pred_df["precip_pred_mm_hr"]))
            r2 = r2_score(pred_df["precip_real_mm_hr"], pred_df["precip_pred_mm_hr"])
            
            logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")


def predict(
    model_type: str = "lstm",
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Función de conveniencia para generar predicciones.
    
    Args:
        model_type: Tipo de modelo ('lstm' o 'xgboost')
        input_path: Ruta de datos curados
        output_dir: Directorio de salida
        
    Returns:
        DataFrame con predicciones
    """
    predictor = UnifiedPredictor(model_type)
    predictor.load_model()
    return predictor.predict(input_path, output_dir)


class HybridPredictor:
    """
    Predictor híbrido que combina LSTM y XGBoost mediante promedio pesado.

    Pesos configurables desde configs/hyperparameters.json (sección 'ensemble').
    """

    def __init__(self, w_lstm: float = 0.5, w_xgboost: float = 0.5):
        """
        Args:
            w_lstm: Peso para las predicciones del LSTM.
            w_xgboost: Peso para las predicciones del XGBoost.
        """
        self.w_lstm = w_lstm
        self.w_xgboost = w_xgboost
        self.lstm_predictor = UnifiedPredictor(ModelType.LSTM)
        self.xgb_predictor = UnifiedPredictor(ModelType.XGBOOST)
        self._loaded = False

    def load_models(self, model_dir: Optional[Path] = None):
        """Carga ambos modelos."""
        self.lstm_predictor.load_model(model_dir)
        self.xgb_predictor.load_model(model_dir)
        self._loaded = True
        logger.info(
            f"HybridPredictor cargado (w_lstm={self.w_lstm}, w_xgb={self.w_xgboost})"
        )

    def predict_single(
        self,
        features_lstm: np.ndarray,
        features_xgb: np.ndarray,
    ) -> float:
        """
        Predicción híbrida para una sola muestra.

        Args:
            features_lstm: Secuencia LSTM (lookback, n_features).
            features_xgb: Fila XGBoost (n_features,).

        Returns:
            Predicción combinada (mm/hr), clipeada a >= 0.
        """
        if not self._loaded:
            raise ValueError("Modelos no cargados. Llama a load_models() primero.")

        pred_lstm = self.lstm_predictor.model_wrapper.predict(
            features_lstm[np.newaxis, ...]
        ).ravel()[0]

        pred_xgb = self.xgb_predictor.model_wrapper.predict(
            features_xgb.reshape(1, -1)
        ).ravel()[0]

        combined = self.w_lstm * pred_lstm + self.w_xgboost * pred_xgb
        combined = max(0.0, float(combined))

        logger.debug(
            f"Hybrid: lstm={pred_lstm:.4f} xgb={pred_xgb:.4f} -> {combined:.4f}"
        )
        return combined
