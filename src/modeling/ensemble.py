from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from ..utils import Config, ensure_dir, load_dataframe, setup_logger
from .base import BaseModel


config = Config()
logger = setup_logger(
    "modeling.ensemble",
    log_file=config.reports_dir / "ensemble.log",
    level=config.log_level,
    format_type=config.log_format,
)


class StackedEnsemble(BaseModel):
    """
    Meta-learner entrenado sobre las predicciones de LSTM y XGBoost.
    Hereda BaseModel ---> implementa train / predict / save / load.

    Flujo:
    1. Carga LSTM (lstm_latest.pt + scaler.pkl) y XGBoost (xgboost_latest.json).
    2. Genera predicciones alineadas en el split de validación.
    3. Construye X_meta = [[lstm_pred_i, xgb_pred_i], ...].
    4. Ajusta RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10]) sobre X_meta → y_true.
    5. Guarda ensemble_meta.pkl + ensemble_report.json.

    Alineación temporal
    -------------------
    El LSTM produce N_val - lookback - horizon + 1 secuencias (empieza en t=24).
    El XGBoost opera sobre la muestra plana en ese mismo instante.
    Se recorta el array plano: X_xgb_aligned = X_val_raw[lookback : lookback + N_seq].
    """

    def __init__(self):
        super().__init__("ensemble")
        self.meta_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
        self.lstm_wrapper: Optional[object] = None
        self.xgb_wrapper: Optional[object] = None
        self.lookback: int = 24

    def _load_base_models(self, model_dir: Path) -> None:

        from .train_lstm import LSTMModelWrapper
        from .train_xgboost import XGBoostModelWrapper

        # --- LSTM ---
        self.lstm_wrapper = LSTMModelWrapper()
        self.lstm_wrapper.load(model_dir / "lstm_latest.pt")

        scaler_path = model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scalers del LSTM no encontrados: {scaler_path}"
            )
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        self.lstm_wrapper.scaler_X = scalers["scaler_X"]
        self.lstm_wrapper.scaler_y = scalers["scaler_y"]

        self.lookback = self.lstm_wrapper.lookback

        # XGBoost
        self.xgb_wrapper = XGBoostModelWrapper()
        self.xgb_wrapper.load(model_dir / "xgboost_latest.json")

        logger.info(
            f"Modelos base cargados — lookback={self.lookback} | "
            f"LSTM features={len(self.lstm_wrapper.feature_cols)} | "
            f"XGB features={len(self.xgb_wrapper.feature_names)}"
        )

    def _get_meta_features(
        self,
        df: pd.DataFrame,
        train_split: float = 0.7,
        val_split: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        from .train_lstm import prepare_data as lstm_prepare_data

        target_col = "precip_mm_hr"
        feature_cols = self.lstm_wrapper.feature_cols

        # ── 1. Preparar datos LSTM (escala + secuencias) ───────────────
        lstm_data = lstm_prepare_data(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            lookback=self.lookback,
            train_split=train_split,
            val_split=val_split,
        )
        scaler_y = lstm_data["scaler_y"]

        # ── 2. Predicciones LSTM en validación → escala original ───────
        y_pred_lstm_scaled = self.lstm_wrapper.predict(lstm_data["X_val"])

        # Normalizar forma: quedarse con el primer paso del horizonte
        if y_pred_lstm_scaled.ndim == 1:
            y_pred_lstm_scaled = y_pred_lstm_scaled.reshape(-1, 1)
        elif y_pred_lstm_scaled.shape[1] > 1:
            y_pred_lstm_scaled = y_pred_lstm_scaled[:, 0:1]

        y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled).ravel()
        y_pred_lstm = np.maximum(y_pred_lstm, 0.0)

        # ── 3. Split raw (sin escalar) para XGBoost ────────────────────
        valid_mask = ~(
            np.isnan(df[feature_cols].values).any(axis=1)
            | np.isnan(df[target_col].values)
        )
        X_raw = df[feature_cols].values[valid_mask]
        y_raw = df[target_col].values[valid_mask]

        n = len(X_raw)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))

        X_val_raw = X_raw[train_end:val_end]
        y_val_raw = y_raw[train_end:val_end]

        # ── 4. Alinear XGBoost con las secuencias del LSTM ─────────────
        # Secuencia LSTM i predice el instante (i + lookback) del raw val.
        n_seq = len(y_pred_lstm)
        X_xgb_aligned = X_val_raw[self.lookback : self.lookback + n_seq]
        y_true = y_val_raw[self.lookback : self.lookback + n_seq]

        y_pred_xgb = self.xgb_wrapper.predict(X_xgb_aligned)
        y_pred_xgb = np.maximum(y_pred_xgb, 0.0)

        logger.info(
            f"Meta-features generadas: {n_seq} muestras alineadas "
            f"(val_raw={len(y_val_raw)}, lookback={self.lookback})"
        )

        X_meta = np.column_stack([y_pred_lstm, y_pred_xgb])
        return X_meta, y_true, y_pred_lstm, y_pred_xgb

    # ------------------------------------------------------------------
    # Interfaz BaseModel
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs,
    ) -> Dict:
        self.meta_model.fit(X_train, y_train)
        self.is_trained = True
        return {
            "alpha": float(self.meta_model.alpha_),
            "coef_lstm": float(self.meta_model.coef_[0]),
            "coef_xgb": float(self.meta_model.coef_[1]),
            "intercept": float(self.meta_model.intercept_),
        }

    def train_from_df(
        self,
        df: pd.DataFrame,
        model_dir: Optional[Path] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
    ) -> Dict:
        if model_dir is None:
            model_dir = config.models_dir

        self._load_base_models(model_dir)

        X_meta, y_true, y_pred_lstm, y_pred_xgb = self._get_meta_features(
            df, train_split=train_split, val_split=val_split
        )

        fit_info = self.train(X_meta, y_true, X_meta, y_true)
        logger.info(
            f"RidgeCV ajustado — alpha={fit_info['alpha']:.4f} | "
            f"coef=[lstm={fit_info['coef_lstm']:.4f}, "
            f"xgb={fit_info['coef_xgb']:.4f}] | "
            f"intercept={fit_info['intercept']:.4f}"
        )

        y_pred_ensemble = np.maximum(self.meta_model.predict(X_meta), 0.0)
        metrics = self._compute_metrics(y_true, y_pred_lstm, y_pred_xgb, y_pred_ensemble)
        metrics["meta_model"] = fit_info

        return metrics

    def predict(self, X_lstm_seq: np.ndarray, X_xgb_flat: np.ndarray) -> np.ndarray:
        """
        Predicción combinada.
        Parameters:
        X_lstm_seq : (N, lookback, n_features) — secuencias LSTM escaladas
        X_xgb_flat : (N, n_features)           — features sin escalar

        Returns:
        np.ndarray de forma (N,) con precipitación predicha (mm/hr), >= 0.
        """

        y_scaled = self.lstm_wrapper.predict(X_lstm_seq)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        elif y_scaled.shape[1] > 1:
            y_scaled = y_scaled[:, 0:1]

        y_pred_lstm = self.lstm_wrapper.scaler_y.inverse_transform(y_scaled).ravel()
        y_pred_lstm = np.maximum(y_pred_lstm, 0.0)

        y_pred_xgb = np.maximum(self.xgb_wrapper.predict(X_xgb_flat), 0.0)

        X_meta = np.column_stack([y_pred_lstm, y_pred_xgb])
        return np.maximum(self.meta_model.predict(X_meta), 0.0)

    def save(self, output_dir: Path) -> Dict[str, Path]:
        ensure_dir(output_dir)

        model_path = output_dir / "ensemble_meta.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.meta_model, f)

        additional: Dict = {"lookback": self.lookback}
        if self.is_trained:
            additional.update(
                {
                    "alpha": float(self.meta_model.alpha_),
                    "coef_lstm": float(self.meta_model.coef_[0]),
                    "coef_xgb": float(self.meta_model.coef_[1]),
                    "intercept": float(self.meta_model.intercept_),
                }
            )

        metadata_path = self._save_metadata(output_dir, additional)
        logger.info(f"modelo guardado en {model_path}")

        return {"model": model_path, "metadata": metadata_path}

    def load(self, model_path: Path, metadata_path: Optional[Path] = None):
        with open(model_path, "rb") as f:
            self.meta_model = pickle.load(f)

        if metadata_path is None:
            metadata_path = model_path.parent / "ensemble_metadata.json"

        if metadata_path.exists():
            meta = self._load_metadata(metadata_path)
            self.lookback = meta.get("lookback", 24)

        self.is_trained = True


    @staticmethod
    def _compute_metrics( #metricas para los tres modelos
        y_true: np.ndarray,
        y_pred_lstm: np.ndarray,
        y_pred_xgb: np.ndarray,
        y_pred_ensemble: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict:

        def _metrics(y_t: np.ndarray, y_p: np.ndarray) -> Dict:
            mae = float(mean_absolute_error(y_t, y_p))
            rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
            r2 = float(r2_score(y_t, y_p))
            y_tb = (y_t >= threshold).astype(int)
            y_pb = (y_p >= threshold).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_tb, y_pb, average="binary", zero_division=0
            )
            f2 = float(fbeta_score(y_tb, y_pb, beta=2, zero_division=0))
            return {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "f2_score": f2,
            }

        metrics = {
            "lstm": _metrics(y_true, y_pred_lstm),
            "xgboost": _metrics(y_true, y_pred_xgb),
            "ensemble": _metrics(y_true, y_pred_ensemble),
            "n_samples": int(len(y_true)),
        }

        for name, m in metrics.items():
            if isinstance(m, dict):
                logger.info(
                    f"  [{name:>8}] MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}"
                    f"  R^2={m['r2']:.4f}  F2={m['f2_score']:.4f}"
                )

        return metrics
######

    def run(self) -> None:
        # carga de datos
        curated_path = config.curated_data_dir / "curated_latest.parquet"
        df = load_dataframe(curated_path)
        metrics = self.train_from_df(df)
        saved = self.save(config.models_dir)
        ensure_dir(config.reports_dir)
        report = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_path": str(saved["model"]),
            "metrics": metrics,
        }
        report_path = config.reports_dir / "ensemble_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"  MAE   : {metrics['ensemble']['mae']:.4f}")
        logger.info(f"  RMSE  : {metrics['ensemble']['rmse']:.4f}")
        logger.info(f"  R^2    : {metrics['ensemble']['r2']:.4f}")
        logger.info(f"  F2    : {metrics['ensemble']['f2_score']:.4f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    StackedEnsemble().run()
