"""
Comparación de métricas entre modelos LSTM y XGBoost.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from ..utils import Config, setup_logger, ensure_dir


config = Config()
logger = setup_logger(
    'modeling.compare',
    log_file=config.reports_dir / 'comparison.log',
    level=config.log_level,
    format_type=config.log_format
)


def load_models(models_dir: Path) -> Dict:
    """Carga ambos modelos."""
    logger.info("Cargando modelos...")
    
    models = {}
    
    # XGBoost
    xgb_path = models_dir / 'xgboost_latest.json'
    if xgb_path.exists():
        models['xgboost'] = xgb.XGBRegressor()
        models['xgboost'].load_model(str(xgb_path))
        logger.info("✓ XGBoost cargado")
    
    # LSTM evaluation results
    lstm_eval_path = config.reports_dir / 'evaluation_results.json'
    if lstm_eval_path.exists():
        with open(lstm_eval_path, 'r') as f:
            models['lstm_evaluation'] = json.load(f)
        logger.info("✓ LSTM evaluación cargada")
    else:
        logger.warning(f"Archivo de evaluación LSTM no encontrado: {lstm_eval_path}")
    
    # XGBoost metadata
    xgb_meta_path = models_dir / 'xgboost_metadata.json'
    if xgb_meta_path.exists():
        with open(xgb_meta_path, 'r') as f:
            models['xgboost_metadata'] = json.load(f)
        logger.info("✓ XGBoost metadata cargado")
    
    return models




def compare_metrics(models: Dict, output_dir: Path, data_dict: Dict = None):
    """
    Compara métricas entre LSTM y XGBoost.
    
    Args:
        models: Diccionario con modelos y metadata
        output_dir: Directorio de salida
        data_dict: Diccionario con datos de test (opcional)
    """
    logger.info("Comparando métricas LSTM vs XGBoost...")
    
    # Intentar cargar métricas LSTM de evaluation_results.json
    lstm_eval = models.get('lstm_evaluation', {})
    lstm_splits = lstm_eval.get('splits', {})
    lstm_test_metrics = lstm_splits.get('test', {})
    
    lstm_metrics = {
        'mae': lstm_test_metrics.get('regression', {}).get('MAE'),
        'rmse': lstm_test_metrics.get('regression', {}).get('RMSE'),
        'r2': lstm_test_metrics.get('regression', {}).get('R2'),
        'precision': lstm_test_metrics.get('classification', {}).get('precision'),
        'recall': lstm_test_metrics.get('classification', {}).get('recall'),
        'f1_score': lstm_test_metrics.get('classification', {}).get('f1_score')
    }
    
    # Si no hay métricas LSTM, intentar evaluar sobre la marcha
    if all(v is None for v in lstm_metrics.values()):
        logger.warning("Métricas LSTM no encontradas en evaluation_results.json")
        if data_dict is not None:
            logger.info("Evaluando modelo LSTM con data_dict...")
            lstm_metrics = evaluate_lstm_on_the_fly(models, data_dict)
        else:
            logger.warning("data_dict no disponible. No se pueden calcular métricas LSTM.")
    
    xgb_metrics = models.get('xgboost_metadata', {}).get('metrics', {})
    
    # Crear tabla comparativa
    comparison = {
        'Métrica': ['MAE', 'RMSE', 'R²', 'Precision', 'Recall', 'F1-Score', 'F2-Score'],
        'LSTM': [
            lstm_metrics.get('mae') if lstm_metrics.get('mae') is not None else 'N/A',
            lstm_metrics.get('rmse') if lstm_metrics.get('rmse') is not None else 'N/A',
            lstm_metrics.get('r2') if lstm_metrics.get('r2') is not None else 'N/A',
            lstm_metrics.get('precision') if lstm_metrics.get('precision') is not None else 'N/A',
            lstm_metrics.get('recall') if lstm_metrics.get('recall') is not None else 'N/A',
            lstm_metrics.get('f1_score') if lstm_metrics.get('f1_score') is not None else 'N/A',
            lstm_metrics.get('f2_score') if lstm_metrics.get('f2_score') is not None else 'N/A'
        ],
        'XGBoost': [
            xgb_metrics.get('mae', 'N/A'),
            xgb_metrics.get('rmse', 'N/A'),
            xgb_metrics.get('r2', 'N/A'),
            xgb_metrics.get('precision', 'N/A'),
            xgb_metrics.get('recall', 'N/A'),
            xgb_metrics.get('f1_score', 'N/A'),
            xgb_metrics.get('f2_score', 'N/A')
        ]
    }
    
    comp_df = pd.DataFrame(comparison)
    
    # Guardar CSV
    comp_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    logger.info("✓ Comparación completada")
    logger.info("\nResumen:")
    print("\n" + comp_df.to_string(index=False))
    
    return comp_df




def evaluate_lstm_on_the_fly(models: Dict, data_dict: Dict) -> Dict:
    """Evalúa modelo LSTM sobre la marcha si no hay métricas guardadas."""
    try:
        from .train_lstm import LSTMModel
        from .evaluate import predict_sequences
        import pickle
        
        logger.info("Evaluando modelo LSTM con datos de test...")
        
        # Cargar modelo
        model_path = config.models_dir / 'lstm_latest.pt'
        metadata_path = config.models_dir / 'lstm_metadata.json'
        scaler_path = config.models_dir / 'scaler.pkl'
        
        # Verificar archivos
        logger.info(f"Verificando archivos del modelo LSTM:")
        logger.info(f"  - Modelo: {model_path.exists()} ({model_path})")
        logger.info(f"  - Metadata: {metadata_path.exists()} ({metadata_path})")
        logger.info(f"  - Scaler: {scaler_path.exists()} ({scaler_path})")
        
        if not all([model_path.exists(), metadata_path.exists(), scaler_path.exists()]):
            logger.warning("Archivos del modelo LSTM no encontrados")
            return {'mae': None, 'rmse': None, 'r2': None, 'precision': None, 'recall': None, 'f1_score': None, 'f2_score': None}
        
        # Cargar metadata y scalers
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        
        # Crear y cargar modelo
        arch = metadata['model_architecture']
        data_config = metadata.get('data', {})
        horizon = data_config.get('horizon', 6)
        model = LSTMModel(
            input_size=arch['input_size'],
            hidden_size=arch['hidden_size'],
            num_layers=arch['num_layers'],
            dropout=arch.get('dropout', 0.2),
            output_size=horizon
        )
        
        import torch
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Obtener datos de test
        X_test = data_dict['X_test']
        y_test_scaled = data_dict['y_test']
        
        # Predecir
        y_pred_scaled = predict_sequences(model, X_test, 'cpu')
        
        # Des-escalar
        scaler_y = scalers['scaler_y']
        y_test = scaler_y.inverse_transform(y_test_scaled).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
        y_pred = np.maximum(y_pred, 0)
        
        # Calcular métricas
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support, fbeta_score
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Clasificación binaria
        threshold = config.precip_event_mmhr
        y_test_binary = (y_test >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_binary, y_pred_binary, average='binary', zero_division=0
        )
        
        f2 = fbeta_score(y_test_binary, y_pred_binary, beta=2, zero_division=0)
        
        logger.info(f"✓ LSTM evaluado: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, F1={f1:.4f}, F2={f2:.4f}")
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'f2_score': float(f2)
        }
        
    except Exception as e:
        logger.error(f"Error evaluando LSTM: {e}", exc_info=True)
        return {'mae': None, 'rmse': None, 'r2': None, 'precision': None, 'recall': None, 'f1_score': None, 'f2_score': None}


def main():
    """Comparación de métricas entre modelos."""
    logger.info("="*60)
    logger.info("COMPARACIÓN DE MÉTRICAS: LSTM vs XGBOOST")
    logger.info("="*60)
    
    # Paths
    models_dir = config.models_dir
    reports_dir = config.reports_dir
    ensure_dir(reports_dir)
    
    # 1. Cargar data_dict si existe (para evaluación LSTM)
    data_dict = None
    data_dict_path = models_dir / 'data_dict.pkl'
    if data_dict_path.exists():
        import pickle
        logger.info("Cargando data_dict para evaluación LSTM...")
        with open(data_dict_path, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        logger.warning(f"data_dict.pkl no encontrado en {data_dict_path}")
    
    # 2. Verificar si existe evaluation_results.json, si no, generarlo
    eval_results_path = reports_dir / 'evaluation_results.json'
    if not eval_results_path.exists() and data_dict is not None:
        logger.info("evaluation_results.json no existe. Ejecutando evaluación LSTM...")
        try:
            from .evaluate import evaluate_model
            evaluate_model(data_dict, device='cpu')
            logger.info("✓ Evaluación LSTM completada")
        except Exception as e:
            logger.error(f"Error ejecutando evaluación LSTM: {e}", exc_info=True)
    
    # 3. Cargar modelos y métricas
    models = load_models(models_dir)
    
    # 4. Comparar métricas
    compare_metrics(models, reports_dir, data_dict)
    
    # 5. Generar reporte final
    report = {
        'timestamp': datetime.now().isoformat(),
        'models_compared': list(models.keys()),
        'comparison_file': 'model_comparison.csv'
    }
    
    with open(reports_dir / 'comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("="*60)
    logger.info("✓ COMPARACIÓN COMPLETADA")
    logger.info(f"  Resultados en: {reports_dir / 'model_comparison.csv'}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
