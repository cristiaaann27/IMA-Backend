"""CLI para el pipeline de predicción de precipitación."""

import sys
import argparse
import pickle
from pathlib import Path
from typing import Callable

from .utils import Config, setup_logger, load_dataframe
from .etl import run_etl_pipeline
from .features import build_features
from .modeling.train_lstm import prepare_data, train_model, save_model
from .modeling.evaluate import evaluate_model
from .modeling import UnifiedPredictor, ModelType
from .diagnostics.recommender import diagnose
from .modeling.train_xgboost import main as train_xgboost_main
from .modeling.compare_models import main as compare_models_main
from .cleanup import clean_all_files


config = Config()
logger = setup_logger(
    "cli",
    log_file=config.reports_dir / "cli.log",
    level=config.log_level,
    format_type=config.log_format
)


# ============================================================================
# FUNCIONES HELPER
# ============================================================================

def print_section(title: str, width: int = 60):
    """Imprime una sección con formato."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_error(message: str):
    """Imprime un mensaje de error."""
    print(f"❌ Error: {message}", file=sys.stderr)


def execute_command(command_name: str, func: Callable) -> int:
    """Ejecuta un comando con manejo de errores estandarizado."""
    try:
        return func()
    except Exception as e:
        logger.error(f"Error en {command_name}: {e}", exc_info=True)
        print_error(f"{command_name}: {e}")
        return 1


# ============================================================================
# COMANDOS
# ============================================================================


def cmd_etl(args):
    """Ejecuta pipeline ETL."""
    logger.info("Comando: ETL")
    
    def execute():
        result = run_etl_pipeline(
            delimiter=args.delimiter,
            skip_rows=args.skip_rows
        )
        print_section("ETL COMPLETADO")
        print(f"Registros: {result['n_records']}")
        print(f"Variables: {result['n_variables']}")
        print("="*60 + "\n")
        return 0
    
    return execute_command("ETL", execute)


def cmd_features(args):
    """Ejecuta construcción de features."""
    logger.info("Comando: Features")
    
    def execute():
        df = build_features(
            keep_wind_10m=args.keep_wind_10m,
            lags=args.lags,
            rolling_windows=args.rolling_windows
        )
        curated_path = config.curated_data_dir / 'curated_latest.parquet'
        print_section("FEATURES COMPLETADOS")
        print(f"Registros: {len(df)}")
        print(f"Features: {len(df.columns)}")
        print(f"Archivo: {curated_path}")
        print("="*60 + "\n")
        return 0
    
    return execute_command("Features", execute)


def cmd_train_lstm(args):
    """Ejecuta entrenamiento de LSTM."""
    logger.info("Comando: Train LSTM")
    
    def execute():
        curated_path = config.curated_data_dir / "curated_latest.parquet"
        if not curated_path.exists():
            raise FileNotFoundError(f"No se encontró {curated_path}. Ejecuta 'features' primero.")
        
        df = load_dataframe(curated_path)
        
        print("Preparando datos...")
        data_dict = prepare_data(
            df,
            lookback=args.lookback,
            horizon=args.horizon,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split
        )
        
        print("Entrenando modelo LSTM...")
        model, history = train_model(
            data_dict,
            hidden_size=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch,
            early_stopping_patience=args.early_stopping
        )
        
        hyperparams = {
            "hidden_size": args.hidden,
            "num_layers": args.layers,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "batch_size": args.batch,
            "lookback": args.lookback,
            "horizon": args.horizon
        }
        
        paths = save_model(model, data_dict, history, hyperparams)
        
        print_section("ENTRENAMIENTO COMPLETADO")
        print(f"Modelo: {paths['model']}")
        print(f"Mejor época: {history['best_epoch']}")
        print(f"Mejor Val Loss: {history['best_val_loss']:.6f}")
        print("="*60 + "\n")
        
        # Guardar data_dict
        data_dict_path = config.models_dir / "data_dict.pkl"
        with open(data_dict_path, "wb") as f:
            pickle.dump(data_dict, f)
        logger.info(f"Data dict guardado en {data_dict_path}")
        
        return 0
    
    return execute_command("Train LSTM", execute)


def cmd_eval_lstm(args):
    """Ejecuta evaluación de LSTM."""
    logger.info("Comando: Eval LSTM")
    
    def execute():
        data_dict_path = config.models_dir / "data_dict.pkl"
        if not data_dict_path.exists():
            raise FileNotFoundError(f"No se encontró {data_dict_path}. Ejecuta 'train-lstm' primero.")
        
        with open(data_dict_path, "rb") as f:
            data_dict = pickle.load(f)
        
        print("Evaluando modelo...")
        results = evaluate_model(data_dict)
        
        print_section("EVALUACIÓN COMPLETADA")
        test = results["splits"]["test"]
        reg = test["regression"]
        cls = test["classification"]
        print(f"MAE: {reg['MAE']:.4f} | RMSE: {reg['RMSE']:.4f} | R²: {reg['R2']:.4f}")
        print(f"Precision: {cls['precision']:.4f} | Recall: {cls['recall']:.4f} | F1: {cls['f1_score']:.4f}")
        print("\n" + "="*60 + "\n")
        return 0
    
    return execute_command("Eval LSTM", execute)


def cmd_predict(args):
    """Ejecuta generación de predicciones con LSTM."""
    logger.info("Comando: Predict LSTM")
    
    def execute():
        predictor = UnifiedPredictor(ModelType.LSTM)
        predictor.load_model()
        df_pred = predictor.predict()
        
        print_section("PREDICCIONES LSTM COMPLETADAS")
        print(f"Registros: {len(df_pred)}")
        print(f"Precipitación media: {df_pred['precip_pred_mm_hr'].mean():.4f} mm/hr")
        print(f"Precipitación máxima: {df_pred['precip_pred_mm_hr'].max():.4f} mm/hr")
        print("="*60 + "\n")
        return 0
    
    return execute_command("Predict LSTM", execute)


def cmd_diagnose(args):
    """Ejecuta diagnóstico y recomendaciones."""
    logger.info("Comando: Diagnose")
    
    def execute():
        input_path = Path(args.input) if args.input else None
        output_path = Path(args.export) if args.export else None
        
        report = diagnose(input_path, output_path)
        
        print_section("DIAGNÓSTICO COMPLETADO")
        print(f"Nivel de alerta: {report['nivel_alerta_general']['nivel'].upper()}")
        print(f"Descripción: {report['nivel_alerta_general']['descripcion']}")
        print("="*60 + "\n")
        return 0
    
    return execute_command("Diagnose", execute)


def cmd_train_xgboost(args):
    """Ejecuta entrenamiento de XGBoost."""
    logger.info("Comando: Train XGBoost")
    return execute_command("Train XGBoost", lambda: train_xgboost_main() or 0)


def cmd_compare(args):
    """Ejecuta comparación entre modelos."""
    logger.info("Comando: Compare Models")
    return execute_command("Compare Models", lambda: compare_models_main() or 0)


def cmd_predict_xgboost(args):
    """Ejecuta generación de predicciones con XGBoost."""
    logger.info("Comando: Predict XGBoost")
    
    def execute():
        predictor = UnifiedPredictor(ModelType.XGBOOST)
        predictor.load_model()
        df_pred = predictor.predict()
        
        print_section("PREDICCIONES XGBOOST COMPLETADAS")
        print(f"Registros: {len(df_pred)}")
        print(f"Precipitación media: {df_pred['precip_pred_mm_hr'].mean():.4f} mm/hr")
        print(f"Precipitación máxima: {df_pred['precip_pred_mm_hr'].max():.4f} mm/hr")
        print("="*60 + "\n")
        return 0
    
    return execute_command("Predict XGBoost", execute)


def cmd_cleanup(args):
    """Ejecuta limpieza del sistema."""
    logger.info("Comando: Cleanup")
    
    def execute():
        stats = clean_all_files(confirm=args.confirm)
        title = "LIMPIEZA COMPLETADA" if args.confirm else "SIMULACIÓN DE LIMPIEZA"
        print_section(title)
        action = "eliminados" if args.confirm else "a eliminar"
        print(f"Archivos {action}: {stats['files_removed']}")
        print(f"Espacio {'liberado' if args.confirm else 'a liberar'}: {stats['total_size_mb']:.2f} MB")
        if not args.confirm:
            print("\nPara ejecutar: python -m src.cli cleanup --confirm")
        print("="*60 + "\n")
        return 0
    
    return execute_command("Cleanup", execute)


def cmd_all(args):
    """Ejecuta pipeline completo con LSTM y XGBoost."""
    logger.info("Comando: All (pipeline completo)")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETO - LSTM + XGBOOST")
    print("="*70 + "\n")
    
    steps = [
        ("ETL", cmd_etl),
        ("Features", cmd_features),
        ("Train LSTM", cmd_train_lstm),
        ("Train XGBoost", cmd_train_xgboost),
        ("Eval LSTM", cmd_eval_lstm),
        ("Compare Models", cmd_compare),
        ("Predict LSTM", cmd_predict),
        ("Predict XGBoost", cmd_predict_xgboost),
        ("Diagnose", cmd_diagnose)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*70}")
        print(f"PASO: {step_name}")
        print(f"{'='*70}\n")
        
        result = step_func(args)
        if result != 0:
            print(f"\n[X] Pipeline detenido en: {step_name}")
            return result
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70 + "\n")
    return 0


def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline de predicción de precipitación con LSTM y XGBoost",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Comando ETL
    parser_etl = subparsers.add_parser("etl", help="Ejecutar pipeline ETL")
    parser_etl.add_argument("--delimiter", default=";", help="Delimitador de archivos")
    parser_etl.add_argument("--skip-rows", type=int, default=9, help="Filas de metadatos a saltar")
    parser_etl.set_defaults(func=cmd_etl)
    
    # Comando Features
    parser_features = subparsers.add_parser("features", help="Construir features")
    parser_features.add_argument("--keep-wind-10m", action="store_true", help="Mantener variables de viento a 10m")
    parser_features.add_argument("--lags", type=int, nargs="+", help="Lags personalizados")
    parser_features.add_argument("--rolling-windows", type=int, nargs="+", help="Ventanas móviles personalizadas")
    parser_features.set_defaults(func=cmd_features)
    
    # Comando Train LSTM
    parser_train = subparsers.add_parser("train-lstm", help="Entrenar modelo LSTM")
    parser_train.add_argument("--hidden", type=int, default=64, help="Tamaño de capa oculta")
    parser_train.add_argument("--layers", type=int, default=2, help="Número de capas LSTM")
    parser_train.add_argument("--dropout", type=float, default=0.2, help="Tasa de dropout")
    parser_train.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser_train.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser_train.add_argument("--batch", type=int, default=64, help="Tamaño de batch")
    parser_train.add_argument("--lookback", type=int, default=24, help="Ventana lookback")
    parser_train.add_argument("--horizon", type=int, default=1, help="Horizonte de predicción")
    parser_train.add_argument("--early-stopping", type=int, default=10, help="Paciencia para early stopping")
    parser_train.add_argument("--train-split", type=float, default=0.7, help="Proporción de train")
    parser_train.add_argument("--val-split", type=float, default=0.15, help="Proporción de validación")
    parser_train.add_argument("--test-split", type=float, default=0.15, help="Proporción de test")
    parser_train.set_defaults(func=cmd_train_lstm)
    
    # Comando Eval LSTM
    parser_eval = subparsers.add_parser("eval-lstm", help="Evaluar modelo LSTM")
    parser_eval.set_defaults(func=cmd_eval_lstm)
    
    # Comando Predict
    parser_predict = subparsers.add_parser("predict", help="Generar predicciones")
    parser_predict.set_defaults(func=cmd_predict)
    
    # Comando Diagnose
    parser_diagnose = subparsers.add_parser("diagnose", help="Ejecutar diagnóstico y recomendaciones")
    parser_diagnose.add_argument("--input", help="Archivo de predicciones (opcional)")
    parser_diagnose.add_argument("--export", help="Ruta de exportación (opcional)")
    parser_diagnose.set_defaults(func=cmd_diagnose)
    
    # Comando Train XGBoost
    parser_xgb = subparsers.add_parser("train-xgboost", help="Entrenar modelo XGBoost")
    parser_xgb.set_defaults(func=cmd_train_xgboost)
    
    # Comando Compare
    parser_compare = subparsers.add_parser("compare", help="Comparar LSTM vs XGBoost")
    parser_compare.set_defaults(func=cmd_compare)
    
    # Comando Predict XGBoost
    parser_predict_xgb = subparsers.add_parser("predict-xgboost", help="Generar predicciones con XGBoost")
    parser_predict_xgb.set_defaults(func=cmd_predict_xgboost)
    
    # Comando Cleanup
    parser_cleanup = subparsers.add_parser("cleanup", help="Limpiar todos los archivos generados")
    parser_cleanup.add_argument("--confirm", action="store_true", help="Confirmar eliminación real de archivos")
    parser_cleanup.set_defaults(func=cmd_cleanup)
    
    # Comando All
    parser_all = subparsers.add_parser("all", help="Ejecutar pipeline completo")
    parser_all.add_argument("--delimiter", default=";", help="Delimitador de archivos")
    parser_all.add_argument("--skip-rows", type=int, default=9, help="Filas de metadatos a saltar")
    parser_all.add_argument("--keep-wind-10m", action="store_true", help="Mantener variables de viento a 10m")
    parser_all.add_argument("--lags", type=int, nargs="+", help="Lags personalizados")
    parser_all.add_argument("--rolling-windows", type=int, nargs="+", help="Ventanas móviles personalizadas")
    parser_all.add_argument("--hidden", type=int, default=64, help="Tamaño de capa oculta")
    parser_all.add_argument("--layers", type=int, default=2, help="Número de capas LSTM")
    parser_all.add_argument("--dropout", type=float, default=0.2, help="Tasa de dropout")
    parser_all.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser_all.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser_all.add_argument("--batch", type=int, default=64, help="Tamaño de batch")
    parser_all.add_argument("--lookback", type=int, default=24, help="Ventana lookback")
    parser_all.add_argument("--horizon", type=int, default=1, help="Horizonte de predicción")
    parser_all.add_argument("--early-stopping", type=int, default=10, help="Paciencia para early stopping")
    parser_all.add_argument("--train-split", type=float, default=0.7, help="Proporción de train")
    parser_all.add_argument("--val-split", type=float, default=0.15, help="Proporción de validación")
    parser_all.add_argument("--test-split", type=float, default=0.15, help="Proporción de test")
    parser_all.add_argument("--input", help="Archivo de predicciones para diagnóstico (opcional)")
    parser_all.add_argument("--export", help="Ruta de exportación de diagnóstico (opcional)")
    parser_all.set_defaults(func=cmd_all)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


