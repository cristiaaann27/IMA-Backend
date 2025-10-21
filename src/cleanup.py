"""Comando para limpiar todos los archivos generados del sistema."""

import shutil
from pathlib import Path
from typing import List

from .utils import Config, setup_logger

config = Config()
logger = setup_logger(
    "cleanup",
    log_file=config.reports_dir / "cleanup.log",
    level=config.log_level,
    format_type=config.log_format
)


def clean_all_files(confirm: bool = False) -> dict:
    """
    Elimina todos los archivos generados del sistema, manteniendo las carpetas.
    
    PRESERVA:
    - Archivos .log (logs del sistema)
    - Estructura de carpetas
    - Archivos de configuración
    
    Args:
        confirm: Si True, ejecuta la limpieza. Si False, solo muestra qué se eliminaría.
    
    Returns:
        Diccionario con estadísticas de limpieza
    """
    logger.info("="*60)
    logger.info("INICIANDO LIMPIEZA DEL SISTEMA")
    logger.info("="*60)
    logger.info("PRESERVANDO: Archivos .log, estructura de carpetas, archivos de configuración")
    
    stats = {
        "files_removed": 0,
        "total_size_mb": 0,
        "errors": []
    }
    
    # Directorios a limpiar (solo archivos, no las carpetas)
    cleanup_dirs = [
        # Modelos entrenados
        {
            "path": config.models_dir,
            "description": "Modelos entrenados (LSTM, XGBoost, scalers, metadata)"
        },
        
        # Datos procesados
        {
            "path": config.processed_data_dir,
            "description": "Datos procesados del pipeline ETL"
        },
        
        # Datos curados
        {
            "path": config.curated_data_dir,
            "description": "Datos con features construidos"
        },
        
        # Predicciones
        {
            "path": config.predictions_dir,
            "description": "Archivos de predicciones generadas"
        },
        
        # Reportes
        {
            "path": config.reports_dir,
            "description": "Reportes, logs y gráficos generados"
        },
        
        # Directorios principales (solo archivos)
        {
            "path": Path("models"),
            "description": "Directorio de modelos completo"
        },
        
        {
            "path": Path("data"),
            "description": "Directorio de datos completo"
        },
        
        {
            "path": Path("reports"),
            "description": "Directorio de reportes completo"
        }
    ]
    
    # Limpiar archivos en cada directorio
    for item in cleanup_dirs:
        path = item["path"]
        if path.exists() and path.is_dir():
            try:
                # Obtener todos los archivos en el directorio, EXCEPTO logs
                files_to_remove = list(path.rglob("*"))
                files_to_remove = [f for f in files_to_remove if f.is_file() and not f.suffix == '.log']
                
                if files_to_remove:
                    # Calcular tamaño total
                    total_size = sum(f.stat().st_size for f in files_to_remove)
                    size_mb = total_size / (1024 * 1024)
                    file_count = len(files_to_remove)
                    
                    logger.info(f"[DIR] {path}: {file_count} archivos, {size_mb:.2f} MB")
                    
                    if confirm:
                        # Eliminar solo los archivos, no los directorios
                        for file_path in files_to_remove:
                            try:
                                file_path.unlink()
                                stats["files_removed"] += 1
                            except Exception as e:
                                error_msg = f"Error eliminando archivo {file_path}: {e}"
                                logger.error(error_msg)
                                stats["errors"].append(error_msg)
                        
                        stats["total_size_mb"] += size_mb
                        logger.info(f"[OK] Eliminados {file_count} archivos de: {path}")
                else:
                    logger.info(f"[SKIP] {path}: No hay archivos para eliminar")
                        
            except Exception as e:
                error_msg = f"Error procesando directorio {path}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
        else:
            logger.info(f"[SKIP] No existe o no es directorio: {path}")
    
    # Limpiar archivos Python compilados (solo archivos .pyc, no las carpetas __pycache__)
    pycache_dirs = list(Path(".").rglob("__pycache__"))
    for pycache in pycache_dirs:
        try:
            if pycache.exists() and pycache.is_dir():
                # Obtener solo archivos .pyc
                pyc_files = list(pycache.rglob("*.pyc"))
                
                if pyc_files:
                    file_count = len(pyc_files)
                    total_size = sum(f.stat().st_size for f in pyc_files)
                    size_mb = total_size / (1024 * 1024)
                    
                    logger.info(f"[DIR] __pycache__: {pycache} ({file_count} archivos .pyc, {size_mb:.2f} MB)")
                    
                    if confirm:
                        # Eliminar solo los archivos .pyc
                        for pyc_file in pyc_files:
                            try:
                                pyc_file.unlink()
                                stats["files_removed"] += 1
                            except Exception as e:
                                error_msg = f"Error eliminando {pyc_file}: {e}"
                                logger.error(error_msg)
                                stats["errors"].append(error_msg)
                        
                        stats["total_size_mb"] += size_mb
                        logger.info(f"[OK] Eliminados {file_count} archivos .pyc de: {pycache}")
                else:
                    logger.info(f"[SKIP] __pycache__: {pycache} (no hay archivos .pyc)")
        except Exception as e:
            error_msg = f"Error procesando __pycache__ {pycache}: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
    
    # Resumen
    logger.info("="*60)
    if confirm:
        logger.info("LIMPIEZA COMPLETADA")
        logger.info(f"[FILE] Archivos eliminados: {stats['files_removed']}")
        logger.info(f"[SIZE] Espacio liberado: {stats['total_size_mb']:.2f} MB")
        if stats['errors']:
            logger.info(f"[ERROR] Errores: {len(stats['errors'])}")
            for error in stats['errors']:
                logger.error(f"  - {error}")
    else:
        logger.info("SIMULACIÓN DE LIMPIEZA COMPLETADA")
        logger.info(f"[FILE] Archivos a eliminar: {stats['files_removed']}")
        logger.info(f"[SIZE] Espacio a liberar: {stats['total_size_mb']:.2f} MB")
        logger.info("")
        logger.info("[WARN] Para ejecutar la limpieza real, usa --confirm")
    
    logger.info("="*60)
    
    return stats


def main():
    """Función principal para limpieza."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Limpiar todos los archivos generados del sistema")
    parser.add_argument("--confirm", action="store_true", 
                       help="Confirmar la eliminación real de archivos")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Solo mostrar qué se eliminaría (por defecto)")
    
    args = parser.parse_args()
    
    if args.confirm:
        confirm = True
        logger.info("🚨 MODO DE CONFIRMACIÓN ACTIVADO - LOS ARCHIVOS SERÁN ELIMINADOS")
    else:
        confirm = False
        logger.info("🔍 MODO SIMULACIÓN - Solo mostrando qué se eliminaría")
    
    stats = clean_all_files(confirm=confirm)
    
    if not confirm:
        print("\n" + "="*60)
        print("SIMULACIÓN DE LIMPIEZA")
        print("="*60)
        print(f"[FILE] Archivos a eliminar: {stats['files_removed']}")
        print(f"[SIZE] Espacio a liberar: {stats['total_size_mb']:.2f} MB")
        print("")
        print("[INFO] Solo se eliminarán archivos, las carpetas y logs se mantendrán")
        print("[WARN] Para ejecutar la limpieza real, ejecuta:")
        print("    python -m src.cleanup --confirm")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("LIMPIEZA COMPLETADA")
        print("="*60)
        print(f"[FILE] Archivos eliminados: {stats['files_removed']}")
        print(f"[SIZE] Espacio liberado: {stats['total_size_mb']:.2f} MB")
        print("[INFO] Las carpetas y logs se mantuvieron intactos")
        if stats['errors']:
            print(f"[ERROR] Errores: {len(stats['errors'])}")
        print("="*60)


if __name__ == "__main__":
    main()
