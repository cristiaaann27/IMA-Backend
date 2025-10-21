"""Consolidación de archivos de variables meteorológicas.

NOTA: Este módulo se mantiene solo para retrocompatibilidad.
Para nuevos desarrollos, usar los módulos modulares:
- extract.py (DataExtractor, FileReader)
- transform.py (DataTransformer, DataMerger, DataCleaner)
- load.py (DataLoader, DataSaver)
- pipeline.py (ETLPipeline)
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Literal, Optional
from datetime import datetime

from ..utils import Config, setup_logger, ensure_dir, move_files, save_dataframe
from .validators import DataValidator, ValidationResult

# Importar componentes nuevos para reutilizar lógica
from .extract import (
    normalize_string as _normalize_string,
    identify_variable_from_filename as _identify_variable,
    FileReader as _FileReader
)
from .transform import DataMerger as _DataMerger, DataCleaner as _DataCleaner
from .load import DataLoader as _DataLoader


# Configuración
config = Config()
logger = setup_logger(
    "etl.consolidate",
    log_file=config.reports_dir / "etl.log",
    level=config.log_level,
    format_type=config.log_format
)

# Mapeo de patrones de búsqueda (sin tildes, lowercase) a nombres de variables
# Se buscarán estos patrones en los nombres de archivo
VARIABLE_PATTERNS = {
    # Precipitación (lluvia, rain, precip)
    "precip": ["precip", "precipitacion", "precipitación", "lluvia", "rain"],
    # Humedad relativa
    "rh_2m": ["rh", "humedad", "humidity", "relative", "hr", "humedad relativa"],
    # Temperatura
    "temp_2m": ["temp", "temperatura", "temperature"],
    # Velocidad viento 2m - IMPORTANTE: Buscar "velocidad" + "viento" + "2"
    "wind_speed_2m": [
        "velocidad del viento a 2 metros",
        "velocidad viento 2 metros",
        "velocidad viento 2m",
        "velocidadviento2m",
        "wind speed 2m",
        "ws_2m",
        "velocidad_2m"
    ],
    # Dirección viento 2m - IMPORTANTE: Buscar "direccion" + "viento" + "2"
    "wind_dir_2m": [
        "direccion del viento a 2 metros",
        "direccion viento 2 metros",
        "direccion viento 2m",
        "direccionviento2m",
        "wind dir 2m",
        "wd_2m",
        "direccion_2m"
    ],
    # Velocidad viento 10m
    "wind_speed_10m": [
        "velocidad del viento a 10 metros",
        "velocidad viento 10 metros",
        "velocidad viento 10m",
        "velocidadviento10m",
        "wind speed 10m",
        "ws_10m",
        "velocidad_10m"
    ],
    # Dirección viento 10m
    "wind_dir_10m": [
        "direccion del viento a 10 metros",
        "direccion viento 10 metros",
        "direccion viento 10m",
        "direccionviento10m",
        "wind dir 10m",
        "wd_10m",
        "direccion_10m"
    ],
}

# Nombres de variables finales
VARIABLE_NAMES = {
    "precip": "precip_mm_hr",
    "rh_2m": "rh_2m_pct",
    "temp_2m": "temp_2m_c",
    "wind_speed_2m": "wind_speed_2m_ms",
    "wind_dir_2m": "wind_dir_2m_deg",
    "wind_speed_10m": "wind_speed_10m_ms",
    "wind_dir_10m": "wind_dir_10m_deg",
}


def normalize_string(text: str) -> str:
    """
    Normaliza string para comparación (sin tildes, lowercase, sin espacios).
    
    LEGACY: Usa la función del módulo extract.py
    """
    return _normalize_string(text)


def identify_variable_from_filename(filename: str) -> Optional[str]:
    """
    Identifica la variable meteorológica a partir del nombre de archivo.
    
    LEGACY: Usa la función del módulo extract.py
    """
    return _identify_variable(filename)


def read_variable_file(
    path: Path,
    var_name: str,
    skip_rows: int = 9,
    delimiter: str = ";"
) -> pd.DataFrame:
    """
    Lee un archivo de variable meteorológica.
    
    LEGACY: Usa FileReader del módulo extract.py
    """
    reader = _FileReader(skip_rows=skip_rows, delimiter=delimiter)
    return reader.read_file(path, var_name)


def merge_variables(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge de múltiples DataFrames de variables por timestamp.
    
    LEGACY: Usa DataMerger del módulo transform.py
    """
    merger = _DataMerger()
    return merger.merge_variables(dfs)


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y validación de datos consolidados.
    
    LEGACY: Usa DataCleaner del módulo transform.py
    """
    cleaner = _DataCleaner()
    return cleaner.clean_and_validate(df)


def save_stage(
    df: pd.DataFrame,
    stage: Literal["processed", "curated"]
) -> Path:
    """
    Guarda DataFrame en la etapa correspondiente.
    
    LEGACY: Usa DataLoader del módulo load.py
    """
    loader = _DataLoader()
    paths = loader.save_stage(df, stage, create_version=True)
    return paths['versioned']


def move_raw_done() -> List[Path]:
    """
    Mueve archivos procesados de raw/ a processed/ingested/.
    
    LEGACY: Usa DataLoader del módulo load.py
    """
    loader = _DataLoader()
    return loader.archive_raw_files()


def run_etl_pipeline(delimiter: str = ";", skip_rows: int = 9) -> Dict:
    """
    Ejecuta el pipeline completo de ETL.
    
    LEGACY: Usa el nuevo ETLPipeline del módulo pipeline.py
    
    Args:
        delimiter: Delimitador de archivos
        skip_rows: Filas de metadatos a saltar
    
    Returns:
        Diccionario con información del proceso
    """
    # Importar el nuevo pipeline
    from .pipeline import ETLPipeline
    
    # Usar el nuevo pipeline internamente
    pipeline = ETLPipeline(skip_rows=skip_rows, delimiter=delimiter)
    return pipeline.run()


