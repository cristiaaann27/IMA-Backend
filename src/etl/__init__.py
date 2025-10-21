"""Módulo de ETL para consolidación de datos meteorológicos.

Este módulo implementa un pipeline ETL modular con las siguientes etapas:
- Extract: Extracción de datos raw
- Transform: Transformación y limpieza
- Load: Carga y persistencia

Uso rápido:
    >>> from src.etl import run_etl_pipeline
    >>> result = run_etl_pipeline()

Uso modular:
    >>> from src.etl import ETLPipeline
    >>> pipeline = ETLPipeline()
    >>> result = pipeline.run()
"""

# Pipeline principal
from .pipeline import (
    ETLPipeline,
    run_etl_pipeline,
    extract_only,
    transform_only,
    load_only
)

# Componentes modulares
from .extract import DataExtractor, FileReader, identify_variable_from_filename
from .transform import DataTransformer, DataMerger, DataCleaner, FeatureEngineer
from .load import DataLoader, DataSaver
from .validators import DataValidator, ValidationResult

# Retrocompatibilidad con código antiguo
from .consolidate import (
    read_variable_file,
    merge_variables,
    clean_and_validate,
    save_stage,
    move_raw_done,
    run_etl_pipeline as run_etl_pipeline_legacy
)

__all__ = [
    # Pipeline principal
    "ETLPipeline",
    "run_etl_pipeline",
    "extract_only",
    "transform_only",
    "load_only",
    
    # Componentes Extract
    "DataExtractor",
    "FileReader",
    "identify_variable_from_filename",
    
    # Componentes Transform
    "DataTransformer",
    "DataMerger",
    "DataCleaner",
    "FeatureEngineer",
    
    # Componentes Load
    "DataLoader",
    "DataSaver",
    
    # Validadores
    "DataValidator",
    "ValidationResult",
    
    # Retrocompatibilidad
    "read_variable_file",
    "merge_variables",
    "clean_and_validate",
    "save_stage",
    "move_raw_done",
    "run_etl_pipeline_legacy"
]


