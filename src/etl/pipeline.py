"""Pipeline ETL completo y orquestación."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Literal

from ..utils import Config, setup_logger
from .extract import DataExtractor
from .transform import DataTransformer
from .load import DataSaver


config = Config()
logger = setup_logger(
    "etl.pipeline",
    log_file=config.reports_dir / "etl.log",
    level=config.log_level,
    format_type=config.log_format
)


class ETLPipeline:
    """Pipeline ETL completo para datos meteorológicos."""
    
    def __init__(
        self,
        skip_rows: int = 9,
        delimiter: str = ";"
    ):
        """
        Inicializa el pipeline ETL.
        
        Args:
            skip_rows: Filas de metadatos a saltar en archivos raw
            delimiter: Delimitador de archivos CSV
        """
        self.extractor = DataExtractor(skip_rows, delimiter)
        self.transformer = DataTransformer()
        self.saver = DataSaver()
        self.config = config
    
    def run(
        self,
        fill_gaps: bool = True,
        freq: str = "H",
        archive_raw: bool = True,
        save_formats: List[str] = None,
        save_metadata: bool = True
    ) -> Dict:
        """
        Ejecuta el pipeline ETL completo.
        
        Args:
            fill_gaps: Completar gaps temporales
            freq: Frecuencia temporal ('H' para horaria)
            archive_raw: Archivar archivos raw después de procesar
            save_formats: Formatos adicionales para guardar ['csv', 'json']
            save_metadata: Guardar archivo de metadata
        
        Returns:
            Diccionario con información del proceso
        """
        logger.info("="*80)
        logger.info("PIPELINE ETL COMPLETO")
        logger.info("="*80)
        
        try:
            # EXTRACT
            extracted_data = self.extractor.extract_all()
            
            # TRANSFORM
            df_transformed = self.transformer.transform(
                extracted_data,
                fill_gaps=fill_gaps,
                freq=freq
            )
            
            # LOAD
            saved_paths = self.saver.save(
                df_transformed,
                stage="processed",
                formats=save_formats,
                save_metadata=save_metadata
            )
            
            # ARCHIVE
            if archive_raw:
                archived = self.saver.loader.archive_raw_files()
                logger.info(f"📦 Archivados {len(archived)} archivos raw")
            
            # RESULTADO
            result = {
                "success": True,
                "n_records": len(df_transformed),
                "n_variables": len(df_transformed.columns) - 1,
                "date_range": {
                    "start": df_transformed["timestamp"].min().isoformat(),
                    "end": df_transformed["timestamp"].max().isoformat()
                },
                "saved_paths": {k: str(v) for k, v in saved_paths.items()},
                "variables": [col for col in df_transformed.columns if col != "timestamp"]
            }
            
            logger.info("="*80)
            logger.info("✅ PIPELINE ETL COMPLETADO EXITOSAMENTE")
            logger.info(f"   • Registros: {result['n_records']}")
            logger.info(f"   • Variables: {result['n_variables']}")
            logger.info(f"   • Rango: {result['date_range']['start']} → {result['date_range']['end']}")
            logger.info("="*80)
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Error en pipeline ETL: {e}")
            raise
    
    def run_partial(
        self,
        stage: Literal["extract", "transform", "load"],
        input_data: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        """
        Ejecuta una etapa específica del pipeline.
        
        Args:
            stage: Etapa a ejecutar ('extract', 'transform', 'load')
            input_data: Datos de entrada (para transform/load)
            **kwargs: Argumentos adicionales para la etapa
        
        Returns:
            Resultado de la etapa
        """
        if stage == "extract":
            return self.extractor.extract_all()
        
        elif stage == "transform":
            if input_data is None:
                raise ValueError("Se requiere input_data para transform")
            return self.transformer.transform(input_data, **kwargs)
        
        elif stage == "load":
            if input_data is None:
                raise ValueError("Se requiere input_data para load")
            return self.saver.save(input_data, **kwargs)
        
        else:
            raise ValueError(f"Stage inválido: {stage}")


def run_etl_pipeline(
    delimiter: str = ";",
    skip_rows: int = 9,
    fill_gaps: bool = True,
    archive_raw: bool = True
) -> Dict:
    """
    Función de conveniencia para ejecutar el pipeline ETL completo.
    
    Args:
        delimiter: Delimitador de archivos
        skip_rows: Filas de metadatos a saltar
        fill_gaps: Completar gaps temporales
        archive_raw: Archivar archivos raw
    
    Returns:
        Diccionario con información del proceso
    """
    pipeline = ETLPipeline(skip_rows=skip_rows, delimiter=delimiter)
    return pipeline.run(fill_gaps=fill_gaps, archive_raw=archive_raw)


def extract_only(delimiter: str = ";", skip_rows: int = 9) -> Dict[str, pd.DataFrame]:
    """
    Solo ejecuta la etapa de extracción.
    
    Args:
        delimiter: Delimitador de archivos
        skip_rows: Filas de metadatos a saltar
    
    Returns:
        Diccionario {variable: DataFrame}
    """
    extractor = DataExtractor(skip_rows=skip_rows, delimiter=delimiter)
    return extractor.extract_all()


def transform_only(
    extracted_data: Dict[str, pd.DataFrame],
    fill_gaps: bool = True,
    freq: str = "H"
) -> pd.DataFrame:
    """
    Solo ejecuta la etapa de transformación.
    
    Args:
        extracted_data: Datos extraídos
        fill_gaps: Completar gaps temporales
        freq: Frecuencia temporal
    
    Returns:
        DataFrame transformado
    """
    transformer = DataTransformer()
    return transformer.transform(extracted_data, fill_gaps=fill_gaps, freq=freq)


def load_only(
    df: pd.DataFrame,
    stage: Literal["processed", "curated"] = "processed",
    formats: List[str] = None
) -> Dict[str, Path]:
    """
    Solo ejecuta la etapa de carga.
    
    Args:
        df: DataFrame a guardar
        stage: Etapa del pipeline
        formats: Formatos adicionales
    
    Returns:
        Diccionario con rutas guardadas
    """
    saver = DataSaver()
    return saver.save(df, stage=stage, formats=formats)
