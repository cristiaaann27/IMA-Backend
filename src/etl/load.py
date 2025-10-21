"""Módulo de carga y persistencia de datos (Load)."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Literal, Optional
from datetime import datetime

from ..utils import Config, setup_logger, ensure_dir, move_files, save_dataframe


config = Config()
logger = setup_logger(
    "etl.load",
    log_file=config.reports_dir / "etl.log",
    level=config.log_level,
    format_type=config.log_format
)


class DataLoader:
    """Cargador de datos a diferentes etapas del pipeline."""
    
    def __init__(self):
        """Inicializa el cargador."""
        self.config = config
    
    def save_processed(
        self,
        df: pd.DataFrame,
        create_version: bool = True
    ) -> Dict[str, Path]:
        """
        Guarda datos en la etapa 'processed'.
        
        Args:
            df: DataFrame a guardar
            create_version: Si True, crea versión con timestamp
        
        Returns:
            Diccionario con rutas de archivos guardados
        """
        logger.info("💾 Guardando datos en 'processed'")
        
        output_dir = self.config.processed_data_dir
        ensure_dir(output_dir)
        
        paths = {}
        
        # Guardar versión latest
        latest_path = output_dir / "processed_latest.parquet"
        save_dataframe(df, latest_path, format="parquet")
        paths["latest"] = latest_path
        logger.info(f"  ✅ Latest: {latest_path}")
        
        # Guardar versión con timestamp
        if create_version:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_path = output_dir / f"processed_{timestamp_str}.parquet"
            save_dataframe(df, versioned_path, format="parquet")
            paths["versioned"] = versioned_path
            logger.info(f"  ✅ Versión: {versioned_path}")
        
        return paths
    
    def save_curated(
        self,
        df: pd.DataFrame,
        create_version: bool = True
    ) -> Dict[str, Path]:
        """
        Guarda datos en la etapa 'curated'.
        
        Args:
            df: DataFrame a guardar
            create_version: Si True, crea versión con timestamp
        
        Returns:
            Diccionario con rutas de archivos guardados
        """
        logger.info("💾 Guardando datos en 'curated'")
        
        output_dir = self.config.curated_data_dir
        ensure_dir(output_dir)
        
        paths = {}
        
        # Guardar versión latest
        latest_path = output_dir / "curated_latest.parquet"
        save_dataframe(df, latest_path, format="parquet")
        paths["latest"] = latest_path
        logger.info(f"  ✅ Latest: {latest_path}")
        
        # Guardar versión con timestamp
        if create_version:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_path = output_dir / f"curated_{timestamp_str}.parquet"
            save_dataframe(df, versioned_path, format="parquet")
            paths["versioned"] = versioned_path
            logger.info(f"  ✅ Versión: {versioned_path}")
        
        return paths
    
    def save_stage(
        self,
        df: pd.DataFrame,
        stage: Literal["processed", "curated"],
        create_version: bool = True
    ) -> Dict[str, Path]:
        """
        Guarda datos en una etapa específica.
        
        Args:
            df: DataFrame a guardar
            stage: 'processed' o 'curated'
            create_version: Si True, crea versión con timestamp
        
        Returns:
            Diccionario con rutas de archivos guardados
        """
        if stage == "processed":
            return self.save_processed(df, create_version)
        elif stage == "curated":
            return self.save_curated(df, create_version)
        else:
            raise ValueError(f"Stage inválido: {stage}. Use 'processed' o 'curated'")
    
    def archive_raw_files(
        self,
        archive_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Archiva archivos raw procesados.
        
        Args:
            archive_dir: Directorio destino (default: processed/ingested)
        
        Returns:
            Lista de archivos movidos
        """
        logger.info("📦 Archivando archivos raw")
        
        source_dir = self.config.raw_data_dir
        
        if archive_dir is None:
            archive_dir = self.config.processed_data_dir / "ingested"
        
        ensure_dir(archive_dir)
        
        # Buscar archivos a mover
        patterns = ["*.csv", "*.tsv", "*.txt"]
        moved = []
        
        for pattern in patterns:
            files = list(source_dir.glob(pattern))
            for file_path in files:
                dest_path = archive_dir / file_path.name
                file_path.rename(dest_path)
                moved.append(dest_path)
                logger.debug(f"  📁 {file_path.name} → {archive_dir}")
        
        if moved:
            logger.info(f"  ✅ Archivados {len(moved)} archivos")
        else:
            logger.info("  ℹ️ No hay archivos para archivar")
        
        return moved
    
    def export_to_csv(
        self,
        df: pd.DataFrame,
        output_path: Path,
        include_index: bool = False
    ) -> Path:
        """
        Exporta DataFrame a CSV.
        
        Args:
            df: DataFrame a exportar
            output_path: Ruta de salida
            include_index: Incluir índice en CSV
        
        Returns:
            Ruta del archivo guardado
        """
        logger.info(f"📄 Exportando a CSV: {output_path}")
        
        ensure_dir(output_path.parent)
        df.to_csv(output_path, index=include_index)
        
        logger.info(f"  ✅ Exportado: {len(df)} registros")
        
        return output_path
    
    def create_metadata(
        self,
        df: pd.DataFrame,
        stage: str,
        additional_info: Optional[Dict] = None
    ) -> Dict:
        """
        Crea metadata sobre el dataset.
        
        Args:
            df: DataFrame
            stage: Etapa del pipeline
            additional_info: Información adicional
        
        Returns:
            Diccionario con metadata
        """
        metadata = {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "n_records": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if "timestamp" in df.columns else None,
                "end": df["timestamp"].max().isoformat() if "timestamp" in df.columns else None
            },
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        return metadata
    
    def save_metadata(
        self,
        metadata: Dict,
        output_path: Path
    ) -> Path:
        """
        Guarda metadata en JSON.
        
        Args:
            metadata: Diccionario con metadata
            output_path: Ruta de salida
        
        Returns:
            Ruta del archivo guardado
        """
        import json
        
        ensure_dir(output_path.parent)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Metadata guardada: {output_path}")
        
        return output_path


class DataSaver:
    """Guardador de datos con múltiples formatos."""
    
    def __init__(self):
        """Inicializa el guardador."""
        self.loader = DataLoader()
    
    def save(
        self,
        df: pd.DataFrame,
        stage: Literal["processed", "curated"],
        formats: List[str] = None,
        create_version: bool = True,
        save_metadata: bool = True
    ) -> Dict[str, Path]:
        """
        Guarda datos en múltiples formatos.
        
        Args:
            df: DataFrame a guardar
            stage: Etapa del pipeline
            formats: Formatos adicionales ['csv', 'json']
            create_version: Crear versión con timestamp
            save_metadata: Guardar archivo de metadata
        
        Returns:
            Diccionario con todas las rutas guardadas
        """
        logger.info("="*60)
        logger.info(f"CARGA DE DATOS (LOAD) - Stage: {stage}")
        logger.info("="*60)
        
        all_paths = {}
        
        # 1. Guardar en formato principal (parquet)
        paths = self.loader.save_stage(df, stage, create_version)
        all_paths.update(paths)
        
        # 2. Guardar en formatos adicionales
        if formats:
            output_dir = (
                self.loader.config.processed_data_dir if stage == "processed"
                else self.loader.config.curated_data_dir
            )
            
            for fmt in formats:
                if fmt == "csv":
                    csv_path = output_dir / f"{stage}_latest.csv"
                    self.loader.export_to_csv(df, csv_path)
                    all_paths["csv"] = csv_path
                
                elif fmt == "json":
                    json_path = output_dir / f"{stage}_latest.json"
                    df.to_json(json_path, orient="records", date_format="iso")
                    all_paths["json"] = json_path
                    logger.info(f"  ✅ JSON: {json_path}")
        
        # 3. Guardar metadata
        if save_metadata:
            metadata = self.loader.create_metadata(df, stage)
            metadata_path = (
                self.loader.config.processed_data_dir / f"{stage}_metadata.json"
                if stage == "processed"
                else self.loader.config.curated_data_dir / f"{stage}_metadata.json"
            )
            self.loader.save_metadata(metadata, metadata_path)
            all_paths["metadata"] = metadata_path
        
        logger.info("="*60)
        
        return all_paths
