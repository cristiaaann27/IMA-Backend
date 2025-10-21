"""Módulo de extracción de datos meteorológicos (Extract)."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import unicodedata

from ..utils import Config, setup_logger


config = Config()
logger = setup_logger(
    "etl.extract",
    log_file=config.reports_dir / "etl.log",
    level=config.log_level,
    format_type=config.log_format
)


# Mapeo de patrones de búsqueda a nombres de variables
VARIABLE_PATTERNS = {
    "precip": ["precip", "precipitacion", "precipitación", "lluvia", "rain"],
    "rh_2m": ["rh", "humedad", "humidity", "relative", "hr", "humedad relativa"],
    "temp_2m": ["temp", "temperatura", "temperature"],
    "wind_speed_2m": [
        "velocidad del viento a 2 metros",
        "velocidad viento 2 metros",
        "velocidad viento 2m",
        "velocidadviento2m",
        "wind speed 2m",
        "ws_2m",
        "velocidad_2m"
    ],
    "wind_dir_2m": [
        "direccion del viento a 2 metros",
        "direccion viento 2 metros",
        "direccion viento 2m",
        "direccionviento2m",
        "wind dir 2m",
        "wd_2m",
        "direccion_2m"
    ],
    "wind_speed_10m": [
        "velocidad del viento a 10 metros",
        "velocidad viento 10 metros",
        "velocidad viento 10m",
        "velocidadviento10m",
        "wind speed 10m",
        "ws_10m",
        "velocidad_10m"
    ],
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
    
    Args:
        text: Texto a normalizar
    
    Returns:
        Texto normalizado
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = text.replace(' ', '').replace('_', '').replace('-', '')
    return text


def identify_variable_from_filename(filename: str) -> Optional[str]:
    """
    Identifica la variable meteorológica a partir del nombre de archivo.
    
    Args:
        filename: Nombre del archivo (sin extensión)
    
    Returns:
        Nombre de variable o None si no se identifica
    """
    normalized_filename = normalize_string(filename)
    logger.debug(f"Identificando variable para: '{filename}'")
    
    for var_key, patterns in VARIABLE_PATTERNS.items():
        for pattern in patterns:
            normalized_pattern = normalize_string(pattern)
            if normalized_pattern in normalized_filename:
                var_name = VARIABLE_NAMES[var_key]
                logger.info(f"✅ '{filename}' → '{var_name}'")
                return var_name
    
    logger.warning(f"⚠️ No se identificó variable para '{filename}'")
    return None


class FileReader:
    """Lector de archivos de variables meteorológicas."""
    
    def __init__(self, skip_rows: int = 9, delimiter: str = ";"):
        """
        Inicializa el lector.
        
        Args:
            skip_rows: Filas de metadatos a saltar
            delimiter: Delimitador por defecto
        """
        self.skip_rows = skip_rows
        self.delimiter = delimiter
    
    def read_file(self, path: Path, var_name: str) -> pd.DataFrame:
        """
        Lee un archivo de variable meteorológica.
        
        Estructura esperada:
        - Primeras skip_rows filas: metadatos
        - Columnas: YEAR, MO, DY, HR, value
        
        Args:
            path: Ruta del archivo
            var_name: Nombre de la variable
        
        Returns:
            DataFrame con timestamp y la variable
        """
        logger.info(f"Leyendo: {path.name} → {var_name}")
        
        try:
            df = self._read_csv(path)
            df = self._normalize_columns(df)
            df = self._extract_temporal_columns(df)
            df = self._create_timestamp(df)
            df = self._extract_value_column(df, var_name)
            df = self._clean_invalid_rows(df, var_name)
            
            logger.info(
                f"✅ {len(df)} registros | "
                f"{df['timestamp'].min()} → {df['timestamp'].max()}"
            )
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error leyendo {path.name}: {e}")
            raise
    
    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Lee CSV con auto-detección de delimitador."""
        try:
            return pd.read_csv(
                path,
                skiprows=self.skip_rows,
                delimiter=self.delimiter,
                skipinitialspace=True
            )
        except Exception:
            return pd.read_csv(
                path,
                skiprows=self.skip_rows,
                sep=None,
                engine='python',
                skipinitialspace=True
            )
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de columnas."""
        df.columns = df.columns.str.strip().str.lower()
        logger.debug(f"Columnas: {list(df.columns)}")
        return df
    
    def _extract_temporal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifica y extrae columnas temporales."""
        year_col = month_col = day_col = hour_col = None
        
        # Búsqueda por nombre
        for col in df.columns:
            if 'year' in col or col == 'yr':
                year_col = col
            elif col == 'mo' or 'month' in col:
                month_col = col
            elif 'dy' in col or 'day' in col:
                day_col = col
            elif col == 'hr' or 'hour' in col:
                hour_col = col
        
        # Fallback: usar posición
        if not all([year_col, month_col, day_col, hour_col]):
            if len(df.columns) >= 4:
                year_col, month_col, day_col, hour_col = df.columns[:4]
        
        if not all([year_col, month_col, day_col, hour_col]):
            raise ValueError(f"No se identificaron columnas temporales: {list(df.columns)}")
        
        # Renombrar
        df = df.rename(columns={
            year_col: 'year',
            month_col: 'month',
            day_col: 'day',
            hour_col: 'hour'
        })
        
        return df
    
    def _create_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea columna timestamp."""
        df["timestamp"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]],
            errors="coerce"
        )
        return df
    
    def _extract_value_column(self, df: pd.DataFrame, var_name: str) -> pd.DataFrame:
        """Extrae columna de valor y la renombra."""
        time_cols = {'year', 'month', 'day', 'hour', 'timestamp'}
        value_col = None
        
        for col in df.columns:
            if col not in time_cols:
                value_col = col
                break
        
        if value_col is None:
            raise ValueError("No se encontró columna de valor")
        
        df = df[["timestamp", value_col]].copy()
        df.rename(columns={value_col: var_name}, inplace=True)
        df[var_name] = pd.to_numeric(df[var_name], errors="coerce")
        
        return df
    
    def _clean_invalid_rows(self, df: pd.DataFrame, var_name: str) -> pd.DataFrame:
        """Elimina filas con timestamp inválido."""
        initial_rows = len(df)
        df = df.dropna(subset=["timestamp"])
        dropped = initial_rows - len(df)
        
        if dropped > 0:
            logger.warning(f"⚠️ Eliminadas {dropped} filas con timestamp inválido")
        
        return df


class DataExtractor:
    """Extractor principal de datos meteorológicos."""
    
    def __init__(self, skip_rows: int = 9, delimiter: str = ";"):
        """
        Inicializa el extractor.
        
        Args:
            skip_rows: Filas de metadatos a saltar
            delimiter: Delimitador de archivos
        """
        self.reader = FileReader(skip_rows, delimiter)
        self.config = config
    
    def find_raw_files(self, extensions: List[str] = None) -> List[Path]:
        """
        Busca archivos en el directorio raw.
        
        Args:
            extensions: Lista de extensiones (default: ['.csv', '.tsv', '.txt'])
        
        Returns:
            Lista de archivos encontrados
        """
        if extensions is None:
            extensions = ['.csv', '.tsv', '.txt']
        
        raw_dir = self.config.raw_data_dir
        files = []
        
        for ext in extensions:
            files.extend(raw_dir.glob(f"*{ext}"))
        
        logger.info(f"📁 Encontrados {len(files)} archivos en {raw_dir}")
        return files
    
    def extract_all(self) -> Dict[str, pd.DataFrame]:
        """
        Extrae todos los archivos del directorio raw.
        
        Returns:
            Diccionario {nombre_variable: DataFrame}
        """
        logger.info("="*60)
        logger.info("EXTRACCIÓN DE DATOS (EXTRACT)")
        logger.info("="*60)
        
        files = self.find_raw_files()
        
        if not files:
            raise FileNotFoundError(f"No hay archivos en {self.config.raw_data_dir}")
        
        extracted_data = {}
        
        for file_path in files:
            var_name = identify_variable_from_filename(file_path.stem)
            
            if var_name is None:
                logger.warning(f"⏭️ Saltando: {file_path.name}")
                continue
            
            try:
                df = self.reader.read_file(file_path, var_name)
                extracted_data[var_name] = df
            except Exception as e:
                logger.error(f"❌ Error en {file_path.name}: {e}")
                continue
        
        if not extracted_data:
            raise ValueError("No se pudo extraer ningún archivo")
        
        logger.info(f"✅ Extraídas {len(extracted_data)} variables")
        logger.info("="*60)
        
        return extracted_data
    
    def extract_file(self, file_path: Path, var_name: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
        """
        Extrae un archivo específico.
        
        Args:
            file_path: Ruta del archivo
            var_name: Nombre de variable (si None, se auto-detecta)
        
        Returns:
            Tupla (nombre_variable, DataFrame)
        """
        if var_name is None:
            var_name = identify_variable_from_filename(file_path.stem)
            if var_name is None:
                raise ValueError(f"No se pudo identificar variable para {file_path.name}")
        
        df = self.reader.read_file(file_path, var_name)
        return var_name, df
