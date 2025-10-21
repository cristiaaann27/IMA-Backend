"""Utilidades de entrada/salida."""

import shutil
from pathlib import Path
from typing import Union, List
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Asegura que un directorio existe, creándolo si es necesario.
    
    Args:
        path: Ruta del directorio
    
    Returns:
        Path object del directorio
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_files(source_pattern: Union[str, Path], dest_dir: Union[str, Path], 
               copy: bool = False) -> List[Path]:
    """
    Mueve o copia archivos que coincidan con un patrón a un directorio destino.
    
    Args:
        source_pattern: Patrón de búsqueda (puede incluir wildcards)
        dest_dir: Directorio destino
        copy: Si True, copia en lugar de mover
    
    Returns:
        Lista de archivos movidos/copiados
    """
    source_pattern = Path(source_pattern)
    dest_dir = ensure_dir(dest_dir)
    
    moved_files = []
    
    # Si es un patrón con wildcards
    if "*" in str(source_pattern):
        source_dir = source_pattern.parent
        pattern = source_pattern.name
        files = source_dir.glob(pattern)
    else:
        files = [source_pattern] if source_pattern.exists() else []
    
    for file_path in files:
        if file_path.is_file():
            dest_path = dest_dir / file_path.name
            
            if copy:
                shutil.copy2(file_path, dest_path)
            else:
                shutil.move(str(file_path), str(dest_path))
            
            moved_files.append(dest_path)
    
    return moved_files


def save_dataframe(df: pd.DataFrame, path: Union[str, Path], 
                   format: str = "parquet") -> Path:
    """
    Guarda un DataFrame en el formato especificado.
    
    Args:
        df: DataFrame a guardar
        path: Ruta de destino
        format: Formato ('parquet', 'csv', 'json')
    
    Returns:
        Path del archivo guardado
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    if format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    elif format == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Formato no soportado: {format}")
    
    return path


def load_dataframe(path: Union[str, Path], format: str = None) -> pd.DataFrame:
    """
    Carga un DataFrame desde un archivo.
    
    Args:
        path: Ruta del archivo
        format: Formato ('parquet', 'csv', 'json'). Si None, se infiere de la extensión
    
    Returns:
        DataFrame cargado
    """
    path = Path(path)
    
    if format is None:
        format = path.suffix.lstrip(".")
    
    if format == "parquet":
        return pd.read_parquet(path)
    elif format == "csv":
        return pd.read_csv(path)
    elif format == "json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Formato no soportado: {format}")


