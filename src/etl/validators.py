"""Validadores de datos meteorológicos."""

import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, Check, Index
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Esquema Pandera para datos meteorológicos crudos
# ---------------------------------------------------------------------------
WeatherDataSchema = pa.DataFrameSchema(
    columns={
        "timestamp": Column(pa.DateTime, nullable=False, coerce=True),
        "precip_mm_hr": Column(float, [
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(200),
        ], nullable=True, coerce=True, required=False),
        "rh_2m_pct": Column(float, [
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(100),
        ], nullable=True, coerce=True, required=False),
        "temp_2m_c": Column(float, [
            Check.greater_than_or_equal_to(-40),
            Check.less_than_or_equal_to(60),
        ], nullable=True, coerce=True, required=False),
        "wind_speed_2m_ms": Column(float, [
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(100),
        ], nullable=True, coerce=True, required=False),
        "wind_dir_2m_deg": Column(float, [
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(360),
        ], nullable=True, coerce=True, required=False),
        "wind_speed_10m_ms": Column(float, [
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(100),
        ], nullable=True, coerce=True, required=False),
        "wind_dir_10m_deg": Column(float, [
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(360),
        ], nullable=True, coerce=True, required=False),
    },
    # Solo validar columnas que existan en el DataFrame
    strict=False,
    coerce=True,
)


@dataclass
class ValidationResult:
    """Resultado de validación."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fixed_rows: int = 0
    
    def add_error(self, message: str) -> None:
        """Añade un error."""
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str) -> None:
        """Añade una advertencia."""
        self.warnings.append(message)


class DataValidator:
    """Validador de datos meteorológicos con rangos razonables."""
    
    # Rangos válidos para cada variable (min, max)
    VALID_RANGES = {
        "precip_mm_hr": (0, 200),           # Precipitación: 0-200 mm/hr
        "rh_2m_pct": (0, 100),              # Humedad relativa: 0-100%
        "temp_2m_c": (-40, 60),             # Temperatura: -40 a 60°C
        "wind_speed_2m_ms": (0, 100),       # Velocidad viento: 0-100 m/s
        "wind_dir_2m_deg": (0, 360),        # Dirección viento: 0-360°
        "wind_speed_10m_ms": (0, 100),      # Velocidad viento 10m: 0-100 m/s
        "wind_dir_10m_deg": (0, 360),       # Dirección viento 10m: 0-360°
    }
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
        """
        Valida un DataFrame completo.
        
        Args:
            df: DataFrame a validar
        
        Returns:
            ValidationResult con errores y warnings
        """
        errors = []
        warnings = []
        fixed_rows = 0
        
        # --- Validación Pandera (esquema tipado) ---
        try:
            WeatherDataSchema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as schema_err:
            for _, row in schema_err.failure_cases.iterrows():
                col = row.get("column", "unknown")
                check = row.get("check", "unknown")
                warnings.append(f"Pandera: columna '{col}' falló check '{check}'")
        except Exception as e:
            warnings.append(f"Pandera: error inesperado durante validación: {e}")
        
        # --- Validaciones manuales complementarias ---
        # Validar que existe timestamp
        if "timestamp" not in df.columns:
            errors.append("Columna 'timestamp' no encontrada")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Validar tipos de datos
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception as e:
                errors.append(f"No se pudo convertir 'timestamp' a datetime: {e}")
        
        # Verificar duplicados de timestamp
        duplicates = df["timestamp"].duplicated().sum()
        if duplicates > 0:
            warnings.append(f"Se encontraron {duplicates} timestamps duplicados")
        
        # Validar rangos de cada variable
        for var_name, (min_val, max_val) in DataValidator.VALID_RANGES.items():
            if var_name in df.columns:
                out_of_range = (
                    (df[var_name] < min_val) | (df[var_name] > max_val)
                ) & df[var_name].notna()
                
                if out_of_range.any():
                    count = out_of_range.sum()
                    warnings.append(
                        f"{var_name}: {count} valores fuera de rango "
                        f"[{min_val}, {max_val}]"
                    )
        
        # Verificar valores nulos
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0 and col != "timestamp":
                pct = (count / len(df)) * 100
                warnings.append(f"{col}: {count} valores nulos ({pct:.2f}%)")
        
        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            fixed_rows=fixed_rows
        )
    
    @staticmethod
    def check_temporal_consistency(df: pd.DataFrame) -> ValidationResult:
        """
        Verifica la consistencia temporal (orden, gaps).
        
        Args:
            df: DataFrame con columna 'timestamp'
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        if "timestamp" not in df.columns:
            errors.append("No se encontró columna 'timestamp'")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Verificar orden temporal
        if not df["timestamp"].is_monotonic_increasing:
            warnings.append("Timestamps no están ordenados cronológicamente")
        
        # Detectar gaps temporales (asumiendo frecuencia horaria)
        if len(df) > 1:
            time_diffs = df["timestamp"].diff()
            expected_freq = pd.Timedelta(hours=1)
            
            gaps = time_diffs[time_diffs > expected_freq]
            if len(gaps) > 0:
                warnings.append(
                    f"Se detectaron {len(gaps)} gaps temporales "
                    f"(mayor a 1 hora)"
                )
        
        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def clean_outliers(
        df: pd.DataFrame,
        method: str = "clip",
        std_threshold: float = 5.0
    ) -> Tuple[pd.DataFrame, int]:
        """
        Limpia outliers extremos.
        
        Args:
            df: DataFrame a limpiar
            method: 'clip' (recortar) o 'remove' (eliminar)
            std_threshold: Umbral de desviaciones estándar
        
        Returns:
            DataFrame limpio y número de valores modificados
        """
        df_clean = df.copy()
        modified = 0
        
        for var_name in DataValidator.VALID_RANGES.keys():
            if var_name not in df_clean.columns:
                continue
            
            col = df_clean[var_name]
            min_val, max_val = DataValidator.VALID_RANGES[var_name]
            
            if method == "clip":
                # Recortar a rangos válidos
                original = col.copy()
                df_clean[var_name] = col.clip(lower=min_val, upper=max_val)
                modified += (df_clean[var_name] != original).sum()
            
            elif method == "remove":
                # Marcar como NaN valores fuera de rango
                mask = (col < min_val) | (col > max_val)
                df_clean.loc[mask, var_name] = np.nan
                modified += mask.sum()
        
        return df_clean, modified


