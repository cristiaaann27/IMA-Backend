"""Utilidades de sanitización de datos post-escalado."""

import numpy as np


def sanitize_scaled(X: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Reemplaza NaN e Inf producidos por escalado en datos numéricos.

    Tras aplicar StandardScaler, columnas con varianza cero o valores
    extremos pueden producir inf/nan. Esta función los reemplaza por
    un valor seguro.

    Args:
        X: Array numérico (puede ser 1D, 2D o 3D).
        fill_value: Valor con el que reemplazar NaN/Inf (default: 0.0).

    Returns:
        Array con NaN/Inf reemplazados.
    """
    X_clean = np.copy(X)
    mask = ~np.isfinite(X_clean)
    if mask.any():
        X_clean[mask] = fill_value
    return X_clean
