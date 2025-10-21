"""Módulo de ingeniería de características.

Este módulo proporciona herramientas para crear features a partir de datos procesados.

Uso rápido (legacy):
    >>> from src.features import build_features
    >>> df = build_features()

Uso modular (recomendado):
    >>> from src.features import FeaturePipeline
    >>> pipeline = FeaturePipeline()
    >>> df_features = pipeline.transform(df)
"""

# Pipeline modular (recomendado)
from .feature_engineering import (
    FeaturePipeline,
    TemporalFeatureCreator,
    LagFeatureCreator,
    InteractionFeatureCreator,
    BaseFeatureCreator
)

# Funciones legacy (retrocompatibilidad)
from .build_features import (
    build_features,
    create_temporal_features,
    create_lag_features,
    create_interaction_features,
    filter_wind_variables
)

__all__ = [
    # Pipeline modular
    "FeaturePipeline",
    "TemporalFeatureCreator",
    "LagFeatureCreator",
    "InteractionFeatureCreator",
    "BaseFeatureCreator",
    
    # Legacy
    "build_features",
    "create_temporal_features",
    "create_lag_features",
    "create_interaction_features",
    "filter_wind_variables"
]


