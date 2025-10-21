"""Módulo de diagnósticos y recomendaciones."""

from .rules import DiagnosticEngine, AlertLevel, DiagnosticResult
from .recommender import Recommender, diagnose

__all__ = [
    "DiagnosticEngine",
    "AlertLevel", 
    "DiagnosticResult",
    "Recommender",
    "diagnose"
]


