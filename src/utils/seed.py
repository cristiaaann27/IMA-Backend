"""Utilidades de reproducibilidad para fijar semillas globales."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """
    Fija semillas globales para reproducibilidad.

    Afecta: random, numpy, torch (CPU y CUDA), y cuDNN.

    Args:
        seed: Valor de la semilla (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Determinismo en cuDNN (puede reducir rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
