"""General-purpose utilities shared across the pipeline."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and OpenCV-random consumers for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import cv2

        cv2.setRNGSeed(seed)
    except ImportError:
        # OpenCV might be absent in certain documentation contexts; ignore quietly.
        pass


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create the directory if it does not exist and return the `Path` object."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: str | os.PathLike[str]) -> dict:
    """Load a YAML configuration file into a plain dictionary."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def chunk_pairs(items: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return consecutive frame pairs, useful for optical-flow metrics."""
    return [(items[i], items[i + 1]) for i in range(len(items) - 1)]
