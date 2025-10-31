"""Preprocessing operations tailored for retinal footage."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to grayscale with float32 precision."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Boost local contrast to highlight low-light vessels."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray, mode: str) -> np.ndarray:
    """Reduce sensor noise while preserving edges based on requested mode."""
    if mode == "median":
        return cv2.medianBlur(gray, 5)
    if mode == "bilateral":
        return cv2.bilateralFilter(gray, 9, 75, 75)
    return gray


def retina_mask(h: int, w: int, inner_radius_pct: float) -> np.ndarray:
    """Create a circular mask centred in the frame to suppress eyelid borders."""
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(center) * inner_radius_pct)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    # Provide a gentle feathering to avoid sharp edges in the mask.
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask
