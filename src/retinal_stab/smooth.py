"""Temporal smoothing of camera motion parameters."""
from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np


def affine_to_params(M: np.ndarray) -> np.ndarray:
    """Convert a 2x3 affine matrix into [dx, dy, da] parameters."""
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    da = math.atan2(M[1, 0], M[0, 0])
    return np.array([dx, dy, da], dtype=np.float32)


def params_to_affine(params: np.ndarray) -> np.ndarray:
    """Reconstruct a 2x3 affine transform from [dx, dy, da] parameters."""
    dx, dy, da = params
    cos_a = math.cos(float(da))
    sin_a = math.sin(float(da))
    matrix = np.array([[cos_a, -sin_a, dx], [sin_a, cos_a, dy]], dtype=np.float32)
    return matrix


def accumulate_params(Ms: list[np.ndarray]) -> np.ndarray:
    """Integrate per-frame motion into an absolute trajectory."""
    params = np.array([affine_to_params(M) for M in Ms], dtype=np.float32)
    trajectory = np.cumsum(params, axis=0)
    return trajectory


def moving_average(params: np.ndarray, win: int) -> np.ndarray:
    """Apply a centred moving average to stabilise trajectories."""
    if win <= 1:
        return params
    pad = win // 2
    padded = np.pad(params, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win) / win
    smoothed = np.vstack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(params.shape[1])]).T
    return smoothed.astype(np.float32)


def compose_affines_from_params(smoothed_params: np.ndarray) -> list[np.ndarray]:
    """Convert smoothed incremental motion back to affine transforms."""
    matrices = [params_to_affine(p) for p in smoothed_params]
    return matrices
