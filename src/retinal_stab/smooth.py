"""Temporal smoothing of camera motion parameters."""
from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np


def affine_to_params(M: np.ndarray) -> np.ndarray:
    """Convert a 2x3 affine matrix into [dx, dy, da, log_s] parameters."""
    dx = float(M[0, 2])
    dy = float(M[1, 2])

    # Extract rotation before scale so that we can smooth each component independently.
    da = math.atan2(M[1, 0], M[0, 0])

    scale = math.sqrt(float(M[0, 0]) ** 2 + float(M[1, 0]) ** 2)
    if scale <= 0.0:
        scale = 1.0
    log_s = math.log(scale)

    return np.array([dx, dy, da, log_s], dtype=np.float32)


def params_to_affine(params: np.ndarray) -> np.ndarray:
    """Reconstruct a 2x3 affine transform from [dx, dy, da, log_s] parameters."""
    dx = float(params[0])
    dy = float(params[1])
    da = float(params[2])
    if len(params) >= 4:
        scale = math.exp(float(params[3]))
    else:
        scale = 1.0

    cos_a = math.cos(da) * scale
    sin_a = math.sin(da) * scale
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


def smooth_param_sequence(params: np.ndarray, win: int) -> np.ndarray:
    """Center-smooth a sequence of affine motion parameters.

    Parameters are assumed to be incremental transforms per frame. We convert to a
    cumulative trajectory, smooth it, and then recover new incremental steps so
    that the resulting transforms follow the stabilised trajectory.
    """

    if len(params) == 0:
        return params

    trajectory = np.cumsum(params, axis=0)
    smoothed_traj = moving_average(trajectory, win)

    # Preserve the first pose to avoid drifting the sequence away from the
    # starting frame when the window is wider than the available observations.
    smoothed_traj[0] = trajectory[0]

    smoothed_params = np.empty_like(params)
    smoothed_params[0] = smoothed_traj[0]
    if len(params) > 1:
        smoothed_params[1:] = smoothed_traj[1:] - smoothed_traj[:-1]

    return smoothed_params.astype(np.float32)
