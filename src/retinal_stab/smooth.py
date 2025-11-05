"""Temporal smoothing of camera motion parameters."""
from __future__ import annotations

import math

import cv2
import numpy as np


_EPS = 1e-9


def motion_to_params(M: np.ndarray, model: str) -> np.ndarray:
    """Decompose a 2x3 motion matrix into additive parameters for smoothing."""
    model = model.lower()
    if model not in {"similarity", "affine", "euclidean", "lk_similarity"}:
        raise ValueError(f"Unsupported motion model: {model}")

    retval, R, t, scales, shear = cv2.decomposeAffine2D(np.asarray(M, dtype=np.float32))
    if retval == 0:
        raise ValueError("Affine decomposition failed")

    tx, ty = [float(v) for v in t.flatten()[:2]]
    angle = math.atan2(float(R[1, 0]), float(R[0, 0]))
    sx, sy = [float(v) for v in scales.flatten()[:2]]
    shear_val = float(shear) if np.ndim(shear) == 0 else float(np.asarray(shear).reshape(-1)[0])

    if model == "euclidean":
        return np.array([tx, ty, angle], dtype=np.float32)

    if model in {"similarity", "lk_similarity"}:
        scale = max((sx + sy) * 0.5, _EPS)
        log_scale = math.log(scale)
        return np.array([tx, ty, angle, log_scale], dtype=np.float32)

    # full affine (6 DOF)
    sx = max(abs(sx), _EPS)
    sy = max(abs(sy), _EPS)
    log_sx = math.log(sx)
    log_sy = math.log(sy)
    return np.array([tx, ty, angle, log_sx, log_sy, shear_val], dtype=np.float32)


def params_to_motion(params: np.ndarray, model: str) -> np.ndarray:
    """Recompose a motion matrix from smoothed parameters."""
    model = model.lower()
    if model not in {"similarity", "affine", "euclidean", "lk_similarity"}:
        raise ValueError(f"Unsupported motion model: {model}")

    if model == "euclidean":
        tx, ty, angle = [float(v) for v in params[:3]]
        scale_x = scale_y = 1.0
        shear_val = 0.0
    elif model in {"similarity", "lk_similarity"}:
        tx, ty, angle, log_scale = [float(v) for v in params[:4]]
        scale = math.exp(log_scale)
        scale_x = scale_y = scale
        shear_val = 0.0
    else:  # affine
        tx, ty, angle, log_sx, log_sy, shear_val = [float(v) for v in params[:6]]
        scale_x = math.exp(log_sx)
        scale_y = math.exp(log_sy)

    matrix = _compose_affine(tx, ty, angle, scale_x, scale_y, shear_val)
    return matrix.astype(np.float32)


def moving_average(params: np.ndarray, win: int) -> np.ndarray:
    """Apply a centred moving average to stabilise trajectories."""
    if win <= 1:
        return params
    pad = win // 2
    padded = np.pad(params, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win) / win
    smoothed = np.vstack(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(params.shape[1])]
    ).T
    return smoothed.astype(np.float32)


def _compose_affine(
    tx: float,
    ty: float,
    angle: float,
    scale_x: float,
    scale_y: float,
    shear: float,
) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    shear_matrix = np.array([[scale_x, shear * scale_x], [0.0, scale_y]], dtype=np.float32)
    linear = rotation @ shear_matrix
    matrix = np.array(
        [[linear[0, 0], linear[0, 1], tx], [linear[1, 0], linear[1, 1], ty]],
        dtype=np.float32,
    )
    return matrix
