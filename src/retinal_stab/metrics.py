"""Metrics for quantifying motion stability."""
from __future__ import annotations

import cv2
import numpy as np

from .utils import chunk_pairs


def stability_index(frames: list[np.ndarray]) -> float:
    """Compute the mean optical-flow magnitude between consecutive frames."""
    if len(frames) < 2:
        return 0.0
    magnitudes: list[float] = []
    for prev, curr in chunk_pairs(frames):
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.linalg.norm(flow, axis=2)
        magnitudes.append(float(np.mean(mag)))
    return float(np.mean(magnitudes))


def feature_drift(trajectories: np.ndarray) -> float:
    """Aggregate the variance of consecutive transform deltas."""
    if len(trajectories) < 2:
        return 0.0
    deltas = np.diff(trajectories, axis=0)
    return float(np.sum(np.var(deltas, axis=0)))
