"""Pose estimation utilities for retinal video stabilisation."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


def keypoints_to_points(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert matched keypoints into floating-point coordinate arrays."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def estimate_motion_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    model: str,
    ransac_thresh: float,
    *,
    max_iters: int = 2000,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a motion model using RANSAC from point correspondences."""
    if len(pts1) < 3:
        raise ValueError("Not enough correspondences for motion estimation")

    model = model.lower()
    if model not in {"similarity", "affine", "euclidean"}:
        raise ValueError(f"Unsupported motion model: {model}")

    estimator = cv2.estimateAffinePartial2D if model != "affine" else cv2.estimateAffine2D
    matrix, inliers = estimator(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_thresh),
        maxIters=int(max_iters),
        confidence=float(confidence),
    )
    if matrix is None:
        raise RuntimeError("Motion estimation failed")

    matrix = np.asarray(matrix, dtype=np.float32)
    if model == "euclidean":
        matrix = _enforce_euclidean(matrix)
    elif model == "similarity":
        matrix = _enforce_similarity(matrix)

    if inliers is None:
        inliers = np.zeros((len(pts1), 1), dtype=np.uint8)
    inliers = np.asarray(inliers, dtype=np.uint8).ravel()
    return matrix, inliers


def track_lucas_kanade(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    max_corners: int = 600,
    quality: float = 0.01,
    min_distance: float = 7.0,
    block_size: int = 7,
    win_size: Tuple[int, int] = (21, 21),
    max_level: int = 3,
    criteria: Optional[Tuple[int, int, float]] = None,
    max_error: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Track Shi-Tomasi features with pyramidal Lucas-Kanade optical flow."""

    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=int(max_corners),
        qualityLevel=float(quality),
        minDistance=float(min_distance),
        blockSize=int(block_size),
        mask=mask,
    )
    if features is None or len(features) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    lk_criteria = criteria or (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        features,
        None,
        winSize=tuple(int(v) for v in win_size),
        maxLevel=int(max_level),
        criteria=lk_criteria,
    )

    if next_pts is None or status is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    status = status.reshape(-1).astype(bool)
    if err is not None:
        err = err.reshape(-1)
        status &= err <= float(max_error)

    prev_good = features.reshape(-1, 2)[status]
    next_good = next_pts.reshape(-1, 2)[status]

    if prev_good.size == 0 or next_good.size == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    return prev_good.astype(np.float32), next_good.astype(np.float32)


def _enforce_similarity(matrix: np.ndarray) -> np.ndarray:
    """Project a 2x3 matrix onto the Similarity group (uniform scale + rotation)."""
    angle = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    scale = math.sqrt(max(matrix[0, 0] ** 2 + matrix[1, 0] ** 2, 1e-12))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    result = matrix.copy()
    result[0, 0] = float(scale * cos_a)
    result[0, 1] = float(-scale * sin_a)
    result[1, 0] = float(scale * sin_a)
    result[1, 1] = float(scale * cos_a)
    return result.astype(np.float32)


def _enforce_euclidean(matrix: np.ndarray) -> np.ndarray:
    """Project a 2x3 matrix onto the Euclidean motion group (unit scale)."""
    angle = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    result = matrix.copy()
    result[0, 0] = float(cos_a)
    result[0, 1] = float(-sin_a)
    result[1, 0] = float(sin_a)
    result[1, 1] = float(cos_a)
    return result.astype(np.float32)
