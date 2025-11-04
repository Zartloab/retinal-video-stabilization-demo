"""Pose estimation from feature correspondences."""
from __future__ import annotations

import cv2
import numpy as np


def keypoints_to_points(kp1: list[cv2.KeyPoint], kp2: list[cv2.KeyPoint], matches: list[cv2.DMatch]) -> tuple[np.ndarray, np.ndarray]:
    """Convert matched keypoints into floating-point coordinate arrays."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def estimate_affine(pts1: np.ndarray, pts2: np.ndarray, ransac_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an affine transform robustly using RANSAC."""
    if len(pts1) < 3:
        raise ValueError("Not enough correspondences for affine estimation")
    matrix, inliers = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.99,
    )
    if matrix is None:
        raise RuntimeError("Affine estimation failed")
    if inliers is None:
        inliers = np.zeros((len(pts1), 1), dtype=np.uint8)
    return matrix, inliers.ravel()
