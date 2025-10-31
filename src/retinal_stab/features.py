"""Feature detection and matching utilities."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def detect_and_describe(gray: np.ndarray, method: str, n_features: int) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect keypoints and compute descriptors using ORB or SIFT."""
    method = method.upper()
    if method == "SIFT":
        detector = cv2.SIFT_create(n_features)
    else:
        detector = cv2.ORB_create(nfeatures=n_features, edgeThreshold=15, patchSize=31)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.zeros((0, detector.descriptorSize()), dtype=np.uint8)
    return keypoints, descriptors


def match_descriptors(des1: np.ndarray, des2: np.ndarray, method: str, ratio: float) -> list[cv2.DMatch]:
    """Match descriptors with a Lowe-style ratio test."""
    method = method.upper()
    if method == "SIFT":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if des2.dtype != np.float32:
            des2 = des2.astype(np.float32)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches: list[cv2.DMatch] = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches
