"""Frame warping and full video stabilisation pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import warnings

import cv2
import numpy as np

from .estimate import estimate_affine, keypoints_to_points
from .features import detect_and_describe, match_descriptors
from .ingest import get_video_meta, iter_video_frames, video_writer_like
from .metrics import stability_index
from .preprocess import apply_clahe, denoise, retina_mask, to_gray
from .smooth import affine_to_params, moving_average, params_to_affine
from .utils import ensure_dir, set_seed


IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def warp_frame(frame: np.ndarray, M: np.ndarray, crop_pct: float) -> np.ndarray:
    """Warp a frame and crop borders to hide black regions."""
    h, w = frame.shape[:2]
    warped = cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    if crop_pct <= 0:
        return warped
    margin = int(min(h, w) * crop_pct)
    if margin <= 0 or (h - 2 * margin) <= 0 or (w - 2 * margin) <= 0:
        return warped
    cropped = warped[margin : h - margin, margin : w - margin]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def preprocess_for_features(frame: np.ndarray, cfg: dict, mask: np.ndarray | None) -> np.ndarray:
    gray = to_gray(frame)
    gray = denoise(gray, cfg.get("denoise", "median"))
    if cfg.get("clahe", True):
        gray = apply_clahe(gray)
    if mask is not None:
        gray = cv2.bitwise_and(gray, mask)
    return gray


def stabilise_video(in_path: str, out_path: str, cfg: dict) -> Dict[str, object]:
    """Run the end-to-end stabilisation pipeline on a single video."""
    set_seed(int(cfg.get("seed", 42)))
    ensure_dir(Path(out_path).parent)

    meta = get_video_meta(in_path)
    frames_color = list(iter_video_frames(in_path))
    if not frames_color:
        raise RuntimeError("Input video contains no frames")

    method = str(cfg.get("method", "affine")).lower()
    if method != "affine":
        warnings.warn("Only affine stabilisation is implemented; falling back to affine.")

    mask = retina_mask(meta["height"], meta["width"], float(cfg.get("mask_inner_radius_pct", 0.85)))

    preprocessed = [preprocess_for_features(frame, cfg, mask) for frame in frames_color]

    matrices: list[np.ndarray] = [IDENTITY.copy()]
    inliers_history: list[int] = [0]

    for idx in range(1, len(preprocessed)):
        prev_gray = preprocessed[idx - 1]
        curr_gray = preprocessed[idx]

        kp1, des1 = detect_and_describe(prev_gray, cfg.get("detector", "ORB"), int(cfg.get("n_features", 2000)))
        kp2, des2 = detect_and_describe(curr_gray, cfg.get("detector", "ORB"), int(cfg.get("n_features", 2000)))
        if len(des1) == 0 or len(des2) == 0:
            matrices.append(matrices[-1])
            inliers_history.append(0)
            continue

        matches = match_descriptors(des1, des2, cfg.get("detector", "ORB"), float(cfg.get("ratio_test", 0.75)))
        if len(matches) < int(cfg.get("min_inliers", 30)):
            matrices.append(matrices[-1])
            inliers_history.append(len(matches))
            continue

        pts1, pts2 = keypoints_to_points(kp1, kp2, matches)
        try:
            M, inliers = estimate_affine(pts1, pts2, float(cfg.get("ransac_reproj_thresh", 4.0)))
        except Exception:
            matrices.append(matrices[-1])
            inliers_history.append(0)
            continue

        inlier_count = int(np.sum(inliers))
        if inlier_count < int(cfg.get("min_inliers", 30)):
            matrices.append(matrices[-1])
            inliers_history.append(inlier_count)
            continue

        matrices.append(M.astype(np.float32))
        inliers_history.append(inlier_count)

    params = np.array([affine_to_params(M) for M in matrices], dtype=np.float32)
    trajectory = np.cumsum(params, axis=0)
    smoothed_traj = moving_average(trajectory, int(cfg.get("smooth_win", 45)))
    correction = smoothed_traj - trajectory
    smoothed_params = params + correction
    smoothed_matrices = [params_to_affine(p) for p in smoothed_params]

    writer = video_writer_like(in_path, out_path, meta["fps"], (meta["width"], meta["height"]))
    stabilised_frames: list[np.ndarray] = []
    for frame, M in zip(frames_color, smoothed_matrices):
        inv_M = cv2.invertAffineTransform(M.astype(np.float64)).astype(np.float32)
        warped = warp_frame(frame, inv_M, float(cfg.get("crop_pct", 0.06)))
        writer.write(warped)
        stabilised_frames.append(warped)
    writer.release()

    stability_before = stability_index(frames_color)
    stability_after = stability_index(stabilised_frames)

    return {
        "frame_count": len(frames_color),
        "avg_inliers": float(np.mean(inliers_history[1:]) if len(inliers_history) > 1 else 0.0),
        "inliers": inliers_history,
        "correction_norm": np.linalg.norm(correction, axis=1).tolist(),
        "stability_before": stability_before,
        "stability_after": stability_after,
        "output_path": out_path,
    }
