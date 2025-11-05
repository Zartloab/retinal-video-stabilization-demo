"""Frame warping and full video stabilisation pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .estimate import estimate_motion_matrix, keypoints_to_points, track_lucas_kanade
from .features import detect_and_describe, match_descriptors
from .ingest import get_video_meta, iter_video_frames, video_writer_like
from .metrics import stability_index
from .preprocess import apply_clahe, denoise, retina_mask, to_gray
from .smooth import motion_to_params, moving_average, params_to_motion
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

    method = str(cfg.get("method", "similarity")).lower()
    valid_methods = {"similarity", "affine", "euclidean", "lk_similarity"}
    if method not in valid_methods:
        raise ValueError(f"Unknown stabilisation method: {method}")

    estimation_model = "similarity" if method == "lk_similarity" else method
    detector_name = str(cfg.get("detector", "ORB"))
    n_features = int(cfg.get("n_features", 2000))
    ratio_test = float(cfg.get("ratio_test", 0.75))
    min_inliers = int(cfg.get("min_inliers", 30))
    ransac_thresh = float(cfg.get("ransac_reproj_thresh", 4.0))

    lk_win_size_cfg = cfg.get("lk_win_size", 21)
    if isinstance(lk_win_size_cfg, (list, tuple)) and len(lk_win_size_cfg) == 2:
        lk_win_size = (int(lk_win_size_cfg[0]), int(lk_win_size_cfg[1]))
    else:
        lk_win_dim = int(lk_win_size_cfg)
        lk_win_size = (lk_win_dim, lk_win_dim)
    lk_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(cfg.get("lk_max_iterations", 30)),
        float(cfg.get("lk_epsilon", 0.01)),
    )
    lk_settings = {
        "max_corners": int(cfg.get("lk_max_corners", 600)),
        "quality": float(cfg.get("lk_quality", 0.01)),
        "min_distance": float(cfg.get("lk_min_distance", 7.0)),
        "block_size": int(cfg.get("lk_block_size", 7)),
        "win_size": lk_win_size,
        "max_level": int(cfg.get("lk_max_level", 3)),
        "criteria": lk_criteria,
        "max_error": float(cfg.get("lk_max_error", 12.0)),
    }

    mask = retina_mask(meta["height"], meta["width"], float(cfg.get("mask_inner_radius_pct", 0.85)))

    preprocessed = [preprocess_for_features(frame, cfg, mask) for frame in frames_color]

    matrices: list[np.ndarray] = [IDENTITY.copy()]
    inliers_history: list[int] = [0]

    for idx in range(1, len(preprocessed)):
        prev_gray = preprocessed[idx - 1]
        curr_gray = preprocessed[idx]

        if method == "lk_similarity":
            pts1, pts2 = track_lucas_kanade(
                prev_gray,
                curr_gray,
                mask=mask,
                **lk_settings,
            )
            if len(pts1) < min_inliers:
                matrices.append(matrices[-1])
                inliers_history.append(len(pts1))
                continue
        else:
            kp1, des1 = detect_and_describe(prev_gray, detector_name, n_features)
            kp2, des2 = detect_and_describe(curr_gray, detector_name, n_features)
            if len(des1) == 0 or len(des2) == 0:
                matrices.append(matrices[-1])
                inliers_history.append(0)
                continue

            matches = match_descriptors(des1, des2, detector_name, ratio_test)
            if len(matches) < min_inliers:
                matrices.append(matrices[-1])
                inliers_history.append(len(matches))
                continue

            pts1, pts2 = keypoints_to_points(kp1, kp2, matches)

        try:
            M, inliers = estimate_motion_matrix(pts1, pts2, estimation_model, ransac_thresh)
        except Exception:
            matrices.append(matrices[-1])
            inliers_history.append(0)
            continue

        inlier_count = int(np.count_nonzero(inliers)) if inliers.size else len(pts1)
        if inlier_count < min_inliers:
            matrices.append(matrices[-1])
            inliers_history.append(inlier_count)
            continue

        matrices.append(M.astype(np.float32))
        inliers_history.append(inlier_count)

    smoothing_model = "similarity" if method == "lk_similarity" else method
    params = np.array([motion_to_params(M, smoothing_model) for M in matrices], dtype=np.float32)
    trajectory = np.cumsum(params, axis=0)
    smoothed_traj = moving_average(trajectory, int(cfg.get("smooth_win", 45)))
    correction = smoothed_traj - trajectory
    smoothed_params = params + correction
    smoothed_matrices = [params_to_motion(p, smoothing_model) for p in smoothed_params]

    writer = video_writer_like(in_path, out_path, meta["fps"], (meta["width"], meta["height"]))
    stabilised_frames: list[np.ndarray] = []
    for frame, M in zip(frames_color, smoothed_matrices):
        warped = warp_frame(frame, M, float(cfg.get("crop_pct", 0.06)))
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
