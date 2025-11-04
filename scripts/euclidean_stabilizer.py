"""Stabilise handheld retinal videos using Euclidean motion and trajectory smoothing."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class FrameBatch:
    """Container for video frames and metadata used during stabilisation."""

    frames: List[np.ndarray]
    fps: float
    width: int
    height: int


def read_video(video_path: str) -> FrameBatch:
    """Load all frames from ``video_path`` and capture basic metadata.

    Parameters
    ----------
    video_path: str
        Location of the MP4/AVI file to ingest.

    Returns
    -------
    FrameBatch
        Frames (BGR), frames-per-second, width, and height.

    Notes
    -----
    We read the entire clip into memory to simplify later trajectory smoothing.
    For short interview demos this is acceptable and keeps the pipeline easy to follow.
    """

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: List[np.ndarray] = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)

    capture.release()
    if not frames:
        raise RuntimeError("Input video contains no frames")

    return FrameBatch(frames=frames, fps=fps, width=width, height=height)


def _extract_green_channel(frame: np.ndarray) -> np.ndarray:
    """Return the green channel which maximises vessel contrast."""

    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("Expected a colour frame with at least three channels")
    return frame[:, :, 1]


def get_motion_estimation(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    feature_params: dict[str, float | int],
    lk_params: dict[str, object],
) -> Tuple[np.ndarray, int, int, float]:
    """Estimate Euclidean motion between ``prev_frame`` and ``curr_frame``.

    The function performs the following steps:
      1. Detect high-quality corner features (Shi-Tomasi) on the previous frame.
      2. Track those features into the current frame via Lucas-Kanade optical flow.
      3. Reject outlier correspondences using RANSAC while solving for a
         similarity transform (rotation + translation + uniform scale).

    This combination is robust to noise yet keeps vascular geometry intact because
    similarity transforms cannot introduce shear or projective distortions.

    Returns
    -------
    M: np.ndarray
        A 2x3 affine matrix encoding the Euclidean transform.
    inliers: int
        Number of correspondences retained by RANSAC.
    total: int
        Total tracked correspondences prior to RANSAC.
    scale: float
        Estimated uniform scale factor (used later during smoothing).
    """

    prev_green = _extract_green_channel(prev_frame)
    curr_green = _extract_green_channel(curr_frame)

    prev_gray = prev_green.astype(np.float32)
    curr_gray = curr_green.astype(np.float32)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if prev_pts is None:
        # Without features we fallback to an identity transform and report zeros.
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return identity, 0, 0, 1.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    if curr_pts is None or status is None:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return identity, 0, int(prev_pts.shape[0]), 1.0

    status = status.reshape(-1)
    matched_prev = prev_pts[status == 1]
    matched_curr = curr_pts[status == 1]

    total = matched_prev.shape[0]
    if total < 4:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return identity, 0, int(total), 1.0

    # ``estimateAffinePartial2D`` enforces a similarity transform (rotation,
    # translation, uniform scale) when ``fullAffine=False``. We enable RANSAC so
    # that spurious tracks (e.g. from blinking frames or specular highlights)
    # cannot skew the solution.
    M, inlier_mask = cv2.estimateAffinePartial2D(
        matched_prev,
        matched_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )

    if M is None:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return identity, 0, int(total), 1.0

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else total

    # Extract the uniform scale encoded in the rotation portion of the matrix.
    scale = float(math.hypot(M[0, 0], M[0, 1]))

    return M.astype(np.float32), inliers, int(total), scale


def smooth_trajectory(trajectory: np.ndarray, window_size: int = 30) -> np.ndarray:
    """Apply a sliding-window mean filter to ``trajectory``.

    Smoothing attenuates the high-frequency jitter induced by handheld capture
    while keeping slow movements (e.g. deliberate panning) intact.
    """

    if trajectory.ndim != 2:
        raise ValueError("Trajectory must be a 2D array")
    if trajectory.shape[0] == 0:
        return trajectory

    half_window = max(1, window_size // 2)
    smoothed = np.empty_like(trajectory)
    for idx in range(trajectory.shape[0]):
        start = max(0, idx - half_window)
        end = min(trajectory.shape[0], idx + half_window + 1)
        smoothed[idx] = trajectory[start:end].mean(axis=0)
    return smoothed


def _matrix_from_params(dx: float, dy: float, da: float, scale: float) -> np.ndarray:
    """Compose a 2x3 affine matrix from Euclidean parameters."""

    cos_a = math.cos(da)
    sin_a = math.sin(da)
    return np.array(
        [
            [scale * cos_a, -scale * sin_a, dx],
            [scale * sin_a, scale * cos_a, dy],
        ],
        dtype=np.float32,
    )


def stabilize_video(
    input_path: str,
    output_path: str,
    window_size: int = 30,
) -> None:
    """Run the full Euclidean stabilization pipeline on ``input_path``.

    Detailed procedure:
      * Load frames and metadata.
      * Estimate noisy per-frame motion using feature tracking + RANSAC.
      * Accumulate the motion into a trajectory of translation, rotation, and scale.
      * Smooth the trajectory to suppress high-frequency jitter.
      * Compute per-frame compensation transforms and warp frames accordingly.
      * Write the stabilised frames to ``output_path``.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive to perform smoothing")

    batch = read_video(input_path)
    frames = batch.frames

    feature_params = {
        "maxCorners": 1200,
        "qualityLevel": 0.01,
        "minDistance": 7,
        "blockSize": 7,
    }
    lk_params = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }

    transforms: List[dict[str, float]] = []
    trajectory: List[List[float]] = []

    for idx in range(1, len(frames)):
        prev_frame = frames[idx - 1]
        curr_frame = frames[idx]
        M, inliers, total, scale = get_motion_estimation(prev_frame, curr_frame, feature_params, lk_params)

        dx = float(M[0, 2])
        dy = float(M[1, 2])
        da = float(math.atan2(M[1, 0], M[0, 0]))
        ds = float(math.log(scale + 1e-8))

        transforms.append({"dx": dx, "dy": dy, "da": da, "ds": ds, "inliers": inliers, "total": total})

        if not trajectory:
            trajectory.append([dx, dy, da, ds])
        else:
            prev = trajectory[-1]
            trajectory.append([prev[0] + dx, prev[1] + dy, prev[2] + da, prev[3] + ds])

    if not transforms:
        raise RuntimeError("Video must contain at least two frames for stabilisation")

    trajectory_array = np.array(trajectory, dtype=np.float32)
    smoothed_trajectory = smooth_trajectory(trajectory_array, window_size=window_size)

    # The difference between smoothed and raw trajectories tells us how much
    # additional motion to cancel out on each frame.
    compensation = smoothed_trajectory - trajectory_array

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, batch.fps, (batch.width, batch.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer at {output_path}")

    writer.write(frames[0])  # First frame is used as the reference.

    for idx in range(1, len(frames)):
        frame = frames[idx]
        transform = transforms[idx - 1]
        diff = compensation[idx - 1]

        dx = transform["dx"] + diff[0]
        dy = transform["dy"] + diff[1]
        da = transform["da"] + diff[2]
        ds = transform["ds"] + diff[3]

        scale = math.exp(ds)
        M = _matrix_from_params(dx, dy, da, scale)

        stabilised = cv2.warpAffine(
            frame,
            M,
            (batch.width, batch.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        writer.write(stabilised)

    writer.release()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilise a retinal fundus video using Euclidean motion.")
    parser.add_argument("input", help="Path to the shaky input video")
    parser.add_argument("output", help="Path where the stabilised video will be written")
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Sliding window (frames) for trajectory smoothing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stabilize_video(args.input, args.output, window_size=args.window_size)
    print(f"Stabilised video written to {args.output}")


if __name__ == "__main__":
    main()
