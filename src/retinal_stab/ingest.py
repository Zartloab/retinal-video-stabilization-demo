"""Video I/O helpers that hide OpenCV boilerplate."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def iter_video_frames(path: str) -> Iterator[np.ndarray]:
    """Yield frames from a video file as BGR NumPy arrays."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            yield frame
    finally:
        capture.release()


def video_writer_like(in_path: str, out_path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    """Create a video writer mirroring the input container settings when possible."""
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # We prefer mp4v for broad compatibility; fall back to MJPG if the former fails.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    if writer.isOpened():
        return writer
    writer.release()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {out_path}")
    return writer


def get_video_meta(path: str) -> dict:
    """Return width, height, fps, and frame count for a video."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to read metadata from {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return {"width": width, "height": height, "fps": fps, "frame_count": frame_count}
