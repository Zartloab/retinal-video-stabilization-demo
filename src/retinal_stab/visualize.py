"""Utility functions for presenting stabilisation results."""
from __future__ import annotations

import cv2
import numpy as np

from pathlib import Path


def side_by_side(before_path: str, after_path: str, out_path: str, label_left: str = "Before", label_right: str = "After") -> None:
    """Create a labelled side-by-side comparison video."""
    before_cap = cv2.VideoCapture(before_path)
    after_cap = cv2.VideoCapture(after_path)
    if not before_cap.isOpened() or not after_cap.isOpened():
        raise RuntimeError("Could not open input videos")

    fps = after_cap.get(cv2.CAP_PROP_FPS) or before_cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames: list[np.ndarray] = []
    while True:
        ret1, frame1 = before_cap.read()
        ret2, frame2 = after_cap.read()
        if not ret1 or not ret2:
            break
        if frame1.shape[0] != frame2.shape[0]:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        combined = np.hstack([frame1, frame2])
        cv2.putText(combined, label_left, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(combined, label_right, (frame1.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        frames.append(combined)

    before_cap.release()
    after_cap.release()

    if not frames:
        raise RuntimeError("No overlapping frames to compose")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video at {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
