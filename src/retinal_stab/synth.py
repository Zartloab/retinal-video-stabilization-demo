"""Synthetic retinal video generator for reproducible demos."""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_dir


def _draw_vessels(canvas: np.ndarray, center: tuple[int, int]) -> None:
    """Draw branching vessels emanating from the optic disc."""
    h, w = canvas.shape[:2]
    num_branches = 5
    for _ in range(num_branches):
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(int(0.2 * w), int(0.4 * w))
        points = [center]
        for step in range(1, 6):
            theta = angle + random.uniform(-0.3, 0.3) * step
            radius = length * (step / 5)
            x = int(center[0] + radius * math.cos(theta))
            y = int(center[1] + radius * math.sin(theta))
            points.append((x, y))
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (80, 30, 30), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (120, 50, 50), thickness=1, lineType=cv2.LINE_AA)


def make_synthetic_retina_video(out_path: str, length_sec: int = 5, fps: int = 30, size: tuple[int, int] = (640, 640)) -> None:
    """Render a jittery retina-like video with occasional blinks."""
    ensure_dir(Path(out_path).parent)
    width, height = size
    center = (width // 2, height // 2)

    base = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(base, center, int(0.48 * min(size)), (20, 20, 20), thickness=-1)
    cv2.circle(base, center, int(0.45 * min(size)), (35, 35, 35), thickness=-1)
    cv2.circle(base, center, int(0.18 * min(size)), (90, 90, 90), thickness=-1)
    _draw_vessels(base, center)

    total_frames = length_sec * fps
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer at {out_path}")

    for idx in range(total_frames):
        angle = random.gauss(0.0, 0.4)
        dx = random.gauss(0.0, 2.0)
        dy = random.gauss(0.0, 2.0)
        scale = 1.0 + random.uniform(-0.01, 0.01)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        frame = cv2.warpAffine(base, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        flicker = 0.85 + 0.2 * random.random()
        noisy = frame.astype(np.float32) * flicker
        noise = np.random.normal(0, 12, frame.shape).astype(np.float32)
        noisy += noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        if random.random() < 0.03:
            blink_strength = random.uniform(0.2, 0.5)
            noisy = (noisy.astype(np.float32) * blink_strength).astype(np.uint8)

        writer.write(noisy)

    writer.release()
