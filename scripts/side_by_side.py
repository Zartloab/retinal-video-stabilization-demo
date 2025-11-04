"""Compose a side-by-side comparison video with simple labels."""
from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import cv2
import numpy as np


def read_video_frames(path: pathlib.Path) -> Tuple[float, list[np.ndarray]]:
    """Read all frames from a video file, returning fps and list of BGR frames."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():  # Extensive comment clarifying fallback
        raise RuntimeError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return fps, frames


def side_by_side_frames(before: list[np.ndarray], after: list[np.ndarray], label_left: str, label_right: str) -> list[np.ndarray]:
    """Stack frames horizontally and draw text labels on the top-left corners."""
    count = min(len(before), len(after))
    combined_frames: list[np.ndarray] = []
    for idx in range(count):
        left = before[idx]
        right = after[idx]
        # Harmonise frame heights to avoid mismatched stacking due to encoding quirks
        if left.shape[0] != right.shape[0]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))
        frame = np.hstack([left, right])
        cv2.putText(frame, label_left, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, label_right, (left.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        combined_frames.append(frame)
    return combined_frames


def write_video(out_path: pathlib.Path, fps: float, frame_size: tuple[int, int], frames: list[np.ndarray]) -> None:
    """Persist frames as an MP4 file using the H.264 codec when available."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer at {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a side-by-side comparison video")
    parser.add_argument("--before", required=True, help="Path to the original video")
    parser.add_argument("--after", required=True, help="Path to the stabilised video")
    parser.add_argument("--out", required=True, help="Output path for the comparison MP4")
    parser.add_argument("--label-left", default="Before")
    parser.add_argument("--label-right", default="After")
    args = parser.parse_args()

    before_path = pathlib.Path(args.before)
    after_path = pathlib.Path(args.after)
    out_path = pathlib.Path(args.out)

    fps_before, before_frames = read_video_frames(before_path)
    fps_after, after_frames = read_video_frames(after_path)
    fps = fps_after if fps_after > 0 else fps_before

    frames = side_by_side_frames(before_frames, after_frames, args.label_left, args.label_right)
    if not frames:
        raise RuntimeError("No overlapping frames between the input videos")

    height, width, _ = frames[0].shape
    write_video(out_path, fps, (width, height), frames)

    print(f"Wrote side-by-side comparison to {out_path}")


if __name__ == "__main__":
    main()
