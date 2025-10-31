"""Command-line interface for the retinal stabilisation demo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from .metrics import stability_index
from .synth import make_synthetic_retina_video
from .utils import ensure_dir, load_yaml, set_seed
from .warp import stabilise_video


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = load_yaml(path)
    if cfg is None:
        cfg = {}
    return cfg


def _print_summary(results: list[dict]) -> None:
    if not results:
        return
    print("\nSummary")
    print("=" * 60)
    print(f"{'Video':30} {'Frames':>8} {'Inliers':>10} {'StabΔ':>10}")
    for res in results:
        name = Path(res['output_path']).name
        print(
            f"{name:30} {res['frame_count']:>8} {res['avg_inliers']:>10.1f} "
            f"{res['stability_before'] - res['stability_after']:>10.3f}"
        )
    print("=" * 60)
    avg_gain = sum(r['stability_before'] - r['stability_after'] for r in results) / len(results)
    print(f"Average stability gain: {avg_gain:.3f}\n")


def cmd_synth(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out)
    set_seed(42)
    for idx in range(args.count):
        path = out_dir / f"demo_{idx:03d}.mp4"
        print(f"Generating {path} ...")
        make_synthetic_retina_video(str(path), length_sec=3, fps=30)
    print(f"Generated {args.count} synthetic videos in {out_dir}.")


def cmd_stabilise(args: argparse.Namespace) -> None:
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")
    videos = sorted(in_dir.glob("*.mp4"))
    if not videos:
        print("No videos found. Run `python -m retinal_stab.cli synth` first.")
        return

    cfg = _load_config(Path(args.config))
    cfg.update(
        {
            "method": args.method,
            "detector": args.detector,
            "smooth_win": args.smooth_win,
            "crop_pct": args.crop,
        }
    )

    results = []
    out_dir = ensure_dir(args.out)
    for video in videos:
        out_path = out_dir / f"{video.stem}_stab.mp4"
        print(f"Stabilising {video} -> {out_path}")
        res = stabilise_video(str(video), str(out_path), cfg)
        results.append(res)
        print(
            f"  Frames: {res['frame_count']} | Avg inliers: {res['avg_inliers']:.1f} | "
            f"Stability {res['stability_before']:.3f} -> {res['stability_after']:.3f}"
        )
    _print_summary(results)


def cmd_metrics(args: argparse.Namespace) -> None:
    before = Path(args.before)
    after = Path(args.after)
    if not before.exists() or not after.exists():
        raise FileNotFoundError("Both before and after paths must exist")

    import cv2

    def _load(path: Path) -> list:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    before_frames = _load(before)
    after_frames = _load(after)
    print(f"Stability index (before): {stability_index(before_frames):.4f}")
    print(f"Stability index (after):  {stability_index(after_frames):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retinal video stabilisation demo")
    sub = parser.add_subparsers(dest="command")

    synth = sub.add_parser("synth", help="Generate synthetic retinal videos")
    synth.add_argument("--count", type=int, default=3)
    synth.add_argument("--out", dest="out", default="data/synthetic")
    synth.set_defaults(func=cmd_synth)

    stab = sub.add_parser("stabilise", help="Stabilise videos in a folder")
    stab.add_argument("--in", dest="in_dir", default="data/input")
    stab.add_argument("--out", dest="out", default="data/output")
    stab.add_argument("--config", default="configs/default.yaml")
    stab.add_argument("--method", choices=["affine", "homography"], default="affine")
    stab.add_argument("--detector", choices=["ORB", "SIFT"], default="ORB")
    stab.add_argument("--smooth_win", type=int, default=45)
    stab.add_argument("--crop", type=float, default=0.06)
    stab.set_defaults(func=cmd_stabilise)

    metrics = sub.add_parser("metrics", help="Evaluate stability index for a video pair")
    metrics.add_argument("--before", required=True)
    metrics.add_argument("--after", required=True)
    metrics.set_defaults(func=cmd_metrics)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
