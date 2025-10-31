"""Integration smoke test for the stabilisation pipeline."""
from __future__ import annotations

from retinal_stab.ingest import get_video_meta
from retinal_stab.synth import make_synthetic_retina_video
from retinal_stab.utils import load_yaml
from retinal_stab.warp import stabilise_video


def test_pipeline_smoke(tmp_path) -> None:
    src = tmp_path / "demo.mp4"
    make_synthetic_retina_video(str(src), length_sec=1, fps=15, size=(320, 320))
    cfg = load_yaml("configs/default.yaml")
    out = tmp_path / "demo_stab.mp4"
    result = stabilise_video(str(src), str(out), cfg)
    assert out.exists()
    meta_in = get_video_meta(str(src))
    meta_out = get_video_meta(str(out))
    assert abs(meta_in["frame_count"] - meta_out["frame_count"]) <= 1
    assert result["stability_after"] <= result["stability_before"] * 1.2
