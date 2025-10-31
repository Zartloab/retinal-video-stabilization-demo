"""Basic smoke tests ensuring package wiring works."""
from __future__ import annotations

from pathlib import Path

from retinal_stab import __version__
from retinal_stab.utils import load_yaml


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_config_loads() -> None:
    cfg = load_yaml(Path("configs/default.yaml"))
    assert cfg["detector"] in {"ORB", "SIFT"}
    assert cfg["seed"] == 42
