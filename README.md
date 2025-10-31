# Retinal Video Stabilization Demo

[![CI - Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#) [![Lint](https://img.shields.io/badge/lint-black%20%2B%20ruff-blue)](#) [![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

A polished, reproducible pipeline that demonstrates feature-based stabilization of handheld retinal videos. The project highlights the value of high-quality preprocessing, robust feature matching, and temporal smoothing when medical hardware introduces jitter.

## Pipeline at a Glance
```
┌──────────┐   ┌────────────┐   ┌─────────────┐   ┌───────────┐   ┌──────────────┐
│ Ingest   │─▶│ Preprocess  │─▶│ Feature/     │─▶│ Motion     │─▶│ Warp &        │
│ frames   │  │ (gray +     │  │ match +      │  │ smoothing  │  │ crop output   │
│ (BGR)    │  │ CLAHE + mask│  │ RANSAC affine│  │ (moving avg│  │ (+ metrics)   │
└──────────┘   └────────────┘   └─────────────┘   └───────────┘   └──────────────┘
```

## Quickstart

If you are new to Python projects, follow these steps **in order**. Every command is run from the root of the cloned repository (the folder that contains this README).

1. **Open a terminal.**
   - macOS / Linux: Launch the *Terminal* app.
   - Windows: Use *Command Prompt* or *PowerShell*. If you have [Windows Subsystem for Linux](https://learn.microsoft.com/windows/wsl/install), that works great too.
2. **Change into the project directory.** The prompt should look similar to `.../retinal-video-stabilization-demo$` after running:
   ```bash
   cd path/to/retinal-video-stabilization-demo
   ```
3. **(Optional but recommended) Create a virtual environment** so the dependencies stay isolated from other projects:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
4. **Install the Python packages** listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
5. **Explore the command-line interface** and check that it loads:
   ```bash
   python -m retinal_stab.cli --help
   ```
6. **Generate practice videos** (synthetic, retina-like footage):
   ```bash
   python -m retinal_stab.cli synth --count 2
   ```
   You will see new `.mp4` files appear under `data/synthetic/`.
7. **Run the stabilizer** on that folder and write results to `data/output/`:
   ```bash
   python -m retinal_stab.cli stabilise \
     --in data/synthetic \
     --out data/output \
     --method affine \
     --smooth_win 45 \
     --crop 0.06
   ```
   The program prints a mini report (frame counts, inliers, stability index) so you can confirm it worked.
8. **Create a side-by-side comparison video** (optional but great for presentations):
   ```bash
   python scripts/side_by_side.py \
     --before data/synthetic/demo_000.mp4 \
     --after data/output/demo_000_stab.mp4 \
     --out data/output/demo_000_compare.mp4
   ```
   Play the output file to visually compare the stabilization.

Each of the commands above can be re-run safely. If you ever forget where to execute them, remember: **always run them from the project root** so the relative paths (like `data/synthetic`) resolve correctly.

## Synthetic-to-Stable Demo

1. `python -m retinal_stab.cli synth --count 2` creates jittery synthetic fundus videos in `data/synthetic/` with baked-in blinks and low-light noise.
2. `python -m retinal_stab.cli stabilise ...` runs the affine pipeline, reports inliers, and writes stabilised videos to `data/output/`.
3. `python scripts/side_by_side.py ...` stacks the before/after results for qualitative review.
4. Run `python -m retinal_stab.cli metrics --before ... --after ...` to quantify the stability index.

## Metrics

- **Stability index** – average dense optical-flow magnitude between consecutive frames (lower is better).
- **Feature drift** – variance of transform deltas before and after smoothing (exposed via the returned metrics dictionary and notebook).

The accompanying notebook `notebooks/01_demo_metrics.ipynb` loads synthetic data, computes these metrics, and plots distributions to illustrate the improvement.

## Reproducibility & Configuration

- Deterministic seeds are enforced through `configs/default.yaml` and `retinal_stab.utils.set_seed`.
- Tuning knobs (detector, feature count, smoothing window, crop percentage, mask radius, denoising strategy) live in YAML for consistent experimentation.
- All dependencies are pinned in `requirements.txt` with mirrors in `pyproject.toml` and `environment.yml` for Conda users.
- A `Dockerfile` and `Makefile` (`make install`, `make lint`, `make test`, `make demo`) keep the environment turnkey.

## Ethics & Privacy

This repository contains **no patient data**. Synthetic retina-like footage is generated on the fly, ensuring privacy compliance while offering realistic motion challenges.

## Continuous Integration

GitHub Actions (see `.github/workflows/ci.yml`) runs lint (`ruff`, `black --check`) and tests (`pytest`) on Python 3.10.

## Developer Workflow

1. Clone the repo and install dependencies: `make install`.
2. Enable consistent formatting: `pre-commit install`.
3. Run the demo end-to-end: `make demo`.
4. Explore metrics and plots via `notebooks/01_demo_metrics.ipynb`.

## Future Work

- Optic disc–anchored micro-stabilisation to refine residual motion.
- Optical flow fallback when features are sparse.
- Learning-based extensions to benchmark against classical methods.

## How I’d explain this in an interview

- Approach: Feature-based global stabilisation (ORB/SIFT → matches → RANSAC affine), temporal smoothing (moving average), warp with crop to avoid borders.
- Why affine (not homography): Avoids overfitting/perspective distortions; retina approximates a plane within FOV.
- Tuning knobs: n_features, ratio_test, ransac_reproj_thresh, smooth_win, crop_pct, mask radius.
- Failure modes & mitigations: blinks/occlusions (inlier gating + hold last), low texture (fallback to SIFT or dense flow extension), drift (keyframe re-anchors).
- Evaluation: stability index (optical flow magnitude) and feature drift; qualitative side-by-side.
- Ethics: no patient data; synthetic demo included.
- Future work: optic-disc anchored micro-stabilisation; optical flow fusion; learning-based benchmark.

## Citation

If this demo informs your research or interviews, please cite using the provided `CITATION.cff` metadata.

## License

Released under the [MIT License](LICENSE).
