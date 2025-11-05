#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "[1/4] Generating synthetic retinal videos..."
python -m retinal_stab.cli synth --count 2 --out data/synthetic

echo "[2/4] Stabilising generated videos..."
python -m retinal_stab.cli stabilise --in data/synthetic --out data/output --method similarity --smooth_win 45 --crop 0.06

FIRST=$(ls data/synthetic/*.mp4 | head -n 1)
BASENAME=$(basename "$FIRST" .mp4)

echo "[3/4] Composing side-by-side comparison for $BASENAME..."
python scripts/side_by_side.py --before "$FIRST" --after "data/output/${BASENAME}_stab.mp4" --out "data/output/${BASENAME}_compare.mp4"

echo "[4/4] Computing stability metrics..."
python -m retinal_stab.cli metrics --before "$FIRST" --after "data/output/${BASENAME}_stab.mp4"

echo "Demo complete. Outputs saved to data/output."
