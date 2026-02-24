#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "${ROOT_DIR}"

uv sync --all-packages
uv run --package training python -m training.prepare_data --config configs/data.yaml
uv run --package training python -m training.train --config configs/train.yaml
uv run --package training python -m training.evaluate --config configs/train.yaml
uv run --package training python -m training.merge_and_export --train-config configs/train.yaml --ollama-config configs/ollama.yaml

echo "Training pipeline finished."
echo "Now run: packages/serving/convert_to_gguf.sh"
