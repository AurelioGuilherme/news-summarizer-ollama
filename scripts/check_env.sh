#!/usr/bin/env bash
set -euo pipefail

echo "== Tooling =="
uv --version || true
nvidia-smi || true

echo "== Python via uv =="
uv run python -c "import sys; print(sys.version)"

echo "== Torch/CUDA =="
uv run --package training python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY

echo "== Ollama =="
if command -v ollama >/dev/null 2>&1; then
  ollama --version
else
  echo "ollama not found"
fi
