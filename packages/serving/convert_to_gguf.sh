#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MERGED_DIR="${PROJECT_ROOT}/artifacts/merged/qwen2.5-0.5b-news-pt-merged"
GGUF_DIR="${PROJECT_ROOT}/artifacts/gguf"
F16_GGUF="${GGUF_DIR}/resumo-noticias-pt-f16.gguf"
Q4_GGUF="${GGUF_DIR}/resumo-noticias-pt-q4_k_m.gguf"

if [[ -z "${LLAMA_CPP_DIR:-}" ]]; then
  echo "Set LLAMA_CPP_DIR pointing to your llama.cpp folder."
  echo "Example: export LLAMA_CPP_DIR=/path/to/llama.cpp"
  exit 1
fi

if [[ ! -d "${MERGED_DIR}" ]]; then
  echo "Merged model dir not found: ${MERGED_DIR}"
  echo "Run merge step first."
  exit 1
fi

mkdir -p "${GGUF_DIR}"

uv run --package training python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "${MERGED_DIR}" --outfile "${F16_GGUF}" --outtype f16
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" "${F16_GGUF}" "${Q4_GGUF}" Q4_K_M

echo "GGUF exported to: ${Q4_GGUF}"
