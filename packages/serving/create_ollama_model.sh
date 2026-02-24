#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_NAME="${1:-resumo-noticias-pt}"
MODELFILE="${PROJECT_ROOT}/packages/serving/Modelfile"

if [[ ! -f "${MODELFILE}" ]]; then
  echo "Modelfile not found: ${MODELFILE}"
  exit 1
fi

ollama create "${MODEL_NAME}" -f "${MODELFILE}"
ollama run "${MODEL_NAME}" "Resuma em 2 frases: O mercado de tecnologia abriu em alta hoje com expectativas de cortes de juros."
