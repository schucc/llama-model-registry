#!/usr/bin/env bash
# Validate manifest models: per-source download → llama-server chat → remove GGUF.
#
# Prerequisites:
#   - macOS Apple Silicon
#   - python3 (stdlib only)
#   - ../diarySwift repo (for the pinned llama-server binary)
#
# The Aeris app does NOT need to be running — this script spawns llama-server directly.
#
# Examples:
#   ./tools/validate_models.sh --models qwen2.5-7b-q4_k_m
#   ./tools/validate_models.sh --report-dir reports/models

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIARY_SWIFT="${DIARY_SWIFT:-$(cd "${ROOT}/../diarySwift" 2>/dev/null && pwd || true)}"
LLAMA_SERVER="${LLAMA_SERVER_BIN:-${DIARY_SWIFT}/Helpers/llama-server}"
FETCH_SCRIPT="${DIARY_SWIFT}/tools/fetch_llama_server.sh"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  if [[ -f "${FETCH_SCRIPT}" ]]; then
    echo "Fetching llama-server via ${FETCH_SCRIPT} …"
    bash "${FETCH_SCRIPT}"
  else
    echo "ERROR: llama-server not found at ${LLAMA_SERVER}" >&2
    echo "Set DIARY_SWIFT or LLAMA_SERVER_BIN, or clone diarySwift next to this repo." >&2
    exit 1
  fi
fi

exec python3 "${ROOT}/tools/validate_models.py" \
  --llama-server-bin "${LLAMA_SERVER}" \
  "$@"
