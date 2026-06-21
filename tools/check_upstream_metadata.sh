#!/usr/bin/env bash
# Static upstream metadata watch — Hugging Face API only, no GGUF download.
#
# Examples:
#   ./tools/check_upstream_metadata.sh
#   ./tools/check_upstream_metadata.sh --models qwen2.5-14b-q4_k_m
#   ./tools/check_upstream_metadata.sh --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "${ROOT}/tools/check_upstream_metadata.py" "$@"
