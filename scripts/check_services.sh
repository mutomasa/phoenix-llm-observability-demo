#!/usr/bin/env bash
# Phoenix と vLLM の疎通を確認する
set -uo pipefail

PHOENIX_BASE_URL="${PHOENIX_BASE_URL:-http://localhost:6006}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
status=0

check() {
  local name="$1" url="$2"
  if curl -sf -o /dev/null --max-time 5 "$url"; then
    echo "[OK]   ${name}: ${url}"
  else
    echo "[NG]   ${name}: ${url}"
    status=1
  fi
}

check "Phoenix" "${PHOENIX_BASE_URL}/healthz"
check "vLLM" "${VLLM_BASE_URL}/models"
exit "${status}"
