#!/usr/bin/env bash
# vLLM + Qwen3 をホストOS上で起動する（RTX A1000 8GB向け設定）
# vLLMはサイズが大きくCUDA依存のため、本プロジェクトの依存には含めず uvx で実行する。
set -euo pipefail

MODEL="${LLM_MODEL:-Qwen/Qwen3-1.7B}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "vLLM is already running on port ${PORT}:"
  curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys;[print(" -",m["id"]) for m in json.load(sys.stdin)["data"]]'
  exit 0
fi

# メモリ不足時: VLLM_MAX_MODEL_LEN=2048 または LLM_MODEL=Qwen/Qwen3-0.6B を指定
exec uvx --from vllm vllm serve "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_LEN}"
