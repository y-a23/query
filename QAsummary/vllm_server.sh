#!/bin/bash
  # --reasoning-parser qwen3 \
# vLLM 服务器启动脚本
export VLLM_LOGGING_LEVEL=DEBUG
CUDA_VISIBLE_DEVICES=0 vllm serve /nfsdata3/yiao/yiao/model/Qwen3-4B \
  --port 9000 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name "Qwen3-4B" \
  --gpu-memory-utilization 0.5 \
  --dtype auto \
  --api-key token-abc123 \
  --enable-log-requests