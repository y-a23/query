CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /nfsdata/yiao/model/Qwen3-Embedding-0.6B \
  --port 8081 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.5
  --tensor-parallel-size 1