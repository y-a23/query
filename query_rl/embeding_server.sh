CUDA_VISIBLE_DEVICES=1 \
LOG_LEVEL=DEBUG \
python -m vllm.entrypoints.openai.api_server \
  --model /nfsdata3/yiao/yiao/model/Qwen3-Embedding-0.6B \
  --port 8081 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1