import requests
import json
from typing import Optional

def qa_with_vllm_api(
    question: str,
    article: str,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    model_name: str = "Qwen3-4B",
    api_key: str = "token-abc123",
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: int = 120
) -> Optional[str]:
    """
    使用本地 vLLM API 进行问答：输入 question + article，返回 answer
    
    Args:
        question: 用户问题
        article: 参考文章/上下文
        api_url: vLLM API 地址
        model_name: 模型名称（需与启动时一致）
        temperature: 生成温度（建议 0~0.3 用于提取型任务）
        max_tokens: 最大生成长度
        timeout: 请求超时时间（秒）
    
    Returns:
        str: 模型生成的答案，失败时返回 None
    """
    # 构造 Qwen 格式的 prompt（使用 ChatML）
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant that answers questions based on the provided context.<|im_end|>\n"
        f"<|im_start|>user\nContext: {article.strip()}\n\nQuestion: {question.strip()}\nPlease answer the question concisely based on the context.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    payload = {
        "model": model_name,
        "api_key": api_key,
        "messages": [{"role": "user", "content": prompt}],  # vLLM 会识别 ChatML
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>", "</s>"],  # 防止生成多余内容
        "stream": False
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        print(f"[vLLM API Error] {e}")
        return None