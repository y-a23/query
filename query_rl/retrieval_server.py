import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from utils import RetrievalSystem
import time

import time
from functools import wraps


# 耗时统计装饰器
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"  [Time Stats] '{func.__name__}' took {end - start:.4f} seconds")
        return result  # 确保返回原函数的结果
    return wrapper

#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

from fastapi import FastAPI
from logger_config import setup_logging, logger, request_id_var
import uuid
import time

app = FastAPI()

# 启动时配置日志
setup_logging()

@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    # 【关键点 1】进入请求立刻生成唯一 ID 并存入 ContextVar
    rid = str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    
    start_time = time.perf_counter()
    logger.info(f">>> Incoming API Request | Queries: {len(request.queries)}")

    resp = []
    for idx, query in enumerate(request.queries):
        q_start = time.perf_counter()
        
        # 在子模块中，你也需要把 print 换成 logger.info
        results, scores = retrieval_system.retrieve(
            question=query,
            k=request.topk,
            rrf_k=request.topk,
        )
        combined = []
        if request.return_scores:
            # If scores are returned, combine them with results
            for i in range(len(results)):
                combined.append({"document": results[i], "score": scores[i]})
            resp.append(combined)
        else:
            resp.append(results)
        
        q_end = time.perf_counter()
        logger.info(f"Query #{idx+1} finished in {q_end - q_start:.4f}s")
        # ... 填充 resp ...

    total_time = time.perf_counter() - start_time
    logger.info(f"<<< Request Processed | Overall Latency: {total_time:.4f}s")
    return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="Contriever", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    parser.add_argument("--corpus_name", type=str, default="PubMed", help="Name of the corpus.")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")

    args = parser.parse_args()
    args.db_dir = "/nfsdata3/yiao/yiao/medRAG"
    args.retriever_name = "BM25"
    # 初始化检索系统
    retrieval_system = RetrievalSystem(args.retriever_name, args.corpus_name, args.db_dir, cache=True, HNSW=True)
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    print("Retrieval system initialized.")
    # 2) Instantiate a global retriever so it is loaded once and reused.
    # retriever = get_retriever(config)
    
    # 3) Launch the server with proper multi-process support
    if False and args.workers > 1:
        # 多进程模式 - 使用import string
        uvicorn.run(
            "retrieval_server:app",  # 注意这里使用模块名:应用名的格式
            host="0.0.0.0", 
            port=8000,
            workers=args.workers,
            reload=False  # 多进程时禁用重载
        )
    else:
        # 单进程模式
        uvicorn.run(app, host="0.0.0.0", port=8000)