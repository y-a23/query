import logging
import uuid
import time
from contextvars import ContextVar

# 1. 定义全局变量，用于跨函数、跨文件传递 Request ID
request_id_var = ContextVar("request_id", default="system")

# 2. 定义 Logger 名称
LOGGER_NAME = "RAG-System"

class RequestIDFilter(logging.Filter):
    """将 ContextVar 中的 request_id 注入到每一条日志记录中"""
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

def setup_logging():
    """在主程序启动时调用一次，初始化日志格式"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        
        # 格式中包含 [%(request_id)s]，这是自动区分并发请求的关键
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | [%(request_id)s] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addFilter(RequestIDFilter())
    return logger

# 方便其他文件直接 import 使用
logger = logging.getLogger(LOGGER_NAME)