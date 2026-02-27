import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import statistics


def test_single_request(query: str, server_url: str = "http://localhost:8000", 
                       topk: int = 3, return_scores: bool = True) -> Dict[str, Any]:
    """
    发送单个检索请求并返回结果和延迟信息
    
    Args:
        query (str): 查询字符串
        server_url (str): 检索服务器的URL
        topk (int): 返回的文档数量
        return_scores (bool): 是否返回相似度分数
    
    Returns:
        dict: 包含结果和延迟信息的字典
    """
    url = f"{server_url}/retrieve"
    
    # 构造请求数据
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": return_scores
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
        response.raise_for_status()
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # 转换为毫秒
        
        result = response.json()
        return {
            "success": True,
            "latency_ms": latency,
            "result": result,
            "status_code": response.status_code
        }
    
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        return {
            "success": False,
            "latency_ms": latency,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }
    except json.JSONDecodeError as e:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        return {
            "success": False,
            "latency_ms": latency,
            "error": f"JSON解析失败: {e}"
        }


def load_test_retrieval_server(queries: List[str], 
                              server_url: str = "http://localhost:8000",
                              topk: int = 3, 
                              return_scores: bool = True,
                              num_threads: int = 10,
                              requests_per_thread: int = 10) -> Dict[str, Any]:
    """
    对检索服务器进行负载测试
    
    Args:
        queries (List[str]): 查询字符串列表
        server_url (str): 服务器URL
        topk (int): 返回文档数量
        return_scores (bool): 是否返回分数
        num_threads (int): 并发线程数
        requests_per_thread (int): 每个线程发送的请求数
    
    Returns:
        dict: 包含统计信息的结果
    """
    total_requests = num_threads * requests_per_thread
    print(f"开始负载测试:")
    print(f"  - 总请求数: {total_requests}")
    print(f"  - 并发线程数: {num_threads}")
    print(f"  - 每线程请求数: {requests_per_thread}")
    print(f"  - 查询样本数: {len(queries)}")
    print("-" * 50)
    
    results = []
    latencies = []
    success_count = 0
    error_count = 0
    
    def worker_thread(thread_id: int):
        thread_results = []
        query_index = 0
        
        for i in range(requests_per_thread):
            # 循环使用查询列表
            query = queries[query_index % len(queries)]
            query_index += 1
            
            result = test_single_request(query, server_url, topk, return_scores)
            thread_results.append(result)
            
            if result["success"]:
                latencies.append(result["latency_ms"])
            
        return thread_results
    
    start_time = time.time()
    
    # 使用线程池执行并发请求
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
        
        # 收集结果
        for future in as_completed(futures):
            thread_results = future.result()
            results.extend(thread_results)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 统计成功和失败的请求数
    for result in results:
        if result["success"]:
            success_count += 1
        else:
            error_count += 1
    
    # 计算统计信息
    stats = {
        "total_requests": total_requests,
        "successful_requests": success_count,
        "failed_requests": error_count,
        "success_rate": (success_count / total_requests) * 100,
        "total_duration_seconds": total_duration,
        "requests_per_second": total_requests / total_duration if total_duration > 0 else 0,
        "latencies": latencies
    }
    
    # 计算延迟统计（只有成功的请求）
    if latencies:
        stats.update({
            "latency_stats": {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "p95_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                "p99_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
            }
        })
    else:
        stats["latency_stats"] = {}
    
    return stats


def print_load_test_results(stats: Dict[str, Any]):
    """
    格式化打印负载测试结果
    
    Args:
        stats (dict): 负载测试统计信息
    """
    print("\n" + "=" * 60)
    print("负载测试结果统计")
    print("=" * 60)
    
    print(f"总请求数: {stats['total_requests']}")
    print(f"成功请求数: {stats['successful_requests']}")
    print(f"失败请求数: {stats['failed_requests']}")
    print(f"成功率: {stats['success_rate']:.2f}%")
    print(f"总耗时: {stats['total_duration_seconds']:.2f} 秒")
    print(f"QPS (每秒请求数): {stats['requests_per_second']:.2f}")
    
    if stats['latency_stats']:
        print("\n延迟统计 (毫秒):")
        print("-" * 30)
        latency_stats = stats['latency_stats']
        print(f"最小延迟: {latency_stats['min_ms']:.2f} ms")
        print(f"最大延迟: {latency_stats['max_ms']:.2f} ms")
        print(f"平均延迟: {latency_stats['avg_ms']:.2f} ms")
        print(f"中位数延迟: {latency_stats['median_ms']:.2f} ms")
        print(f"95% 分位数: {latency_stats['p95_ms']:.2f} ms")
        print(f"99% 分位数: {latency_stats['p99_ms']:.2f} ms")
    
    print("=" * 60)


def test_retrieval_server(query, server_url="http://localhost:8000", topk=3, return_scores=True):
    """
    向检索服务器发送查询并返回结果
    
    Args:
        query (str): 查询字符串
        server_url (str): 检索服务器的URL
        topk (int): 返回的文档数量
        return_scores (bool): 是否返回相似度分数
    
    Returns:
        dict: 服务器返回的检索结果
    """
    url = f"{server_url}/retrieve"
    
    # 构造请求数据
    payload = {
        "queries": [query],  # API期望接收查询列表
        "topk": topk,
        "return_scores": return_scores
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # 检查HTTP错误
        
        result = response.json()
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return None


def print_retrieval_results(result):
    """原始的结果打印函数，保持向后兼容"""
    if not result or "result" not in result:
        print("没有返回有效结果")
        return
    
    results = result["result"]
    if not results or len(results) == 0:
        print("没有检索到任何结果")
        return
    
    print("=" * 60)
    print("检索结果:")
    print("=" * 60)
    
    for query_idx, query_results in enumerate(results):
        print(f"查询 {query_idx + 1} 的结果:")
        print("-" * 40)
        
        for idx, item in enumerate(query_results):
            if isinstance(item, dict) and "document" in item and "score" in item:
                document = item["document"]
                score = item["score"]
                print(f"结果 {idx + 1} (相似度: {score:.4f}):")
                print(f"  内容: {document.get('contents', 'N/A')[:200]}...")
                for key, value in document.items():
                    if key != 'contents':
                        print(f"  {key}: {value}")
                print()
            else:
                print(f"结果 {idx + 1}:")
                print(f"  内容: {item.get('contents', 'N/A')[:200]}...")
                for key, value in item.items():
                    if key != 'contents':
                        print(f"  {key}: {value}")
                print()


def test1():
    # 测试配置
    server_url = "http://localhost:8000"
    
    # 测试查询列表
    test_queries = [
        "Perinatal unilateral hydrocephalus. Atresia of the foramen of Monro.",
        "Treatment options for pediatric brain tumors",
        "Neurological complications in premature infants",
        "Cerebral palsy diagnosis and management",
        "Spinal cord injury rehabilitation protocols"
    ]

    # 压测参数
    num_threads = 20          # 并发线程数
    requests_per_thread = 25  # 每个线程的请求数
    
    print(f"正在对服务器 {server_url} 进行负载测试...")
    print(f"使用 {len(test_queries)} 个不同的查询进行测试")
    
    # 执行负载测试
    stats = load_test_retrieval_server(
        queries=test_queries,
        server_url=server_url,
        topk=3,
        return_scores=True,
        num_threads=num_threads,
        requests_per_thread=requests_per_thread
    )
    
    # 打印结果
    print_load_test_results(stats)

def test2():
    query = "Expression of the rice yellow mottle virus P1 protein in vitro and in vivo and its involvement in virus spread"
    ret = test_single_request(query, server_url="http://localhost:8000", topk=3, return_scores=True)
    print(ret)

def test3():
    import datasets

    # 读取单个 parquet 文件
    dataset = datasets.load_dataset('parquet', data_files="/nfsdata3/yiao/data/PaperSearchQA/data/test-00000-of-00001.parquet")['train']

    same_cnt = 0
    start_time = time.time()
    for batch in dataset.iter(batch_size=1):

        query = batch['paper_title'][0]
        ret = test_single_request(query, server_url="http://localhost:8000", topk=3, return_scores=True)
        if ret['result']['result'][0][0]['document']['title'] == query:
            same_cnt += 1
        #     import pdb; pdb.set_trace()
        # else:
        #     print(ret)
        #     print(query)
        #     import pdb; pdb.set_trace()

    print(f"相同标题数量: {same_cnt}")
    print(f"数据集大小: {len(dataset)}")
    print(f"耗时: {time.time() - start_time:.4f} 秒")

if __name__ == "__main__":
    test1()