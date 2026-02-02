import requests
import json


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
    """
    格式化打印检索结果
    
    Args:
        result (dict): 服务器返回的检索结果
    """
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
    
    # 遍历每个查询的结果（在这个例子中我们只查询了一个）
    for query_idx, query_results in enumerate(results):
        print(f"查询 {query_idx + 1} 的结果:")
        print("-" * 40)
        
        for idx, item in enumerate(query_results):
            if isinstance(item, dict) and "document" in item and "score" in item:
                # 包含分数的结果
                document = item["document"]
                score = item["score"]
                
                print(f"结果 {idx + 1} (相似度: {score:.4f}):")
                print(f"  内容: {document.get('contents', 'N/A')[:200]}...")
                
                # 如果有其他字段也打印出来
                for key, value in document.items():
                    if key != 'contents':
                        print(f"  {key}: {value}")
                print()
            else:
                # 不包含分数的结果
                print(f"结果 {idx + 1}:")
                print(f"  内容: {item.get('contents', 'N/A')[:200]}...")
                
                # 如果有其他字段也打印出来
                for key, value in item.items():
                    if key != 'contents':
                        print(f"  {key}: {value}")
                print()


def main():
    # 设置服务器URL（根据实际情况修改）
    server_url = "http://localhost:8000"
    
    query = "What is the mechanism of action of bevacizumab in treating colorectal cancer?"
    
    print(f"正在向服务器 {server_url} 发送查询: '{query}'")
    
    # 发送查询并获取结果
    result = test_retrieval_server(query, server_url, topk=3, return_scores=True)
    
    if result:
        print_retrieval_results(result)
    else:
        print("未能获取检索结果")


if __name__ == "__main__":
    main()