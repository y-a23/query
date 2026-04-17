import json
import os
import requests
import pyarrow.parquet as pq
import pandas as pd
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher

# 复用QA函数（新增Think模式）
def qa_with_vllm_api(
    question: str,
    article: str,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    model_name: str = "Qwen3-4B",
    api_key: str = "token-abc123",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 120,
    enable_think_mode: bool = True
) -> Optional[str]:
    """使用本地 vLLM API 进行问答（支持Think模式）"""
    think_prompt = """
    Please follow these steps to answer the question:
    1. First, read and understand the entire context carefully.
    2. Identify the EXACT key phrase/word in the context that directly answers the question.
    3. Verify that the key phrase/word is the only answer needed (no extra explanation).
    4. Output ONLY the key phrase/word as the answer (no sentences, no extra words, no parentheses, no abbreviations unless required).
    5. DO NOT add any explanations, descriptions, or additional information of any kind.
    6. Answer ONLY based on the given context, do not make assumptions.
    """ if enable_think_mode else ""

    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant that answers questions based on the provided context.{think_prompt}<|im_end|>\n"
        f"<|im_start|>user\nContext: {article.strip()}\n\nQuestion: {question.strip()}\nPlease answer the question concisely based on the context.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    payload = {
        "model": model_name,
        "api_key": api_key,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>", "</s>"],
        "stream": False
    }
    
    try:
        # 构造 headers
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        # 发送请求时带上
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        # response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        print(f"[vLLM API Error] Question: {question[:50]}... | Error: {e}")
        return None
import re
from difflib import SequenceMatcher
from typing import List

def strip_think_answer_blocks(text: str) -> str:
    """剔除`<think>...</think>`区域并提取 answer 内容（如有）。"""
    if not text or not isinstance(text, str):
        return ""
    txt = text.strip()
    lower = txt.lower()

    # 0. 删除 think 内容，保留后续文本
    if '<think>' in lower:
        start_idx = lower.index('<think>')
        end_tag = '</think>'
        if end_tag in lower[start_idx:]:
            end_idx = lower.index(end_tag, start_idx) + len(end_tag)
            txt = txt[end_idx:].strip()
        else:
            txt = txt[:start_idx].strip()

    # 1. 提取 answer 标签或 answer: / final answer:
    lower = txt.lower()
    if '<answer>' in lower and '</answer>' in lower:
        start = lower.index('<answer>') + len('<answer>')
        end = lower.index('</answer>')
        txt = txt[start:end].strip()
    elif 'final answer:' in lower:
        idx = lower.index('final answer:') + len('final answer:')
        txt = txt[idx:].strip()
    elif 'answer:' in lower:
        idx = lower.index('answer:') + len('answer:')
        txt = txt[idx:].strip()

    return txt.strip()


def tokenize_answer(text: str) -> List[str]:
    """
    将答案拆分为关键词token（清洗+分词）
    步骤：小写 → 处理think/answer标记 → 去符号 → 拆词 → 去停用词
    """
    txt = strip_think_answer_blocks(text)

    # 1. 统一小写
    text = txt.lower()
    # 2. 去掉括号及内容、特殊符号，只保留字母/数字/空格
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # 3. 拆分为单词（按空格），去空，去停用词
    stop_words = {'the', 'a', 'an', 'is', 'are', 'and', 'or', 'of', 'in', 'to', 'for'}
    tokens = [token for token in text.split() if token and token not in stop_words]
    return tokens

def f1_score(model_tokens: List[str], golden_tokens: List[str]) -> float:
    """
    计算Token级F1 Score
    F1 = 2 * (P * R) / (P + R)
    P = 正确关键词数 / 模型答案关键词数
    R = 正确关键词数 / 标准答案关键词数
    """
    if not model_tokens and not golden_tokens:
        return 1.0
    if not model_tokens or not golden_tokens:
        return 0.0
    
    # 计算交集（正确匹配的关键词）
    intersection = len(set(model_tokens) & set(golden_tokens))
    # 精确率
    precision = intersection / len(model_tokens) if len(model_tokens) > 0 else 0.0
    # 召回率
    recall = intersection / len(golden_tokens) if len(golden_tokens) > 0 else 0.0
    # F1
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
def is_answer_correct(
    model_answer: str, 
    golden_answers: List[str], 
    f1_threshold: float = 0.5,
    sim_threshold: float = 0.6
) -> bool:
    """
    混合匹配：包含匹配 → 相似度匹配 → F1 Score匹配
    只要任一条件满足 → 判定正确
    """
    if not model_answer:
        return False

    # 清洗函数（基础版）
    def clean_basic(s: str) -> str:
        s = s.lower()
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'[^a-z0-9]', '', s)
        return s

    model_answer = strip_think_answer_blocks(model_answer)
    model_clean = clean_basic(model_answer)

    for golden in golden_answers:
        golden_norm = strip_think_answer_blocks(golden)
        g_clean = clean_basic(golden_norm)
        
        # 1. 包含匹配（最宽松）
        if model_clean in g_clean or g_clean in model_clean:
            return True
        
        # 2. 字符串相似度匹配
        sim = SequenceMatcher(None, model_clean, g_clean).ratio()
        if sim >= sim_threshold:
            return True
        
        # 3. F1 Score匹配（兜底）
        model_tokens = tokenize_answer(model_answer)
        g_tokens = tokenize_answer(golden_norm)
        f1 = f1_score(model_tokens, g_tokens)
        if f1 >= f1_threshold:
            return True

    return False

# 读取Parquet数据集
def load_parquet_dataset(parquet_path: str) -> List[Dict]:
    """
    读取Parquet格式的数据集，转为Python字典列表
    
    Args:
        parquet_path: Parquet文件路径
    
    Returns:
        List[Dict]: 每条数据为一个字典，和JSON格式一致
    """
    try:
        # 读取Parquet文件
        df = pq.read_table(parquet_path).to_pandas()
        # 将DataFrame转为字典列表（处理嵌套结构）
        dataset = []
        for idx, row in df.iterrows():
            # 确保所有字段转为Python原生类型（处理numpy类型）
            data = row.to_dict()
            for key, value in data.items():
                # 处理嵌套列表/字典（如questions字段）
                if isinstance(value, (pd.Series, pd.DataFrame)):
                    data[key] = value.to_dict() if isinstance(value, pd.DataFrame) else value.tolist()
                # 转换numpy类型为Python类型
                elif "numpy" in str(type(value)):
                    data[key] = value.item() if value.size == 1 else value.tolist()
            dataset.append(data)
        print(f"✅ 成功读取Parquet文件，共 {len(dataset)} 条数据")
        return dataset
    except Exception as e:
        print(f"❌ 读取Parquet文件失败: {e}")
        return []

# 核心测试函数（适配Parquet）
def test_qa_accuracy(
    parquet_path: str,
    output_filtered_path: str = "/nfsdata3/yiao/data/PaperSearchQA_summary/data/merge_data/filtered_correct_data.json",
    threshold: float = 0.8,
    **qa_kwargs
) -> Dict:
    """
    测试QA准确率（支持Parquet输入），筛选所有问题回答正确的数据集
    """
    # 1. 读取Parquet数据集
    dataset = load_parquet_dataset(parquet_path)
    if not dataset:
        return {"error": "数据集读取失败"}
    
    total_questions = 0
    correct_questions = 0
    filtered_data = []  # 存储所有问题都回答正确的数据集
    failed_data = []   # 存储筛选失败的条目

    # 2. 遍历每条数据
    for idx, data in enumerate(dataset):
        pmid = data.get("pmid", f"unknown_{idx}")
        text_path = data.get("text_path")
        questions = data.get("questions", [])

        # 跳过无全文路径或无问题的数据
        if not text_path or not questions:
            print(f"[Warning] PMID {pmid}: 无全文路径或问题列表，跳过")
            continue
        
        # 检查全文文件是否存在
        text_path = text_path + ".readable.txt"
        if not os.path.exists(text_path):
            print(f"[Warning] PMID {pmid}: 全文文件不存在 - {text_path}")
            continue
        
        # 读取文章全文
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                article = f.read()
            # 处理超大文本（可选：截断，避免超出模型上下文）
            if len(article) > 80000:  # 根据模型上下文调整
                article = article[:80000]
                print(f"[Info] PMID {pmid}: 全文过长，截断至80000字符")
        except Exception as e:
            print(f"[Error] PMID {pmid}: 读取全文失败 - {e}")
            continue
        
        print(f"\n=== Testing PMID: {pmid} (Question count: {len(questions)}) ===")
        
        # 3. 测试当前数据的所有问题
        current_data_correct = True
        failed_reasons = []
        for q_idx, q in enumerate(questions):
            # 兼容非 dict 的异常数据
            if not isinstance(q, dict):
                print(f"[Error] PMID {pmid} Q{q_idx+1}: question条目不是dict (type={type(q).__name__})，跳过")
                failed_reasons.append(f"Q{q_idx+1}: q不是dict")
                total_questions += 1
                current_data_correct = False
                continue

            question = q.get("question")
            golden_answers = q.get("golden_answers", [])
            expected_answer = q.get("answer")

            has_golden_answers = False
            if isinstance(golden_answers, (list, tuple)):
                has_golden_answers = len(golden_answers) > 0
            elif isinstance(golden_answers, (str, bytes)):
                has_golden_answers = len(golden_answers.strip()) > 0
            elif hasattr(golden_answers, "size"):
                has_golden_answers = golden_answers.size > 0

            if not question or not has_golden_answers:
                print(f"[Warning] PMID {pmid} Q{q_idx+1}: 问题或标准答案为空，跳过")
                failed_reasons.append(f"Q{q_idx+1}: 问题/标准答案为空")
                total_questions += 1
                current_data_correct = False
                continue

            total_questions += 1
            # 调用QA接口
            model_answer = qa_with_vllm_api(
                question=question,
                article=article,
                **qa_kwargs
            )

            # 判断是否正确
            is_correct = is_answer_correct(model_answer, golden_answers, threshold)
            if is_correct:
                correct_questions += 1
            else:
                current_data_correct = False
                failed_reasons.append(f"Q{q_idx+1}: 答案不正确")

            # 打印详细信息
            print(f"Q{q_idx+1}: {question[:80]}..." if len(question) > 80 else f"Q{q_idx+1}: {question}")
            print(f"Model Answer: {model_answer}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Result: {'✓ Correct' if is_correct else '✗ Incorrect'}\n")

        # 4. 如果当前数据的所有问题都正确，加入筛选列表
        if current_data_correct:
            filtered_data.append(data)
            print(f"✅ PMID {pmid}: All questions answered correctly (added to filtered list)")
        else:
            failed_data.append({
                "pmid": pmid,
                "total_questions": len(questions),
                "failed_reasons": failed_reasons,
            })
            print(f"❌ PMID {pmid}: Some questions answered incorrectly")
    
    # 5. 计算整体准确率
    overall_accuracy = correct_questions / total_questions if total_questions > 0 else 0
    
    # 6. 保存筛选后的数据（JSON格式，便于后续使用）
    try:
        with open(output_filtered_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 筛选后的数据已保存至: {output_filtered_path}")
    except Exception as e:
        print(f"❌ 保存筛选数据失败: {e}")
    
    # 7. 生成测试报告
    report = {
        "total_data_count": len(dataset),
        "total_questions_tested": total_questions,
        "correct_questions": correct_questions,
        "overall_accuracy(%)": round(overall_accuracy * 100, 2),
        "filtered_data_count": len(filtered_data),
        "failed_data_count": len(failed_data),
        "filtered_data_path": output_filtered_path,
        "failed_data": failed_data,
    }
    
    # 打印最终报告
    print("\n" + "="*60)
    print("📊 QA Accuracy Test Report (Parquet Dataset)")
    print("="*60)
    for key, value in report.items():
        if key == "failed_data" and isinstance(value, list) and len(value) > 0:
            print("failed_data:")
            for item in value:
                print(f"  - pmid: {item['pmid']}, total_questions: {item['total_questions']}, reasons: {item['failed_reasons']}")
        elif key != "failed_data":
            print(f"{key}: {value}")
    
    return report

# 主函数
if __name__ == "__main__":
    # 配置参数（适配你的服务器路径）
    PARQUET_PATH = "/nfsdata3/yiao/data/PaperSearchQA_summary/data/merge_data/train_merged_by_pmid.parquet"
    QA_KWARGS = {
        "api_url": "http://localhost:9000/v1/chat/completions",
        "model_name": "Qwen3-4B",  # 替换为你实际使用的模型名
        "api_key": "token-abc123",
        "temperature": 0.1,
        "max_tokens": 512,
        "timeout": 120,
        "enable_think_mode": True  # 启用Think模式
    }
    
    # 执行测试
    test_report = test_qa_accuracy(
        parquet_path=PARQUET_PATH,
        output_filtered_path="/nfsdata3/yiao/data/PaperSearchQA_summary/data/merge_data/filtered_correct_data.json",
        threshold=0.8,
        **QA_KWARGS
    )