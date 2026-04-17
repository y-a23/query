import os
import json
import pyarrow.parquet as pq
import pandas as pd
from typing import List, Dict, Optional

from QAsummary.qa_test1 import qa_with_vllm_api, is_answer_correct, load_parquet_dataset


def summary_with_vllm_api(
    article: str,
    api_url: str = "http://localhost:9000/v1/chat/completions",
    model_name: str = "Qwen3-4B",
    api_key: str = "token-abc123",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: int = 120,
    enable_think_mode: bool = False,
) -> Optional[str]:
    """两轮方案第一步：根据文章生成 15%-25% 比例的摘要。"""
    if not article or not isinstance(article, str):
        return None

    word_count = len(article.split())
    target_min = max(30, int(word_count * 0.15))
    target_max = max(50, int(word_count * 0.25))

    prompt = (
        "Please summarize the following text.\n"
        f"The summary length should be between 15% and 25% of the original word count, approximately {target_min} - {target_max} words.\n"
        "Only output the summary, without additional explanation.\n\n"
        "Original text:\n" + article.strip()
    )

    try:
        answer = qa_with_vllm_api(
            question=prompt,
            article="",
            api_url=api_url,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            enable_think_mode=enable_think_mode,
        )
        return answer
    except Exception as e:
        print(f"[summary API error] {e}")
        return None


def test_qa_accuracy_two_rounds(
    parquet_path: str,
    max_data: int = 100,
    threshold: float = 0.8,
    api_url: str = "http://localhost:9000/v1/chat/completions",
    model_name: str = "Qwen3-4B",
    api_key: str = "token-abc123",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: int = 120,
    enable_think_mode: bool = True,
) -> Dict:
    dataset = load_parquet_dataset(parquet_path)
    if not dataset:
        return {"error": "数据集读取失败"}

    results = []
    total_questions = 0
    correct_questions = 0

    for idx, entry in enumerate(dataset[:max_data]):
        pmid = entry.get("pmid", f"unknown_{idx}")
        text_path = entry.get("text_path")
        questions = entry.get("questions", [])

        if not text_path or not questions:
            print(f"[warn] PMID {pmid} skip missing text_path/questions")
            continue

        # 读取文章内容
        full_text_file = text_path + ".readable.txt"
        if not os.path.exists(full_text_file):
            print(f"[warn] PMID {pmid} text file not found: {full_text_file}")
            continue

        with open(full_text_file, "r", encoding="utf-8") as f:
            article = f.read()

        if not article:
            print(f"[warn] PMID {pmid} empty article")
            continue

        # 1. 第一轮：生成摘要
        summary = summary_with_vllm_api(
            article=article,
            api_url=api_url,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            enable_think_mode=False,
        )

        if not summary:
            print(f"[warn] PMID {pmid} summary failed")
            continue

        # 2. 第二轮：基于摘要和问题做问答
        for q_idx, q in enumerate(questions):
            if not isinstance(q, dict):
                print(f"[warn] PMID {pmid} Q{q_idx+1} is not dict")
                continue

            question_text = q.get("question")
            golden_answers = q.get("golden_answers", [])
            expected_answer = q.get("answer")

            if not question_text:
                continue

            # 兼容 golden 答案为单字符串
            if isinstance(golden_answers, str):
                golden_list = [golden_answers]
            elif isinstance(golden_answers, (list, tuple)):
                golden_list = list(golden_answers)
            else:
                golden_list = [str(golden_answers)]

            qa_prompt = f"Context summary: {summary}\n\nQuestion: {question_text}"

            model_answer = qa_with_vllm_api(
                question=qa_prompt,
                article="",
                api_url=api_url,
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                enable_think_mode=enable_think_mode,
            )

            total_questions += 1
            is_correct = is_answer_correct(model_answer, golden_list, threshold)
            if is_correct:
                correct_questions += 1

            results.append({
                "pmid": pmid,
                "question_idx": q_idx + 1,
                "question": question_text,
                "model_answer": model_answer,
                "expected_answer": expected_answer,
                "golden_answers": golden_list,
                "is_correct": is_correct,
            })

            print(f"PMID {pmid} Q{q_idx+1}: {'✓' if is_correct else '✗'} ({model_answer})")

    accuracy = correct_questions / total_questions if total_questions else 0.0
    report = {
        "total_items": min(max_data, len(dataset)),
        "total_questions": total_questions,
        "correct_questions": correct_questions,
        "accuracy": round(accuracy, 4),
        "results": results,
    }

    summary_out_path = "qa_test2_results.json"
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved result to {summary_out_path}")

    return report


if __name__ == "__main__":
    PARQUET_PATH = "/nfsdata3/yiao/data/PaperSearchQA_summary/data/merge_data/train_merged_by_pmid.parquet"
    report = test_qa_accuracy_two_rounds(
        PARQUET_PATH,
        max_data=100,
        threshold=0.8,
        api_url="http://localhost:9000/v1/chat/completions",
        model_name="Qwen3-4B",
        api_key="token-abc123",
        temperature=0.1,
        max_tokens=512,
        timeout=120,
        enable_think_mode=True,
    )
    print(report)
