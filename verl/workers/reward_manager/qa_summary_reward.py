# -*- coding: utf-8 -*-
"""Reward utilities for QAsummary.

本文件包含多个摘要质量 reward 计算函数，覆盖以下维度：

1. entity_coverage_reward
   - 领域实体一致性：要求 summary 的实体被 ground_truth 覆盖，支持 strict/soft 模式。
2. em_ner_evidence_reward
   - EM+NER证据：要求 summary 至少包含 answer 的 exact match，同时避免仅搬运答案，使用文章实体作为补充证据。
3. summary_length_reward
   - 长度控制：建议 summary 长度保留在 15%-25% 范围内。
4. qa_accuracy_reward
   - QA准确度：基于 question/ground_truth/predictions（或调用 vLLM API）计算答案正确率，作为 reward。

"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Union

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore


def _normalize_entity(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _load_model(model_name: str = "en_core_sci_sm"):
    if spacy is None:
        raise ImportError("spaCy is required for entity extraction, please install with `pip install spacy`.")

    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        # fallback to core web model when scispacy not installed
        if model_name != "en_core_web_sm":
            nlp = spacy.load("en_core_web_sm")
        else:
            raise
    return nlp


def extract_entities(
    text: str,
    model_name: str = "en_core_sci_sm",
    fallback_to_word_token: bool = False,
) -> List[str]:
    """从文本抽取命名实体，若模型失败则用简单分词返回空实体。"""
    text = (text or "").strip()
    if not text:
        return []

    try:
        nlp = _load_model(model_name)
        doc = nlp(text)
        ents = [_normalize_entity(ent.text) for ent in doc.ents if ent.text.strip()]
        if ents:
            return sorted(set(ents))
    except Exception:  # pragma: no cover
        if not fallback_to_word_token:
            raise

    # fallback：使用名词短语提取（如果没实体），或者简单用字符分词
    if fallback_to_word_token and spacy is not None:
        nlp = _load_model("en_core_web_sm")
        doc = nlp(text)
        ents = [_normalize_entity(chunk.text) for chunk in doc.noun_chunks if chunk.text.strip()]
        return sorted(set(ents))

    return []


def entity_coverage_reward(
    ground_truth: Union[str, Iterable[str]],
    summary: str,
    model_name: str = "en_core_sci_sm",
    strict: bool = True,
) -> Dict[str, Union[float, List[str]]]:
    """评估 summary 实体是否全部包含在 ground truth 实体集中。

    strict=True: 如果 summary 列表不完全包含于 ground_truth，reward=0.0.
    strict=False: reward=coverage_ratio.

    Args:
        ground_truth: ground truth 文本或实体列表
        summary: 生成的摘要文本
        model_name: spacy 模型名，默认 en_core_sci_sm
        strict: 是否要求完全覆盖

    Returns:
        dict: 结果，包括 reward, coverage, missing_entities 等。
    """
    if isinstance(ground_truth, str):
        gt_entities = set(extract_entities(ground_truth, model_name=model_name, fallback_to_word_token=True))
    else:
        gt_entities = { _normalize_entity(x) for x in ground_truth if isinstance(x, str) and x.strip() }

    summary_entities = set(extract_entities(summary, model_name=model_name, fallback_to_word_token=True))

    if not summary_entities:
        # summary没有抽取到实体，则奖励0（也可按策略改成1.0）
        reward = 0.0
        coverage = 1.0 if not gt_entities else 0.0
        missing = []
    else:
        missing = sorted([e for e in summary_entities if e not in gt_entities])
        covered = summary_entities - set(missing)
        coverage = len(covered) / len(summary_entities)
        if strict:
            reward = 1.0 if not missing else 0.0
        else:
            reward = coverage

    return {
        "reward": float(reward),
        "coverage": float(coverage),
        "missing_entities": missing,
        "summary_entities": sorted(summary_entities),
        "ground_truth_entities": sorted(gt_entities),
    }


def em_ner_evidence_reward(
    question: str,
    answer: str,
    article: str,
    summary: str,
    model_name: str = "en_core_sci_sm",
    min_evidence_entities: int = 2,
) -> Dict[str, Union[float, bool, int, List[str]]]:
    """EM+NER证据二择一逻辑：防止直接搬答案，用实体证据判断摘要是否可靠。

    1) 首先检查答案是否被 summary 覆盖（Exact Match），避免结论丢失。
    2) 再用 NER 提取 article 和 summary 的实体。
       去掉 summary 里这 3-5 个 answer 词对应实体，检查剩余实体是否至少包含若干article实体。

    如果 summary 只含 answer，或者即使有 answer 但没有额外的article证据，则 reward 低。

    返回：
        "em_hit": bool,
        "evidence_entities": List[str],
        "entity_overlap_ratio": float,
        "reward": float

    奖励设计：
        - EM 必须命中才能进一步得分（否则 reward=0）
        - 如果 summary 是纯命中 answer 与 不足证据 -> reward=0.2
        - 若有额外 evidence 实体且命中 answer -> reward >=0.6
        - 完全满足 -> reward=1.0
    """
    def compatible_normalize(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    answer_norm = compatible_normalize(answer)
    summary_norm = compatible_normalize(summary)
    article_norm = compatible_normalize(article)

    em_hit = False
    if answer_norm and answer_norm in summary_norm:
        em_hit = True

    summary_entities = set(extract_entities(summary, model_name=model_name, fallback_to_word_token=True))
    article_entities = set(extract_entities(article, model_name=model_name, fallback_to_word_token=True))

    # 从 summary 实体中排除 answer 词本身
    answer_entities = set(extract_entities(answer, model_name=model_name, fallback_to_word_token=True))
    summary_entities_ex_ans = summary_entities - answer_entities

    # token 级别统计
    def to_tokens(x: str) -> List[str]:
        x = (x or "").lower()
        x = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", x)
        return [t for t in x.split() if t]

    summary_tokens = to_tokens(summary)
    answer_tokens = to_tokens(answer)
    nonanswer_summary_tokens = [t for t in summary_tokens if t not in answer_tokens]

    # 文章实体 token set
    article_entity_tokens = set()
    for ent in article_entities:
        article_entity_tokens.update(to_tokens(ent))

    # 摘要实体（排除answer后）token set
    summary_entity_tokens_ex_ans = set()
    for ent in summary_entities_ex_ans:
        summary_entity_tokens_ex_ans.update(to_tokens(ent))

    evidence_tokens = [t for t in nonanswer_summary_tokens if t in article_entity_tokens]
    evidence_token_count = len(set(evidence_tokens))

    token_overlap_ratio = float(evidence_token_count / max(1, len(nonanswer_summary_tokens)))

    overlap_entities = sorted(list(summary_entities_ex_ans & article_entities))
    evidence_in_summary = len(overlap_entities)

    # 相对证据占比策略：如果还没命中answer，reward=0；否则基于 token_overlap_ratio 量化
    if not em_hit:
        reward = 0.0
    else:
        if len(nonanswer_summary_tokens) == 0:
            reward = 0.1
        elif token_overlap_ratio < 0.3:
            reward = 0.2
        elif token_overlap_ratio < 0.6:
            reward = 0.6
        else:
            reward = 1.0

    overlap_ratio = token_overlap_ratio

    return {
        "em_hit": em_hit,
        "summary_entities": sorted(summary_entities),
        "article_entities": sorted(article_entities),
        "answer_entities": sorted(answer_entities),
        "summary_entities_ex_answer": sorted(summary_entities_ex_ans),
        "overlap_entities": overlap_entities,
        "evidence_in_summary": evidence_in_summary,
        "entity_overlap_ratio": overlap_ratio,
        "reward": float(reward),
    }


def summary_length_reward(
    article: str,
    summary: str,
    min_ratio: float = 0.15,
    max_ratio: float = 0.25,
) -> Dict[str, Union[float, int, float]]:
    """summary 长度范围奖励：15%~25%为最佳。

    reward:
      - 如果summary_len_ratio在[min,max] -> 1.0
      - 否则 -> 0.0（或可按偏差线性降低/提升）
    """
    def tok_count(x: str) -> int:
        x = (x or "").strip().lower()
        x = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", x)
        return len([t for t in x.split() if t])

    article_len = tok_count(article)
    summary_len = tok_count(summary)

    if article_len == 0:
        ratio = 0.0
    else:
        ratio = summary_len / article_len

    if min_ratio <= ratio <= max_ratio:
        reward = 1.0
    else:
        reward = 0.0

    return {
        "article_len": article_len,
        "summary_len": summary_len,
        "ratio": float(ratio),
        "min_ratio": float(min_ratio),
        "max_ratio": float(max_ratio),
        "reward": float(reward),
    }


# 复用 QA 函数（新增 Think 模式）
import requests
from difflib import SequenceMatcher

def qa_with_vllm_api(
    question: str,
    article: str,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    model_name: str = "Qwen3-4B",
    api_key: str = "token-abc123",
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: int = 120,
    enable_think_mode: bool = True,
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
        "stream": False,
    }

    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vLLM API Error] Question: {question[:50]}... | Error: {e}")
        return None


def tokenize_answer(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    stop_words = {"the", "a", "an", "is", "are", "and", "or", "of", "in", "to", "for"}
    return [token for token in text.split() if token and token not in stop_words]


def f1_score(model_tokens: List[str], golden_tokens: List[str]) -> float:
    if not model_tokens and not golden_tokens:
        return 1.0
    if not model_tokens or not golden_tokens:
        return 0.0
    intersection = len(set(model_tokens) & set(golden_tokens))
    precision = intersection / len(model_tokens)
    recall = intersection / len(golden_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_answer_correct(
    model_answer: str,
    golden_answers: List[str],
    f1_threshold: float = 0.5,
    sim_threshold: float = 0.6,
) -> bool:
    if not model_answer:
        return False

    def clean_basic(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"[^a-z0-9]", "", s)
        return s

    model_clean = clean_basic(model_answer)
    for golden in golden_answers:
        g_clean = clean_basic(golden)
        if model_clean in g_clean or g_clean in model_clean:
            return True
        sim = SequenceMatcher(None, model_clean, g_clean).ratio()
        if sim >= sim_threshold:
            return True
        if f1_score(tokenize_answer(model_answer), tokenize_answer(golden)) >= f1_threshold:
            return True
    return False


def qa_accuracy_reward(
    questions: List[str],
    ground_truths: List[Union[str, List[str]]],
    article: str,
    predictions: Optional[List[str]] = None,
    model_name: str = "Qwen3-4B",
    api_url: str = "http://localhost:8000/v1/chat/completions",
    api_key: str = "token-abc123",
    use_think: bool = True,
    f1_threshold: float = 0.5,
    sim_threshold: float = 0.6,
) -> Dict[str, Union[float, int, List[bool]]]:
    """计算QA准确率作为reward，支持直接传predictions或调用 vLLM API。"""

    n = len(questions)
    if n == 0 or n != len(ground_truths):
        raise ValueError("questions 和 ground_truths 长度要一致且大于0")

    if predictions is None:
        preds = []
        for q in questions:
            ans = qa_with_vllm_api(q, article, api_url=api_url, model_name=model_name, api_key=api_key, enable_think_mode=use_think)
            preds.append(ans or "")
    else:
        if len(predictions) != n:
            raise ValueError("predictions 长度必须等于 questions")
        preds = predictions

    corrects = []
    for pred, gt in zip(preds, ground_truths):
        gts = [gt] if isinstance(gt, str) else list(gt)
        corrects.append(is_answer_correct(pred, gts, f1_threshold=f1_threshold, sim_threshold=sim_threshold))

    accuracy = sum(corrects) / n
    return {
        "questions": questions,
        "predictions": preds,
        "corrects": corrects,
        "accuracy": float(accuracy),
        "reward": float(accuracy),
    }


if __name__ == "__main__":
    import json

    gt_text = "The patient was treated for non-small-cell lung cancer with pembrolizumab and chemotherapy."
    summary_text = "Patient received pembrolizumab for lung cancer."

    result = entity_coverage_reward(gt_text, summary_text, strict=True)
    print(result['reward'])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    import json

    # 简单测试说明
    gt_text = "The patient was treated for non-small-cell lung cancer with pembrolizumab and chemotherapy."
    summary_text = "Patient received pembrolizumab for lung cancer."

    result = entity_coverage_reward(gt_text, summary_text, strict=True)
    print(result['reward'])  # 0.0，因为 summary 中的实体不完全包含于 ground truth 中
    print(json.dumps(result, indent=2, ensure_ascii=False))
