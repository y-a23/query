#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scispacy_test.py

示例：安装 ScispaCy 并从 QA 数据集加载文本做实体抽取测试。

如果当前环境不支持 ScispaCy（Python 3.12 可能导致 thinc 编译问题），
会优先使用 spaCy 自带模型 `en_core_web_sm`。
"""

import os
import sys
import traceback
from pathlib import Path


def install_instructions():
    print("\n--- scispacy 安装建议 ---")
    print("1) 建议使用 Python 3.10 的 conda 环境。")
    print("   conda create -n verl_scispacy python=3.10 -y")
    print("   conda activate verl_scispacy")
    print("   pip install scispacy spacy")
    print("   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz")
    print("2) 或使用 spaCy 通用模型：pip install spacy && python -m spacy download en_core_web_sm")
    print("3) 作为备用：pip install spacy_medspacy（无需C编译, 仅示例）。\n")


def load_sample_texts_from_qa_test1(parquet_path=None, max_samples=3):
    # 尝试用 qa_test1.py 提供的 parquet 数据加载函数
    candidate_texts = []
    if parquet_path is None:
        parquet_path = Path("/home/yiao/verl/QAsummary/data_sample.parquet")

    try:
        from qa_test1 import load_parquet_dataset
    except Exception as e:
        print(f"[WARNING] 无法 import qa_test1.load_parquet_dataset: {e}")
        return candidate_texts

    if not Path(parquet_path).is_file():
        print(f"[WARNING] parquet_path 不存在：{parquet_path}")
        return candidate_texts

    try:
        dataset = load_parquet_dataset(str(parquet_path))
        for row in dataset:
            tex_path = row.get("text_path")
            if not tex_path:
                continue
            full_path = Path(tex_path + ".readable.txt")
            if full_path.exists():
                candidate_texts.append(full_path.read_text(encoding="utf-8"))
            if len(candidate_texts) >= max_samples:
                break
    except Exception as e:
        print(f"[ERROR] 读取数据时异常: {e}")
        traceback.print_exc()

    return candidate_texts


def main():
    try:
        import scispacy
        print(f"scispacy version: {scispacy.__version__}")
    except Exception as e:
        print("[NOTICE] scispacy 未安装或导入失败：", e)

    try:
        import spacy
        print(f"spaCy version: {spacy.__version__}")
    except ImportError as e:
        print("[ERROR] spaCy 需要先安装：", e)
        install_instructions()
        sys.exit(1)

    # 1. 尝试加载 scispacy 模型，失败则降级到 en_core_web_sm
    nlp = None
    model_names = ["en_core_sci_sm", "en_core_web_sm"]
    for model in model_names:
        try:
            print(f"尝试加载模型: {model}")
            nlp = spacy.load(model)
            print(f"成功加载模型: {model}")
            break
        except Exception as e:
            print(f"加载模型 {model} 失败: {e}")

    if nlp is None:
        print("无法加载任何模型，请先安装模型。")
        install_instructions()
        sys.exit(1)

    # 2. 读取样例文本。先尝试 qa_test1 提供的路径；无则用自定义示例。
    texts = load_sample_texts_from_qa_test1(parquet_path=None, max_samples=3)
    if not texts:
        texts = [
            "Diabetes is a chronic condition that affects the way the body processes blood glucose.",
            "Aspirin can reduce the risk of heart attack by inhibiting platelet aggregation.",
            "Cancer immunotherapy with checkpoint inhibitors has shown great promise in melanoma treatment."
        ]

    print(f"\n准备处理文本数量: {len(texts)}")

    for i, txt in enumerate(texts, start=1):
        txt = txt.strip()
        if not txt:
            continue
        doc = nlp(txt)
        print('\n--- 文本 {}/{} ---'.format(i, len(texts)))
        print('原文:', txt)

        ent_lines = []
        for ent in doc.ents:
            ent_lines.append(f"[{ent.text}] {ent.label_} (start={ent.start_char}, end={ent.end_char})")
        if ent_lines:
            print('实体抽取结果:')
            print('\n'.join(ent_lines))
        else:
            print('实体抽取结果: (无)')

        print('名词短语(Noun chunks):')
        for chunk in doc.noun_chunks:
            print(f"  - {chunk.text}")

        print('依存句法样例:')
        for token in doc[:min(20, len(doc))]:
            print(f"{token.text}\t{token.dep_}\t{token.head.text}")

    print("\n测试完成。")


if __name__ == '__main__':
    main()
