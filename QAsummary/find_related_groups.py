#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 2: LLM 从聚类中精选 2-3 篇并合成多跳问题
"""

import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from openai import OpenAI  # 或用其他 LLM API

class MultiHopGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def select_and_generate(self, cluster: Dict) -> Optional[Dict]:
        """LLM 精选文章并生成多跳问题"""
        
        # 构建 prompt
        abstracts_text = ""
        for i, abs_info in enumerate(cluster['abstracts'][:10]):  # 限制最多 10 篇
            abstracts_text += f"""
[Article {i+1}]
PMID: {abs_info['pmid']}
Title: {abs_info['title'][:200]}
Abstract: {abs_info['abstract'][:800]}
"""
        
        prompt = f"""
You are an expert at creating multi-hop biomedical QA pairs.

Given a cluster of related PubMed articles (sharing keywords: {cluster['shared_keywords'][:5]}):

{abstracts_text}

TASK:
1. Select 2-3 articles that can form a GOOD multi-hop question
   - Articles should have COMPLEMENTARY information (not redundant)
   - There should be a clear reasoning chain connecting them
   - Avoid articles that are too similar

2. Generate 1-2 multi-hop questions that require information from ALL selected articles

OUTPUT FORMAT (JSON only):
{{
    "selected_pmids": ["pmid1", "pmid2"],
    "selection_reason": "why these articles work for multi-hop",
    "questions": [
        {{
            "question": "...",
            "answer": "...",
            "reasoning_chain": ["step 1", "step 2"],
            "hop_count": 2
        }}
    ],
    "quality_score": 0-10,
    "skip_reason": null or "why skipped if quality < 6"
}}

CRITERIA FOR GOOD MULTI-HOP:
- Answer CANNOT be found in a single article
- Clear logical connection between articles
- Factual, verifiable answer
- Biomedical domain relevant

If no good multi-hop can be formed, set quality_score < 6 and explain why.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 添加元数据
            result['cluster_id'] = cluster['cluster_id']
            result['all_pmids'] = cluster['pmids']
            result['shared_keywords'] = cluster['shared_keywords']
            
            return result
        
        except Exception as e:
            return {
                'cluster_id': cluster['cluster_id'],
                'error': str(e),
                'quality_score': 0
            }
    
    def batch_generate(self, 
                       clusters: List[Dict], 
                       output_path: str,
                       sample_size: int = None,
                       quality_threshold: int = 6) -> List[Dict]:
        """批量生成多跳 QA"""
        
        if sample_size:
            clusters = clusters[:sample_size]
        
        results = []
        high_quality_count = 0
        
        print(f"🤖 开始 LLM 生成 ({len(clusters)} 个聚类)...")
        
        for cluster in tqdm(clusters, desc="Generating"):
            result = self.select_and_generate(cluster)
            
            if result.get('quality_score', 0) >= quality_threshold:
                high_quality_count += 1
                results.append(result)
            
            # 每 50 个保存一次检查点
            if len(results) % 50 == 0:
                self._save_checkpoint(results, output_path + '.checkpoint')
        
        # 保存最终结果
        self._save_results(results, output_path)
        
        print(f"\n✅ 生成完成:")
        print(f"   处理聚类：{len(clusters)}")
        print(f"   高质量 QA: {high_quality_count} ({high_quality_count/len(clusters)*100:.1f}%)")
        print(f"   输出文件：{output_path}")
        
        return results
    
    def _save_results(self, results: List[Dict], output_path: str):
        """保存结果"""
        # 展平问题列表
        flat_qa = []
        for r in results:
            for q in r.get('questions', []):
                qa = {
                    'question': q.get('question'),
                    'answer': q.get('answer'),
                    'pmid_list': r.get('selected_pmids'),
                    'cluster_id': r.get('cluster_id'),
                    'shared_keywords': r.get('shared_keywords'),
                    'reasoning_chain': q.get('reasoning_chain'),
                    'hop_count': q.get('hop_count', len(r.get('selected_pmids', []))),
                    'quality_score': r.get('quality_score'),
                    'selection_reason': r.get('selection_reason')
                }
                flat_qa.append(qa)
        
        pd.DataFrame(flat_qa).to_parquet(output_path, index=False)
    
    def _save_checkpoint(self, results: List[Dict], checkpoint_path: str):
        """保存检查点"""
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == '__main__':
    import os
    
    # 加载聚类
    clusters = pd.read_parquet('./output/clusters_5-10.parquet').to_dict('records')
    
    # 初始化生成器
    generator = MultiHopGenerator(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model='gpt-4o-mini'  # 成本低，适合此任务
    )
    
    # 批量生成
    generator.batch_generate(
        clusters=clusters,
        output_path='./output/multihop_qa_final.parquet',
        sample_size=1000,  # 先测试 1000 个聚类
        quality_threshold=6
    )