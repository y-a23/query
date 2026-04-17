import pandas as pd
import json
from collections import defaultdict
import numpy as np

# 1. 读取 parquet 文件
datapath = "/nfsdata3/yiao/data/PaperSearchQA_summary/data/train_with_fulltext.parquet"
savepath = "/nfsdata3/yiao/data/PaperSearchQA_summary/data/train_merged_by_pmid.parquet"
df = pd.read_parquet(datapath).convert_dtypes()

# 2. 按 pmid 分组，合并 questions
def merge_by_pmid(df):
    merged = {}
    
    for pmid, group in df.groupby('pmid'):
        # 提取该 pmid 的公共字段（每条记录都相同的字段）
        first_row = group.iloc[0]
        
        # 辅助函数：将 ndarray 转换为 list
        def convert_ndarray(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        merged_entry = {
            'pmid': pmid,
            'pmcid': convert_ndarray(first_row.get('pmcid')),
            'paper_title': first_row.get('paper_title'),
            'cat': first_row.get('cat'),
            'cat_num': first_row.get('cat_num'),
            'text_path': first_row.get('text_path'),
            'mesh_terms_text': convert_ndarray(first_row.get('mesh_terms_text', [])),
            'author_keywords': convert_ndarray(first_row.get('author_keywords', [])),
            # 将多个 question 合并为列表
            'questions': []
        }
        
        # 遍历该 pmid 下的所有 question
        for _, row in group.iterrows():
            q_entry = {
                'question': row['question'],
                'question_original': row.get('question_original'),
                'answer': row.get('answer'),
                'golden_answers': convert_ndarray(row.get('golden_answers', [])),
                'is_paraphrased': row.get('is_paraphrased', False)
            }
            merged_entry['questions'].append(q_entry)
        
        merged[pmid] = merged_entry
    
    return merged

# 执行合并
merged_data = merge_by_pmid(df)

# 3. 保存为 JSON 文件
with open('train_merged_by_pmid.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

# 4. 保存为 parquet（如果需要保持列式存储）
# 先将合并后的结构展平为列表
records = list(merged_data.values())
df_merged = pd.DataFrame(records)
df_merged.to_parquet(savepath, index=False)

print(f"✅ 处理完成！")
print(f"   - 原始记录数: {len(df)}")
print(f"   - 合并后 pmid 数: {len(merged_data)}")
print(f"   - 平均每个 pmid 的 question 数: {len(df)/len(merged_data):.2f}")