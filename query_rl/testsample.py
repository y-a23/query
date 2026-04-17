import pandas as pd

# 1. 读取原始验证集
df = pd.read_parquet("/nfsdata3/yiao/data/PaperSearchQA_verl/test.parquet")

# 2. 随机抽取 500 条 (固定 seed 保证可复现)
df_sample = df.sample(n=400, random_state=42) 

# 3. 保存为新文件
df_sample.to_parquet("/nfsdata3/yiao/data/PaperSearchQA_verl/val_400.parquet", index=False)

print(f"已生成验证集：{len(df_sample)} 条")