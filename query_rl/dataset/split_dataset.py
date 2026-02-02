import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def split_parquet_dataset(input_file, train_ratio=0.8, random_state=42):
    """
    将单个parquet文件分割为训练集和验证集
    
    Args:
        input_file: 输入的parquet文件路径
        train_ratio: 训练集比例，默认0.8
        random_state: 随机种子，默认42
    """
    # 读取parquet文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_parquet(input_file)
    
    print(f"原始数据集大小: {len(df)}")
    
    # 分割数据集
    train_df, val_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state,
        shuffle=True  # 打乱数据
    )
    
    print(f"训练集大小: {len(train_df)} ({train_ratio*100:.1f}%)")
    print(f"验证集大小: {len(val_df)} ({(1-train_ratio)*100:.1f}%)")
    
    # 获取输出文件路径
    input_path = Path(input_file)
    base_path = input_path.parent
    stem = input_path.stem
    
    # 保存分割后的数据
    train_file = base_path / f"split/{stem}_train.parquet"
    val_file = base_path / f"split/{stem}_val.parquet"
    
    print(f"保存训练集到: {train_file}")
    train_df.to_parquet(train_file)
    
    print(f"保存验证集到: {val_file}")
    val_df.to_parquet(val_file)
    
    print("数据集分割完成！")
    
    return str(train_file), str(val_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将单个parquet文件分割为训练集和验证集")
    parser.add_argument("--input_file", type=str, default="/nfsdata/yiao/PubMedQA/pqa_labeled/train-00000-of-00001.parquet", help="输入的parquet文件路径")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，默认0.8")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子，默认42")
    
    args = parser.parse_args()
    
    train_file, val_file = split_parquet_dataset(
        input_file=args.input_file,
        train_ratio=args.train_ratio,
        random_state=args.random_state
    )
    
    print("\n使用说明:")
    print(f"- 将 {train_file} 用于训练数据")
    print(f"- 将 {val_file} 用于验证数据")
    print("- 在VERL配置中分别设置 data.train_files 和 data.val_files 参数")