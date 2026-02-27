import os
import json
import glob
from typing import Iterator, Dict, Any, Optional
import logging
from tqdm import tqdm
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PubmedDataLoader:
    """PubMed数据加载器，用于读取JSONL文件并搜索特定PMID"""
    
    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 包含JSONL文件的目录路径
        """
        self.data_dir = data_dir
        self.jsonl_files = self._get_jsonl_files()
        logger.info(f"找到 {len(self.jsonl_files)} 个JSONL文件")
    
    def _get_jsonl_files(self) -> list:
        """获取目录中所有的JSONL文件"""
        pattern = os.path.join(self.data_dir, "*.jsonl")
        files = glob.glob(pattern)
        files.sort()  # 按文件名排序
        return files
    
    def load_single_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        加载单个JSONL文件
        
        Args:
            file_path: JSONL文件路径
            
        Yields:
            解析后的JSON对象
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    
                    try:
                        data = json.loads(line)
                        yield data
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析错误 {file_path}:{line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
        except Exception as e:
            logger.error(f"读取文件时出错 {file_path}: {e}")
    
    def load_all_files(self) -> Iterator[Dict[str, Any]]:
        """
        加载所有JSONL文件
        
        Yields:
            解析后的JSON对象
        """
        total_processed = 0
        for file_path in self.jsonl_files:
            logger.info(f"正在处理文件: {os.path.basename(file_path)}")
            file_count = 0
            
            for data in self.load_single_file(file_path):
                yield data
                file_count += 1
                total_processed += 1
                
                # 每处理10000条记录打印一次进度
                if total_processed % 10000 == 0:
                    logger.info(f"已处理 {total_processed} 条记录")
            
            logger.info(f"文件 {os.path.basename(file_path)} 处理完成，共 {file_count} 条记录")
    
    def search_pmid(self, target_pmid: int) -> Optional[Dict[str, Any]]:
        """
        搜索指定的PMID
        
        Args:
            target_pmid: 要搜索的PMID
            
        Returns:
            找到的记录字典，如果未找到则返回None
        """
        logger.info(f"开始搜索 PMID: {target_pmid}")
        logger.info("=" * 50)
        
        total_records = 0
        matched_records = []
        start_time = time.time()
        
        # 使用tqdm显示进度条
        with tqdm(total=len(self.jsonl_files), desc="搜索进度", unit="文件") as pbar:
            for file_idx, file_path in enumerate(self.jsonl_files):
                filename = os.path.basename(file_path)
                file_record_count = 0
                file_matches = []
                
                # 处理单个文件
                for record in self.load_single_file(file_path):
                    total_records += 1
                    file_record_count += 1
                    
                    # 检查PMID字段
                    if 'PMID' in record:
                        try:
                            pmid = int(record['PMID'])
                            if pmid == target_pmid:
                                file_matches.append({
                                    'record': record,
                                    'file_index': file_idx,
                                    'filename': filename,
                                    'record_number': file_record_count
                                })
                                matched_records.extend(file_matches)
                                
                                # 实时打印匹配信息
                                logger.info(f"\n🔍 在文件 {filename} 中找到匹配!")
                                logger.info(f"   记录编号: {file_record_count}")
                                logger.info(f"   标题: {record.get('title', 'N/A')[:100]}...")
                                logger.info(f"   PMID: {pmid}")
                        except (ValueError, TypeError):
                            # 如果PMID不是有效的数字，跳过
                            continue
                
                # 打印文件处理总结
                if file_matches:
                    logger.info(f"✅ 文件 {filename} 处理完成 - 找到 {len(file_matches)} 个匹配项")
                else:
                    logger.info(f"📄 文件 {filename} 处理完成 - 0 个匹配项 ({file_record_count} 条记录)")
                
                pbar.update(1)
                pbar.set_postfix({
                    '已处理文件': file_idx + 1,
                    '总记录数': total_records,
                    '匹配数': len(matched_records)
                })
        
        # 搜索完成总结
        elapsed_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("搜索完成统计:")
        logger.info(f"  总耗时: {elapsed_time:.2f} 秒")
        logger.info(f"  处理文件数: {len(self.jsonl_files)}")
        logger.info(f"  总记录数: {total_records}")
        logger.info(f"  匹配记录数: {len(matched_records)}")
        logger.info(f"  平均处理速度: {total_records/elapsed_time:.0f} 条/秒")
        logger.info("=" * 50)
        
        # 返回结果
        if matched_records:
            logger.info(f"🎉 找到 {len(matched_records)} 个匹配的PMID {target_pmid}")
            for i, match in enumerate(matched_records, 1):
                record = match['record']
                logger.info(f"\n--- 匹配项 {i} ---")
                logger.info(f"文件: {match['filename']}")
                logger.info(f"记录编号: {match['record_number']}")
                logger.info(f"标题: {record.get('title', 'N/A')}")
                logger.info(f"PMID: {record['PMID']}")
                logger.info(f"内容长度: {len(record.get('content', ''))} 字符")
            return matched_records[0]['record']  # 返回第一个匹配项
        else:
            logger.info(f"❌ 未找到PMID {target_pmid}")
            return None
    
    def count_total_records(self) -> int:
        """
        统计总记录数
        
        Returns:
            总记录数
        """
        logger.info("开始统计总记录数...")
        count = 0
        
        for _ in self.load_all_files():
            count += 1
            
        logger.info(f"总记录数: {count}")
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            包含统计数据的字典
        """
        logger.info("生成数据统计信息...")
        
        stats = {
            'total_files': len(self.jsonl_files),
            'total_records': 0,
            'pmid_range': {'min': float('inf'), 'max': float('-inf')},
            'files_info': []
        }
        
        for file_path in self.jsonl_files:
            file_record_count = 0
            file_min_pmid = float('inf')
            file_max_pmid = float('-inf')
            
            for record in self.load_single_file(file_path):
                stats['total_records'] += 1
                file_record_count += 1
                
                if 'PMID' in record:
                    try:
                        pmid = int(record['PMID'])
                        file_min_pmid = min(file_min_pmid, pmid)
                        file_max_pmid = max(file_max_pmid, pmid)
                        
                        stats['pmid_range']['min'] = min(stats['pmid_range']['min'], pmid)
                        stats['pmid_range']['max'] = max(stats['pmid_range']['max'], pmid)
                    except (ValueError, TypeError):
                        continue
            
            # 处理无穷大的情况
            if file_min_pmid == float('inf'):
                file_min_pmid = None
            if file_max_pmid == float('-inf'):
                file_max_pmid = None
                
            stats['files_info'].append({
                'filename': os.path.basename(file_path),
                'record_count': file_record_count,
                'pmid_min': file_min_pmid,
                'pmid_max': file_max_pmid
            })
        
        # 处理总体统计中的无穷大情况
        if stats['pmid_range']['min'] == float('inf'):
            stats['pmid_range']['min'] = None
        if stats['pmid_range']['max'] == float('-inf'):
            stats['pmid_range']['max'] = None
            
        return stats

def main():
    """主函数示例"""
    # 设置数据目录
    data_dir = "/nfsdata3/yiao/yiao/medRAG/pubmed/chunk"
    
    # 创建数据加载器
    loader = PubmedDataLoader(data_dir)
    
    # 搜索特定的PMID
    target_pmid = 9581781
    print(f"开始搜索 PMID: {target_pmid}")
    print("=" * 60)
    
    result = loader.search_pmid(target_pmid)
    
    if result:
        print(f"\n{'='*60}")
        print(f"🎉 搜索完成! 找到了PMID {target_pmid}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"❌ 搜索完成! 未找到PMID {target_pmid}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()