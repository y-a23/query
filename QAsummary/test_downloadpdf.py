#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 PMID 是否有 PMC 免费全文，并下载 PDF
"""

import requests
import time
import os
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

PMC_API_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles"
EMAIL = "your_email@example.com"  # ⚠️ 替换为你的邮箱
RATE_LIMIT = 0.3  # 秒/请求


# 支持的全文格式及配置（按优先级排序）
FULLTEXT_FORMATS = {
    'pdf': {'path': 'pdf/', 'ext': '.pdf', 'mime': 'application/pdf'},
    'xml': {'path': 'xml/', 'ext': '.xml', 'mime': 'application/xml'},
    'html': {'path': '', 'ext': '.html', 'mime': 'text/html'},  # 空路径表示文章主页
}
FORMAT_PRIORITY = ['pdf', 'xml', 'html']  # 下载优先级

# ============== 步骤 2: 检测可用格式 ==============
def check_available_formats(pmcid: str, email: str) -> List[str]:
    """
    检测 PMCID 文章有哪些可用的全文格式
    Returns: 可用格式列表，按优先级排序，如 ['pdf', 'xml']
    """
    available = []
    headers = {'User-Agent': 'Mozilla/5.0 (Fulltext Downloader)'}
    
    for fmt in FORMAT_PRIORITY:
        config = FULLTEXT_FORMATS[fmt]
        # 构建检测 URL
        if config['path']:
            check_url = f"{PMC_BASE_URL}/{pmcid}/{config['path']}"
        else:
            check_url = f"{PMC_BASE_URL}/{pmcid}/"
        
        try:
            # 先用 HEAD 请求快速检测
            resp = requests.head(check_url, headers=headers, timeout=15, allow_redirects=True)
            if resp.status_code == 405:  # Method Not Allowed，降级为 GET
                resp = requests.get(check_url, headers=headers, timeout=15, stream=True)
                resp.raw.read(1024)
                resp.close()
            
            if 200 <= resp.status_code < 300:
                content_type = resp.headers.get('Content-Type', '').lower()
                # HTML 格式特殊处理：避免下载到 404 页面
                if fmt == 'html' or config['mime'] in content_type or 'text/html' in content_type:
                    available.append(fmt)
        except Exception:
            continue  # 检测失败不中断
    
    return available

# ============== 步骤 1: PMID → PMCID 转换 ==============
def get_pmcid_from_pmid(pmid: str, email: str) -> Optional[str]:
    """
    通过 NCBI ID Converter 获取 PMCID
    返回: PMCID (如 'PMC1234567') 或 None
    """
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        'ids': pmid,
        'format': 'json',
        'tool': 'pdf_downloader',
        'email': email
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('records') and len(data['records']) > 0:
            record = data['records'][0]
            # 检查是否有 pmcid 字段
            if record.get('pmcid'):
                return record['pmcid']
        
        return None
        
    except Exception as e:
        print(f"⚠️ 获取 PMCID 失败 {pmid}: {e}")
        return None


# ============== 步骤 3: 下载 PMC 全文（支持多格式） ==============
def download_pmc_fulltext(pmcid: str, 
                          output_dir: str = './fulltexts',
                          preferred_formats: List[str] = None) -> Optional[Dict]:
    """
    从 PMC 下载全文，支持 PDF/XML/HTML 多种格式
    
    Returns:
        下载结果字典: {'format': 'pdf', 'path': '...', 'size_mb': 1.23, 'url': '...'}
        或 None（下载失败）
    """
    if preferred_formats is None:
        preferred_formats = FORMAT_PRIORITY
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. 检测可用格式
        available = check_available_formats(pmcid, EMAIL)
        if not available:
            print(f"⚠️ 未找到可用全文格式 {pmcid}")
            return None
        
        # 2. 按优先级选择第一个可用格式
        chosen_fmt = None
        for fmt in preferred_formats:
            if fmt in available:
                chosen_fmt = fmt
                break
        if not chosen_fmt:
            chosen_fmt = available[0]  # 兜底
        
        config = FULLTEXT_FORMATS[chosen_fmt]
        
        # 3. 构建下载 URL 和输出路径
        if config['path']:
            download_url = f"{PMC_BASE_URL}/{pmcid}/{config['path']}"
        else:
            download_url = f"{PMC_BASE_URL}/{pmcid}/"
        
        output_path = os.path.join(output_dir, f"{pmcid}{config['ext']}")
        
        # 4. 检查文件是否已存在
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1e6
            print(f"✅ 文件已存在 [{chosen_fmt.upper()}]: {output_path} ({size_mb:.2f} MB)")
            return {
                'format': chosen_fmt,
                'path': output_path,
                'size_mb': size_mb,
                'url': download_url
            }
        
        # 5. 执行下载
        headers = {'User-Agent': 'Mozilla/5.0 (Fulltext Downloader)', 'From': EMAIL}
        resp = requests.get(download_url, headers=headers, timeout=60, stream=True)
        
        if resp.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            size_mb = os.path.getsize(output_path) / 1e6
            print(f"✅ 下载成功 [{chosen_fmt.upper()}]: {output_path} ({size_mb:.2f} MB)")
            return {
                'format': chosen_fmt,
                'path': output_path,
                'size_mb': size_mb,
                'url': download_url,
                'all_available': available  # 记录所有可用格式供参考
            }
        else:
            print(f"⚠️ 下载失败 {pmcid} [{chosen_fmt}]: HTTP {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"⚠️ 下载异常 {pmcid}: {e}")
        return None


# ============== 步骤 3: 完整流程 ==============
def fetch_pdf_for_pmid(pmid: str, email: str, output_dir: str = './pdfs') -> Dict:
    """
    完整流程：PMID → 检查 PMCID → 下载 PDF
    
    Returns:
        结果字典，包含状态和路径
    """
    
    result = {
        'pmid': pmid,
        'has_pmc': False,
        'pmcid': None,
        'pdf_path': None,
        'error': None
    }
    
    # 1. 获取 PMCID
    pmcid = get_pmcid_from_pmid(pmid, email)
    
    if not pmcid:
        result['error'] = 'no_pmcid'
        return result
    
    result['has_pmc'] = True
    result['pmcid'] = pmcid
    
    # 2. 下载 PDF
    pdf_path = download_pmc_fulltext(pmcid, output_dir)
    
    if pdf_path:
        result['pdf_path'] = pdf_path
    else:
        result['error'] = 'download_failed_or_no_pdf'
    
    return result


# ============== 批量处理 ==============
def batch_download_pdfs(pmid_list: List[str], 
                        email: str, 
                        output_dir: str = './pdfs',
                        delay: float = 0.5) -> List[Dict]:
    """批量下载 PDF"""
    
    results = []
    success_count = 0
    
    print(f"📥 开始处理 {len(pmid_list)} 个 PMID...")
    
    for pmid in tqdm(pmid_list, desc="Downloading"):
        result = fetch_pdf_for_pmid(str(pmid), email, output_dir)
        results.append(result)
        
        if result['pdf_path']:
            success_count += 1
        
        # 速率限制（遵守 NCBI 政策）
        time.sleep(delay)
    
    # 统计
    total = len(pmid_list)
    has_pmc = sum(1 for r in results if r['has_pmc'])
    
    print(f"\n📊 下载统计:")
    print(f"   总 PMID 数：{total}")
    print(f"   有 PMC 全文：{has_pmc} ({has_pmc/total*100:.1f}%)")
    print(f"   PDF 下载成功：{success_count} ({success_count/total*100:.1f}%)")
    print(f"   无 PMC/下载失败：{total - has_pmc}")
    
    return results


# ============== 测试 ==============
if __name__ == '__main__':
    import json
    
    # 测试 PMID（部分有 PMC，部分没有）
    test_pmids = [
        '22163600',  # 可能没有 PMC
        '31537895',  # 测试
        '28456789',  # 测试
        '35082362',  # 较新文章，可能有 PMC
        '30012345',  # 随机
    ]
    
    EMAIL = "your_email@example.com"  # ⚠️ 替换为你的邮箱
    
    print("🔍 开始测试 PDF 获取...\n")
    
    for pmid in test_pmids:
        print(f"=" * 60)
        print(f"📄 PMID: {pmid}")
        print("-" * 60)
        
        result = fetch_pdf_for_pmid(pmid, EMAIL)
        
        if result['has_pmc']:
            print(f"✅ 有 PMC: {result['pmcid']}")
            if result['pdf_path']:
                print(f"✅ PDF 已下载: {result['pdf_path']}")
            else:
                print(f"⚠️ 有 PMC 但无 PDF（可能只有 XML）")
        else:
            print(f"❌ 无 PMC 全文: {result['error']}")
        
        print()
        time.sleep(0.5)