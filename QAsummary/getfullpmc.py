#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import os
import json
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
from datasets import load_dataset, Dataset

# ============ 配置 ============
API_KEY = '657f9d488dfca67ed2032eaf392d49fda609'  # 可选
# API_KEY = ""
EMAIL = "mail.yiao.ya@gmail.com"
USER_AGENT = f"MemAgent-MedSum/1.0 ({EMAIL})"
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": USER_AGENT}
RATE_LIMIT = 0.12  # NCBI 限制：无 Key 每秒 3 次请求

def fetch_from_ncbi(pmid: str, email: str, api_key: str = None) -> Optional[Dict]:
    """从 NCBI E-utilities 获取元数据（推荐首选）"""
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pmid,
        'retmode': 'xml',
        'rettype': 'abstract',
        'tool': 'mesh_fetcher',
        'email': email
    }
    if api_key:
        params['api_key'] = api_key
    
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        
        # 解析 XML（用 xml.etree.ElementTree）
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        
        article = root.find('.//PubmedArticle')
        if article is None:
            return None
        
        # 提取字段
        result = {
            'pmid': pmid,
            'title': article.findtext('.//ArticleTitle'),
            'abstract': article.findtext('.//AbstractText'),
        }
        
        # ✅ 提取 MeSH（NCBI 格式）
        mesh_terms = []
        for mesh in article.findall('.//MeshHeading'):
            term = mesh.findtext('DescriptorName')
            if term:
                mesh_terms.append(term)
        result['mesh_terms_text'] = mesh_terms
        
        # ✅ 提取 author keywords
        keywords = []
        for kw in article.findall('.//KeywordList/Keyword'):
            if kw.text:
                keywords.append(kw.text)
        result['author_keywords'] = keywords
        
        return result
        
    except Exception as e:
        print(f"⚠️ NCBI 获取失败 {pmid}: {e}")
        return None
    
    
def get_pmcid_from_pmid(pmid: str) -> Optional[str]:
    """通过 PMID 查询对应的 PMCID（如果有）"""
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "xml",
        "retmode": "xml",
    }
    if API_KEY:
        params["api_key"] = API_KEY
    
    try:
        time.sleep(RATE_LIMIT)
        resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        
        # 解析 XML 提取 PMCID
        root = ET.fromstring(resp.content)
        # 路径: PubmedArticle -> MedlineCitation -> ArticleIdList -> ArticleId[@IdType="pmc"]
        for article_id in root.findall(".//ArticleId[@IdType='pmc']"):
            pmcid = article_id.text
            if pmcid:
                print(f"🔗 找到 PMCID: {pmcid}")
                return pmcid
        print(f"⚠️  PMID {pmid} 没有关联的 PMC 全文")
        return None
        
    except Exception as e:
        print(f"❌ 查询 PMCID 失败: {e}")
        return None
    

def fetch_pmc_fulltext(
    pmcid: str,
    rettype: str = "xml",  # xml | text | nlmxml
    save_path: Optional[str] = None
) -> Optional[str]:
    """
    从 PMC 下载全文
    
    rettype 说明:
    - 'xml': JATS XML 格式（结构完整，推荐）
    - 'text': 纯文本（去格式，适合阅读）
    - 'nlmxml': NLM DTD XML（传统格式）
    """
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": rettype,
        "retmode": "xml" if rettype == "xml" else "text",
    }
    if API_KEY:
        params["api_key"] = API_KEY
    
    try:
        time.sleep(RATE_LIMIT)
        resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        
        content = resp.text
        
        # 保存文件
        if save_path:
            ext = "xml" if rettype == "xml" else "txt"
            path = save_path if save_path.endswith(f".{ext}") else f"{save_path}.{ext}"
            if os.path.exists(path):
                print(f"⚠️ 文件已存在: {path}")
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"💾 全文已保存: {path}")
        
        return content
        
    except Exception as e:
        print(f"❌ 下载全文失败: {e}")
        return None

def extract_plain_text_from_pmc_xml(xml_content: str) -> str:
    """从 PMC XML 中提取可读的纯文本（标题+摘要+正文）"""
    try:
        root = ET.fromstring(xml_content.encode('utf-8') if isinstance(xml_content, str) else xml_content)
        
        parts = []
        
        # 标题
        title = root.find(".//article-title")
        if title is not None and title.text:
            parts.append(f"Title: {title.text.strip()}\n")
        
        # 摘要
        abstract = root.find(".//abstract")
        if abstract is not None:
            abs_text = ' '.join(abstract.itertext()).strip()
            parts.append(f"\nAbstract:\n{abs_text}\n")
        
        # 正文段落
        body = root.find(".//body")
        if body is not None:
            for p in body.findall(".//p"):
                p_text = ' '.join(p.itertext()).strip()
                if p_text and len(p_text) > 20:  # 过滤短文本
                    parts.append(p_text)
        
        return "\n\n".join(parts)
        
    except Exception as e:
        print(f"⚠️ 文本提取失败: {e}")
        return xml_content  # 降级返回原始内容

def download_fulltext_by_pmid(
    pmid: str,
    pmcid: Optional[str] = None,
    output_dir: str = "./downloads",
    prefer_format: str = "xml"  # xml | text
) -> Optional[str]:
    """一键下载：PMID → 全文"""
    print(f"🔍 开始下载 PMID: {pmid}")
    print("-" * 60)
    
    # Step 1: 获取 PMCID
    if not pmcid:
        pmcid = get_pmcid_from_pmid(pmid)
    if not pmcid:
        print("❌ 无法获取全文：该文献可能未在 PMC 收录")
        print("💡 备选方案: 尝试通过 DOI 访问出版商网站")
        return None
    
    # Step 2: 下载全文
    output_path = f"{output_dir}/pmid_{pmid}_fulltext"
    content = fetch_pmc_fulltext(pmcid, rettype=prefer_format, save_path=output_path)
    
    if not content:
        return None
    
    # Step 3: 如果是 XML，额外生成纯文本版本便于阅读
    if prefer_format == "xml":
        plain_text = extract_plain_text_from_pmc_xml(content)
        txt_path = f"{output_path}.readable.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(plain_text)
        print(f"📝 可读文本已保存: {txt_path}")
    
    print("✅ 下载完成！")
    return output_path

def filter_fulltext_dataset(dataset: Dataset, output_path: str = 'filtered_fulltext.parquet', text_dir: str = './texts'):
    """筛选有免费全文的数据并保存"""
    
    # 存储结果
    filtered_data = []
    stats = {'total': 0, 'has_full_text': 0, 'no_full_text': 0, 'error': 0}
    
    print(f"📊 开始筛选，共 {len(dataset)} 条数据...\n")
    
    for idx, item in enumerate(tqdm(dataset, desc="Checking PMID")):
        pmid = str(item.get('pmid', ''))
        
        # 跳过无效 PMID
        if not pmid or not pmid.isdigit():
            stats['no_pdf'] += 1
            continue
        
        stats['total'] += 1
        
        # 1. 获取 PMCID
        pmcid = get_pmcid_from_pmid(pmid)
        if not pmcid:
            stats['no_full_text'] += 1
            continue
        # 2. 下载 PDF
        text_path = download_fulltext_by_pmid(pmid, pmcid, text_dir)
        
        if not text_path:
            stats['no_full_text'] += 1
            continue

        item['text_path'] = text_path
        
        result = fetch_from_ncbi(pmid, EMAIL, api_key=API_KEY)
        if not result:
            stats['no_full_text'] += 1
            continue
        item['mesh_terms_text'] = result['mesh_terms_text']
        item['author_keywords'] = result['author_keywords']
        item['pmcid'] = pmcid
        
        stats['has_full_text'] += 1
        filtered_data.append(item)  # 只保存有全文的
        
        # 速率限制
        if idx % 100 == 0 and idx > 0:
            time.sleep(1)
            print(f"\n📈 进度：{idx}/{len(dataset)} | 有全文：{stats['has_full_text']} ({stats['has_full_text']/idx*100:.1f}%)")
        
        time.sleep(RATE_LIMIT)
    
    # 保存结果
    if filtered_data:
        # 转换为 Dataset 并保存
        filtered_dataset = Dataset.from_list(filtered_data)
        filtered_dataset.to_parquet(output_path)
        print(f"\n✅ 已保存 {len(filtered_data)} 条有全文的数据到：{output_path}")
    
    # 保存统计信息
    stats_path = output_path.replace('.parquet', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 打印统计
    print("\n" + "="*60)
    print("📊 筛选统计")
    print("="*60)
    print(f"总数据量：{stats['total']}")
    print(f"有免费 PDF：{stats['has_pdf']} ({stats['has_pdf']/max(stats['total'],1)*100:.1f}%)")
    print(f"无免费 PDF：{stats['no_pdf']}")
    print(f"检查错误：{stats['error']}")
    print("="*60)
    
    return filtered_data, stats

from datasets import load_dataset
if __name__ == '__main__':
    # 1. 加载原始数据集
    print("📚 加载数据集...")
    datas = load_dataset(
        'parquet',
        data_files='/nfsdata3/yiao/data/PaperSearchQA/data/test-00000-of-00001.parquet'
    )
    
    # 2. 获取 train split
    train_dataset = datas['train']
    print(f"✅ 加载完成，共 {len(train_dataset)} 条数据")
    print(f"   字段：{train_dataset.column_names}\n")
    
    # 3. 筛选有全文的数据
    filtered_data, stats = filter_fulltext_dataset(
        train_dataset,
        output_path='/nfsdata3/yiao/data/PaperSearchQA_summary/data/test_with_fulltext.parquet',
        text_dir='/nfsdata3/yiao/data/PaperSearchQA_summary/fulltext'
    )
    
    # 4. 可选：查看前几条有全文的数据
    if filtered_data:
        print("\n📋 前 3 条有全文的样本:")
        for i, item in enumerate(filtered_data[:3]):
            print(f"\n--- 样本 {i+1} ---")
            print(f"PMID: {item['pmid']}")
            print(f"PMCID: {item['pmcid']}")
            print(f"标题：{item['paper_title'][:50]}...")
            print(f"PDF: {item['pdf_url']}")