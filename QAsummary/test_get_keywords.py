#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Europe PMC 获取元数据（正确版）
使用 search 接口 + 正确参数获取 MeSH 和 keywords
"""

import requests
import time
from typing import Optional, Dict, List, Union
from tqdm import tqdm

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
    


# ============== 测试 ==============
if __name__ == '__main__':
    import json
    
    # 测试 PMID（包含有/无 MeSH 的情况）
    test_pmids = [
        '22163600',  # 你的测试用例
        '31537895',  # 另一个
        '28456789',  # 随机
        '35082362',  # 较新文章，可能有完整元数据
    ]
    
    print("🔍 开始测试 Europe PMC 获取...\n")
    for pmid in tqdm(test_pmids):
        ans = fetch_from_ncbi(pmid, 'your_email@example.com')
        if ans:
            print(json.dumps(ans, indent=4))