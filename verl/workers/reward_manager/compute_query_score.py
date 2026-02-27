import re
from typing import Dict, Union,Tuple
import random
import string
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score
def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
        
def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

import re
import json

import re
import json

def extract_all_query_lists(test_str):
    """
    从字符串中提取**所有** <tool_call> 标签内的 query_list 列表
    
    参数:
        test_str (str): 包含 tool_call 标签的原始字符串
    
    返回:
        list: 提取到的 query_list 列表的集合（每个元素对应一个 tool_call 标签的结果）；无有效内容返回空列表
    """
    # 匹配 <tool_call> 内的 JSON 内容（匹配所有标签，使用 findall）
    pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
    matches = re.findall(pattern, test_str, re.DOTALL)  # re.DOTALL 匹配换行符，findall 获取所有结果
    
    all_query_lists = []
    if not matches:
        print("提示（extract_all_query_lists）：未找到 <tool_call> 标签或标签内无有效内容")
        return all_query_lists
    
    for idx, tool_call_json in enumerate(matches, 1):
        try:
            # 解析 JSON 并提取 query_list
            tool_call_data = json.loads(tool_call_json)
            query_list = tool_call_data.get("arguments", {}).get("query_list", [])
            all_query_lists.append(query_list)
        except json.JSONDecodeError as e:
            print(f"错误（extract_all_query_lists）：第 {idx} 个 tool_call JSON 解析失败 - {e}")
            all_query_lists.append([])  # 解析失败填充空列表，保证列表长度匹配
        except Exception as e:
            print(f"错误（extract_all_query_lists）：第 {idx} 个 tool_call 提取失败 - {e}")
            all_query_lists.append([])  # 其他错误填充空列表，保证列表长度匹配
    
    return all_query_lists

def extract_all_tool_response_docs(test_str):
    """
    从字符串中提取**所有** <tool_response> 的 result 内容，按 Doc 1/2/3... 拆分为列表
    保留每个 Doc 的完整文本（含 Doc X (Title: ...) 标识），返回多组结果的集合
    
    参数:
        test_str (str): 包含 tool_response 标签的原始字符串
    
    返回:
        list: 每个元素对应一个 tool_response 标签的 Doc 列表；无有效内容返回空列表
    """
    # 步骤1：提取**所有** tool_response 中的原始内容
    tool_response_pattern = r'<tool_response>\s*(.*?)\s*</tool_response>'
    tr_matches = re.findall(tool_response_pattern, test_str, re.DOTALL)
    
    all_doc_lists = []
    if not tr_matches:
        print("提示（extract_all_tool_response_docs）：未找到 <tool_response> 标签或标签内无有效内容")
        return all_doc_lists
    
    for tr_idx, raw_tr_content in enumerate(tr_matches, 1):
        # 步骤2：提取 result 字段的值
        result_pattern = r'"result"\s*:\s*"(.*?)"(?=})'
        result_match = re.search(result_pattern, raw_tr_content, re.DOTALL)
        if not result_match:
            # 兼容单引号情况
            result_pattern_single = r"'result'\s*:\s*'(.*?)'(?=})"
            result_match = re.search(result_pattern_single, raw_tr_content, re.DOTALL)
            if not result_match:
                print(f"错误（extract_all_tool_response_docs）：第 {tr_idx} 个 tool_response 未找到 result 字段或格式不匹配")
                all_doc_lists.append([])
                continue
        
        # 还原转义字符并清理原始 result 内容
        result_content = (
            result_match.group(1)
            .strip()
            .replace("\\n", "\n")  # 还原换行符
            .replace('\\"', '"')   # 还原转义双引号
        )
        
        # 步骤3：匹配完整的 Doc 片段（保留 Doc X 标识）
        doc_pattern = r'(Doc \d+ \(Title: .*?\)\n.*?)(?=Doc \d+ \(Title: |$)'
        doc_matches = re.findall(doc_pattern, result_content, re.DOTALL)
        
        # 步骤4：清理每个 Doc 内容的首尾空白，生成当前 tool_response 的 Doc 列表
        doc_list = [doc.strip() for doc in doc_matches]
        all_doc_lists.append(doc_list)
    
    return all_doc_lists




def extract_paired_tool_data(test_str):
    """
    整合函数：提取文本中所有成对的 <tool_call> 和 <tool_response>，返回成对列表
    
    参数:
        test_str (str): 包含多对 tool_call/tool_response 标签的原始字符串
    
    返回:
        list: 每个元素为 (query_list, doc_list)，一一对应成对的 tool 调用和响应；多余单侧标签忽略
    """
    # 提取所有的 query_list 集合和 doc_list 集合
    all_query_lists = extract_all_query_lists(test_str)
    all_doc_lists = extract_all_tool_response_docs(test_str)
    
    # 按出现顺序一一配对，取两者中较短的长度，忽略多余的单侧标签
    paired_list = []
    pair_count = min(len(all_query_lists), len(all_doc_lists))
    
    for i in range(pair_count):
        paired_element = (all_query_lists[i], all_doc_lists[i])
        paired_list.append(paired_element)
    
    return paired_list


import numpy as np
from scipy.spatial.distance import cosine
import requests

# 配置vLLM Server地址（根据你的实际部署修改，默认通常是这个）
VLLM_EMBEDDING_API_URL = "http://localhost:8081/v1/embeddings"

def get_qwen3_embeddings(texts: list[str]) -> list[np.ndarray]:
    """
    通过vLLM Server获取Qwen3 Embedding向量
    :param texts: 待编码的字符串列表
    :return: 每个文本对应的Embedding向量列表（np.ndarray格式）
    """
    if not texts:
        return []
    
    # 构造vLLM Embedding API请求体（符合vLLM API规范）
    request_payload = {
        "input": texts,
        "model": "/nfsdata3/yiao/yiao/model/Qwen3-Embedding-0.6B"  # 替换为你的Qwen3 Embedding模型名称，确保vLLM已加载
    }
    
    try:
        # 发送POST请求获取Embedding
        response = requests.post(VLLM_EMBEDDING_API_URL, json=request_payload)
        response.raise_for_status()  # 捕获HTTP请求错误
        result = response.json()
        
        # 提取Embedding向量并转为np.ndarray（方便后续计算相似度）
        embeddings = [np.array(item["embedding"], dtype=np.float32) for item in result["data"]]
        return embeddings
    
    except Exception as e:
        print(f"获取Qwen3 Embedding失败：{str(e)}")
        return []
    
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个Embedding向量的余弦相似度（取值范围[0,1]，值越大相关性越高）
    :param vec1: 第一个向量
    :param vec2: 第二个向量
    :return: 余弦相似度值
    """
    # 避免零向量导致计算错误
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    # 余弦相似度 = 1 - 余弦距离（scipy的cosine返回的是距离）
    return 1 - cosine(vec1, vec2)
def compute_correlation(contexts, response_str, similarity_threshold: float = 0.8):
    recall_contexts = extract_paired_tool_data(response_str)

    articles = []
    for i in range(len(recall_contexts)):
        for j in recall_contexts[i][1]:
            articles.append(j)

    # 步骤2：获取两个列表的Qwen3 Embedding
    contexts_embeddings = get_qwen3_embeddings(contexts)
    recall_embeddings = get_qwen3_embeddings(articles)

    # 边界处理：空列表直接返回结果
    if not contexts:
        return {
            "coverage_rate": 0,  # 原始列表为空，默认覆盖度100%
            "avg_similarity": 0.0,
            "recall_contexts": articles,
            "contexts": contexts,
            "match_details": []
        }
    if not articles:
        return {
            "coverage_rate": 0.0,  # 无召回结果，覆盖度0%
            "avg_similarity": 0.0,
            "recall_contexts": articles,
            "contexts": contexts,
            "match_details": []
        }
    
    
    # 边界处理：Embedding获取失败
    if len(contexts_embeddings) != len(contexts) or len(recall_embeddings) != len(articles):
        print("警告：Embedding获取不完整，无法准确计算相关性")
        return {
            "coverage_rate": 0.0,
            "avg_similarity": 0.0,
            "recall_contexts": articles,
            "contexts": contexts,
            "match_details": []
        }
    
    # 步骤3：计算覆盖度和平均相似度
    match_details = []  # 存储每个context的匹配详情
    covered_count = 0  # 被覆盖的context数量
    total_max_similarity = 0.0  # 所有context的最大相似度总和
    
    for idx, (context, ctx_emb) in enumerate(zip(contexts, contexts_embeddings)):
        # 计算当前context与所有recall_context的相似度，取最大值
        max_sim = 0.0
        best_match_recall = None
        
        for recall_idx, (recall_ctx, recall_emb) in enumerate(zip(articles, recall_embeddings)):
            sim = cosine_similarity(ctx_emb, recall_emb)
            if sim > max_sim:
                max_sim = sim
                best_match_recall = recall_ctx
        
        # 判定是否被覆盖
        is_covered = max_sim >= similarity_threshold
        if is_covered:
            covered_count += 1
        total_max_similarity += max_sim
        
        # 记录匹配详情
        match_details.append({
            "context": context,
            "best_match_recall": best_match_recall,
            "max_similarity": round(max_sim, 4),
            "is_covered": is_covered
        })
    
    # 计算最终指标
    coverage_rate = covered_count / len(contexts)  # 覆盖率（核心指标，体现覆盖效果）
    avg_similarity = total_max_similarity / len(contexts)  # 平均相似度（体现匹配质量）
    
    # 打印调试信息（保留你的原有打印逻辑）
    print("=== 召回列表 ===")
    for ctx in articles:
        print(ctx)
    print("\n=== 原始上下文列表 ===")
    for ctx in contexts:
        print(ctx)
    print(f"\n=== 评估结果 ===")
    print(f"覆盖率：{coverage_rate:.2%}（{covered_count}/{len(contexts)} 个原始上下文被覆盖）")
    print(f"平均相似度：{avg_similarity:.4f}")
    
    # 返回完整结果（方便后续进一步处理）
    return {
        "coverage_rate": coverage_rate,
        "avg_similarity": avg_similarity,
        "recall_contexts": articles,
        "contexts": contexts,
        "match_details": match_details,
        "similarity_threshold": similarity_threshold
    }


# <think>
# Okay, let's see. The user is asking, "who sings 'i just died in your arms'?" So first, I need to figure out who the song is. Wait, the song title there is "i just died in your arms." That sounds familiar, maybe a pop song. Maybe from a film or a song that's popular. I should check if there's a song that has that title.

# But wait, the user might have it in mind. But how do I confirm? Since the user included the song title as is, I can try to find out the artist. The process involves using the search tool here. Let me call the search function with "i just died in your arms" as the query. Then, the tool will return the result, which should include the artist. Once I have that, I can present the answer.
# </think>

# <tool_call>
# {"name": "search", "arguments": {"query_list": ["i just died in your arms"]}}
# </tool_call>user
# <tool_response>
# {"result": "Doc 1 (Title: \"(I Just) Died in Your Arms\")\n(I Just) Died in Your Arms \"\"(I Just) Died in Your Arms\"\" is a song by the English rock band Cutting Crew. The song was released as the lead single from their debut studio album, \"\"Broadcast\"\" (1986). It was first released on 25 July 1986 in the United Kingdom, and then released to the United States on 1 January 1987. The song was written by frontman Nick Van Eede, produced by Terry Brown, John Jansen and the band, and mixed at Utopia Studios in London by Tim Palmer. The power ballad was the band's biggest hit, peaking at number-one in\n\nDoc 2 (Title: \"(I Just) Died in Your Arms\")\nthe United States, Canada and Finland, and reaching the top five in the UK, South Africa, Sweden and Switzerland. The actual words \"\"I just died in your arms tonight\"\" originally came allegedly to Van Eede while he was having sex with his girlfriend, the French phrase \"\"la petite mort\"\", or \"\"the little death\"\", being a metaphor for orgasm. After writing down his version of the phrase, he later used it as the opening line to the song as well as using it as the chorus. There were two music videos produced. The North American version featured artistic fragmented shots using\n\nDoc 3 (Title: \"Nick Van Eede\")\nVan Eede married Nikki McFarlane in June 1996. He has a daughter, Lauren (born 13 May 1986). Nick Van Eede Nick Van Eede (born Nicholas Eede, 14 June 1958) is an English musician, producer and songwriter. He is best known for singing and writing the 1986 U.S. #1 power ballad, \"\"(I Just) Died in Your Arms\"\" for his band Cutting Crew. While working as a hospital orderly in the late 1970s, Van Eede was discovered playing by ex member of The Animals, Chas Chandler, who sent him on a tour of Poland as support for Slade. Van Eede was only"}
# </tool_response>
# assistant
# <think>
# Okay, let's break this down. The user asked, "who sings 'i just died in your arms'?" The search results mention a song by Cutting Crew. The songs are from 1986, so the current answer is likely to be the band. The final answer should state the artist. The response included the artist's name and the band's details. So, wrapping it up, the answer is the band Cutting Crew.
# </think>

# <answer>Cutting Crew</answer>

def is_valid_sequence(response_str: str) -> Tuple[bool, str]:
    """
    检查模型响应的标签序列是否合法（支持多轮工具调用）：
    1. 核心标签必须成对闭合
    2. 标签序列需符合逻辑：think → [tool_call → tool_response → think]+ → answer
       （[]+ 表示可重复任意次，支持多轮工具调用）
    3. 自动过滤user/assistant等交互标识，不影响格式检查
    4. 允许标签间存在空白/交互文本，仅校验核心标签序列
    
    参数：
        response_str: 模型的完整响应文本（含所有标签和内容）
    
    返回：
        (是否合法, 提示信息)
    """
    # ========== 配置项（你可手动修改这里的标签规则） ==========
    # 1. 核心标签配置（键：标签类型，值：(开始标记, 结束标记)）
    TAG_CONFIG = {
        "think": ("<think>", "</think>"),       # 推理块标签（可替换为你的标记）
        "tool_call": ("<tool_call>", "</tool_call>"),  # 工具调用标签
        "tool_response": ("<tool_response>", "</tool_response>"),  # 工具响应标签
        "answer": ("<answer>", "</answer>")     # 最终答案标签
    }
    # 2. 需要过滤的干扰文本（如交互标识）
    FILTER_KEYWORDS = ["user", "assistant"]
    
    # ========== 步骤1：预处理文本 ==========
    # 1. 过滤干扰关键词（替换为空）
    filtered_text = response_str
    for keyword in FILTER_KEYWORDS:
        filtered_text = re.sub(
            r"\b" + re.escape(keyword) + r"\b", 
            "", 
            filtered_text, 
            flags=re.IGNORECASE
        )
    # 2. 统一空白字符（换行/制表符替换为空格）
    filtered_text = re.sub(r"\s+", " ", filtered_text).strip()
    
    # ========== 步骤2：检查标签闭合性 ==========
    for tag_type, (start_tag, end_tag) in TAG_CONFIG.items():
        # 转义特殊字符，避免正则匹配出错
        start_tag_esc = re.escape(start_tag)
        end_tag_esc = re.escape(end_tag)
        
        # 统计开始/结束标签数量
        start_count = len(re.findall(start_tag_esc, filtered_text))
        end_count = len(re.findall(end_tag_esc, filtered_text))
        
        if start_count != end_count:
            return (
                False, 
                f"标签闭合错误：{tag_type}标签开始数({start_count})≠结束数({end_count})"
            )
    
    # ========== 步骤3：提取核心标签序列 ==========
    # 生成标签匹配正则（匹配所有核心标签）
    tag_pattern_parts = []
    for tag_type, (start_tag, end_tag) in TAG_CONFIG.items():
        tag_pattern_parts.append(re.escape(start_tag))
        tag_pattern_parts.append(re.escape(end_tag))
    tag_pattern = re.compile("|".join(tag_pattern_parts))
    
    # 提取所有核心标签并按顺序排列
    tag_sequence = []
    for match in tag_pattern.finditer(filtered_text):
        tag = match.group()
        # 映射标签到类型（如"<tool_call>" → "tool_call_start"）
        for tag_type, (start_tag, end_tag) in TAG_CONFIG.items():
            if tag == start_tag:
                tag_sequence.append(f"{tag_type}_start")
                break
            elif tag == end_tag:
                tag_sequence.append(f"{tag_type}_end")
                break
    
    # 空序列直接判定不合法
    if not tag_sequence:
        return False, "未检测到任何核心标签"
    
    # ========== 步骤4：校验标签序列逻辑（支持多轮工具调用） ==========
    # 状态机：start → think_start → think_end → (tool_call_start → tool_call_end → tool_response_start → tool_response_end → think_start → think_end)* → answer_start → answer_end → end
    current_state = "start"
    # 定义合法的状态跳转：当前状态 → 允许的下一个标签
    valid_transitions = {
        "start": ["think_start"],
        "think_start": ["think_end"],
        "think_end": ["tool_call_start", "answer_start"],
        "tool_call_start": ["tool_call_end"],
        "tool_call_end": ["tool_response_start"],
        "tool_response_start": ["tool_response_end"],
        "tool_response_end": ["think_start"],
        "answer_start": ["answer_end"],
        "answer_end": ["end"]  # 可选：如果需要严格到end，可保留；否则可忽略
    }
    
    # 遍历标签序列，逐次校验跳转合法性
    for tag in tag_sequence:
        # 检查当前状态是否允许跳转到当前标签
        if tag not in valid_transitions.get(current_state, []):
            return (
                False, 
                f"序列错误：状态{current_state}下不允许出现{tag}标签"
            )
        # 更新当前状态为当前标签（核心：直接用标签作为状态）
        current_state = tag
    
    # 检查最终状态（必须以answer_end结尾）
    if current_state != "answer_end":
        return False, f"序列未正确结束：最终状态为{current_state}，需以answer_end结尾"
    
    # ========== 所有检查通过 ==========
    return True, "格式校验通过：标签闭合完整，序列逻辑合法"

def extract_all_titles(doc_list):
    all_titles = []
    for doc in doc_list:
        # 匹配到右括号 ) 或换行符 \n 为止
        match = re.search(r'Doc \d+ \(Title: ([^)\n]+)', doc)
        if match:
            all_titles.append(match.group(1).strip())
    return all_titles
def get_all_titles(response_str: str):
    ret = []
    recall_contexts = extract_paired_tool_data(response_str)
    # import pdb; pdb.set_trace()
    for i, recall_context in enumerate(recall_contexts):
        titles = extract_all_titles(recall_context[1])
        ret = ret + titles
    return ret

def get_search_score(response_str: str, ground_truth_title: str):
    titles = get_all_titles(response_str)
    # if len(titles) != 0:
    #     print("====================search======================")
    #     print(f"titles: {titles}")
    #     print(f"ground_truth_title: {ground_truth_title}")

    for title in titles:
        if ground_truth_title in title or title in ground_truth_title:
            return 1.0
    if ground_truth_title in titles:
        return 1.0
    else:
        return 0.0

def main():
    test_str = """
    <think>
    Okay, let's see. The user is asking, "who sings 'i just died in your arms'?" So first, I need to figure out who the song is. Wait, the song title there is "i just died in your arms." That sounds familiar, maybe a pop song. Maybe from a film or a song that's popular. I should check if there's a song that has that title.

    But wait, the user might have it in mind. But how do I confirm? Since the user included the song title as is, I can try to find out the artist. The process involves using the search tool here. Let me call the search function with "i just died in your arms" as the query. Then, the tool will return the result, which should include the artist. Once I have that, I can present the answer.
    </think>

    <tool_call>
    {"name": "search", "arguments": {"query_list": ["i just died in your arms"]}}
    </tool_call>user
    <tool_response>
    {"result": "Doc 1 (Title: \"(I Just) Died in Your Arms\")\n(I Just) Died in Your Arms \"\"(I Just) Died in Your Arms\"\" is a song by the English rock band Cutting Crew. The song was released as the lead single from their debut studio album, \"\"Broadcast\"\" (1986). It was first released on 25 July 1986 in the United Kingdom, and then released to the United States on 1 January 1987. The song was written by frontman Nick Van Eede, produced by Terry Brown, John Jansen and the band, and mixed at Utopia Studios in London by Tim Palmer. The power ballad was the band's biggest hit, peaking at number-one in\n\nDoc 2 (Title: \"(I Just) Died in Your Arms\")\nthe United States, Canada and Finland, and reaching the top five in the UK, South Africa, Sweden and Switzerland. The actual words \"\"I just died in your arms tonight\"\" originally came allegedly to Van Eede while he was having sex with his girlfriend, the French phrase \"\"la petite mort\"\", or \"\"the little death\"\", being a metaphor for orgasm. After writing down his version of the phrase, he later used it as the opening line to the song as well as using it as the chorus. There were two music videos produced. The North American version featured artistic fragmented shots using\n\nDoc 3 (Title: \"Nick Van Eede\")\nVan Eede married Nikki McFarlane in June 1996. He has a daughter, Lauren (born 13 May 1986). Nick Van Eede Nick Van Eede (born Nicholas Eede, 14 June 1958) is an English musician, producer and songwriter. He is best known for singing and writing the 1986 U.S. #1 power ballad, \"\"(I Just) Died in Your Arms\"\" for his band Cutting Crew. While working as a hospital orderly in the late 1970s, Van Eede was discovered playing by ex member of The Animals, Chas Chandler, who sent him on a tour of Poland as support for Slade. Van Eede was only"}
    </tool_response>
    assistant
    <think>
    Okay, let's break this down. The user asked, "who sings 'i just died in your arms'?" The search results mention a song by Cutting Crew. The songs are from 1986, so the current answer is likely to be the band. The final answer should state the artist. The response included the artist's name and the band's details. So, wrapping it up, the answer is the band Cutting Crew.
    </think>
    <tool_call>
    {"name": "search", "arguments": {"query_list": ["i just died in your arms"]}}
    </tool_call>user
    <tool_response>
    {"result": "Doc 1 (Title: \"(I Just) Died in Your Arms\")\n(I Just) Died in Your Arms \"\"(I Just) Died in Your Arms\"\" is a"}
    </tool_response>
    assistant
    <think>
    The user asked, "who sings 'i just died in your arms'?" The search results mention a song by Cutting Crew. The songs are from 1986, so the current answer is likely to be the band. The final answer should state
    </think>
    <answer>Cutting Crew</answer>
    """
    print(is_valid_sequence(test_str))
    print(compute_score_em(test_str, {"target": "Cutting crew"}))

    contexts = ['We sought to develop a more reliable structured implicit chart review instrument for use in assessing the quality of care for chronic disease and to examine if ratings are more reliable for conditions in which the evidence base for practice is more developed.', 'We conducted a reliability study in a cohort with patient records including both outpatient and inpatient care as the objects of measurement. We developed a structured implicit review instrument to assess the quality of care over one year of treatment. 12 reviewers conducted a total of 496 reviews of 70 patient records selected from 26 VA clinical sites in two regions of the country. Each patient had between one and four conditions specified as having a highly developed evidence base (diabetes and hypertension) or a less developed evidence base (chronic obstructive pulmonary disease or a collection of acute conditions). Multilevel analysis that accounts for the nested and cross-classified structure of the data was used to estimate the signal and noise components of the measurement of quality and the reliability of implicit review.', 'For COPD and a collection of acute conditions the reliability of a single physician review was quite low (intra-class correlation = 0.16-0.26) but comparable to most previously published estimates for the use of this method in inpatient settings. However, for diabetes and hypertension the reliability is significantly higher at 0.46. The higher reliability is a result of the reviewers collectively being able to distinguish more differences in the quality of care between patients (p<0.007) and not due to less random noise or individual reviewer bias in the measurement. For these conditions the level of true quality (i.e. the rating of quality of care that would result from the full population of physician reviewers reviewing a record) varied from poor to good across patients.']
    # print(compute_correlation(contexts, test_str))
    recall_contexts = extract_paired_tool_data(test_str)
    print(recall_contexts[0][1])
    print(get_all_titles(test_str))

# 回答包含 答案
import re
import string

def normalize_text(text):
    """文本标准化：小写、去标点、合并空格"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text.strip()

def is_answer_correct(model_answer, ground_truth_list, method='contains'):
    """
    判断模型回答是否与ground_truth匹配
    
    参数:
        model_answer: str, 模型生成的回答
        ground_truth_list: list[str], 正确答案的多种表述
        method: str, 匹配策略
            - 'exact': 完全精确匹配
            - 'contains': 包含匹配（推荐）
            - 'keyword': 提取核心关键词匹配
    
    返回:
        bool: 是否匹配成功
    """
    if not model_answer or not ground_truth_list:
        return False
    
    normalized_answer = normalize_text(model_answer)
    
    for gt in ground_truth_list:
        normalized_gt = normalize_text(gt)
        
        if method == 'exact':
            if normalized_answer == normalized_gt:
                return True
        elif method == 'contains':
            # 双向包含：答案包含GT 或 GT包含答案
            if normalized_gt in normalized_answer or normalized_answer in normalized_gt:
                return True
        elif method == 'keyword':
            # 提取GT的核心词（如"TNM"）进行匹配
            keywords = [w for w in normalized_gt.split() if len(w) > 2 or w.upper() == w]
            if keywords and all(kw in normalized_answer for kw in keywords):
                return True
    
    return False

def test_is_answer_correct():
    model_answer = "The answer is 42."
    ground_truth_list = ['TNM staging', 'TNM classification', 
                         'TNM system', 'TNM staging system', 
                         'TNM', 'Tumor Node Metastasis staging', 
                         'Tumor-Node-Metastasis staging',
                           'Tumor, Node, Metastasis staging', 
                           'Tumor Node Metastasis classification', 
                           'Tumor-Node-Metastasis classification', 
                           'Tumor, Node, Metastasis classification', 
                           'Tumor Node Metastasis system']
    solution_str = (
        "<think>\n"
        "Okay, let's see. The question is asking which staging system is used to classify "
        "gastric cancer based on tumor size, lymph node involvement, and metastasis. "
        "I need to recall the standard staging systems for cancers.\n\n"
        "First, I remember that cancer staging is typically done using the TNM system. "
        "TNM stands for Tumor, Node, and Metastasis. "
        "This system is widely used by the American Joint Committee on Cancer (AJCC) and the International "
        "Union Against Cancer (UICC). \n\n"
        "For gastric cancer specifically, the AJCC has developed a staging "
        "system that builds on the TNM framework. "
        "The T category refers to the size and depth of the tumor, N refers to regional lymph node involvement, "
        "and M refers to distant metastasis. So, the AJCC stage is based on these three factors.\n\n"
        "I should verify if there's another system used, but I think the AJCC is the primary one. "
        "The UICC also uses a similar system, but the AJCC is more commonly referenced in medical literature. "
        "Additionally, the TNM system is the standard for many cancers, including gastric. "
        "So the answer should be the AJCC TNM staging system.\n</think>\n\n"
        "<answer> "
        "The staging system used to classify the extent of gastric cancer based on tumor size, "
        "lymph node involvement, and metastasis is the American Joint Committee on Cancer (AJCC) TNM staging system. "
        "This system categorizes cancer by T (tumor size/depth), N (lymph node involvement), "
        "and M (metastasis) to determine the stage.</answer>"
    )
    answer = extract_solution(solution_str=solution_str)
    # import pdb; pdb.set_trace()
    print(is_answer_correct(answer, ground_truth_list))

def test_get_all_titles():
    response_str = """<think>
Okay, so the user is asking about a nuclear medicine imaging modality used to evaluate lung perfusion abnormalities in patients with Behçet's disease. Let me start by recalling what I know about Behçet's disease. It's a systemic inflammatory disorder that can affect various parts of the body, including the lungs. Lung involvement can lead to issues like vasculitis or thrombosis, which might cause perfusion abnormalities.

Nuclear medicine imaging techniques used for perfusion studies typically include methods like perfusion CT, perfusion MRI, or maybe even radionuclide imaging. However, the question specifically mentions nuclear medicine, so I should focus on that. Common nuclear medicine procedures include single-photon emission computed tomography (SPECT) and positron emission tomography (PET). But which one is used for lung perfusion?

Wait, in the context of lung perfusion, a common test is a perfusion scan, which can be done with technetium-99m. For example, a perfusion scan using technetium-99m labeled red blood cells (Tc-99m RBC) is used to assess blood flow in the lungs. This is often done in patients with suspected pulmonary embolism, but could it be relevant here?

But Behçet's disease can cause vasculitis, which might lead to lung ischemia. So maybe a perfusion scan is used to check for reduced blood flow. However, I'm not entirely sure if this is the standard approach for Behçet's. I should verify if there's a specific imaging modality commonly recommended for this scenario.

Alternatively, maybe the answer is a lung perfusion scan using SPECT. But I need to confirm if this is used in the context of Behçet's disease. Since I might not have the exact information, I should use the search tool to find the answer. Let me check the available functions. The search function allows me to query for information. So I should call the search function with the query "nuclear medicine imaging modality for lung perfusion in Behçet's disease" to get the most relevant information.
</think>

<tool_call>
{"name": "search", "arguments": {"query_list": ["nuclear medicine imaging modality for lung perfusion in Behçet's disease"]}}
</tool_call>

user

<tool_response>
{"result": "Doc 1 (Title: 99Tcm-MAA lung scintigraphy in patients with Behçet's disease: its value and correlation with clinical course and other diagnostic modalities.)
Behçet's disease, initially described as local vasculitis involving the oral, genital mucosa and eyes, can affect many organ systems with a varying clinical course. Pulmonary lesions associated with Behçet's disease are rare and reported to have a prevalence of 1.0-7.7%. In this prospective study, 29 patients aged 16-60 years (mean +/- s: 34 +/- 10.5) were investigated for pulmonary involvement with direct radiography, computerized tomography (CT) and intravenous digital subtraction angiography. Lung scintigraphy was also performed to determine its diagnostic value. In addition, patients had a fibreoptic bronchoscopic examination, spirometric test, carbon monoxide diffusion test and methacholine test. Of the 29 patients with Behçet's disease, 17 had pulmonary involvement. The diagnostic criteria for pulmonary vasculitis and tracheobronchial tree involvement were: (1) hilar vascular prominence or mass, new alveolar and reticulonodular infiltrates on chest X-ray; (2) aphthous lesions, bleeding ulcerations, inflamed mucosal appearance on bronchoscopy; (3) vascular thrombosis or aneurysms on CT; (4) occlusion or aneurysms of the main pulmonary arteries or its branches. The value of lung scintigraphy and its correlation with other imaging modalities were investigated to determine if a lung perfusion scan alone can be used to detect pulmonary involvement in Behçet's disease. Twenty-six patients were investigated with lung scintigraphy. Thirteen of 14 patients with lung involvement were identified. Perfusion lung scintigraphy provided true diagnostic information in 96% of our patients. In eight patients without pulmonary involvement, the lung scans were normal. Although DSA specifically demonstrates pulmonary aneurysms, it involves administration of contrast media and cannot show abnormalities distal to the third pulmonary arterial branches. Although vasculitis caused by collagen disorders is physiologically indistinguishable from pulmonary emboli, most perfusion defects resolve within 3 months, which can be used as a differential diagnostic criterion in the appropriate clinical setting. Therefore, pulmonary perfusion imaging can be used to screen patients suspected of having lung involvement, especially in the absence of thrombophlebitis. In patients with thrombophlebitis, a repeat examination at 3 months is recommended to differentiate pulmonary involvement from pulmonary embolism.

Doc 2 (Title: Ventilation and perfusion magnetic resonance imaging of the lung.)
A close interaction between the respiratory pump, pulmonary parenchyma and blood circulation is essential for a normal lung function. Many pulmonary diseases present, especially in their initial phase, a variable regional impairment of ventilation and perfusion. In the last decades various techniques have been established to measure the lung function. Besides the global pulmonary function tests (PFTs) imaging techniques gained increasing importance to detect local variations in lung function, especially for ventilation and perfusion assessment. Imaging modalities allow for a deeper regional insight into pathophysiological processes and enable improved planning of invasive procedures. In contrast to computed tomography (CT) and the nuclear medicine techniques, magnetic resonance imaging (MRI), as a radiation free imaging modality gained increasing importance since the early 1990 for the assessment of pulmonary function. The major inherent problems of lung tissue, namely the low proton density and the pulmonary and cardiac motion, were overcome in the last years by a constant progress in MR technology. Some MR techniques are still under development, a process which is driven by scientific questions regarding the physiology and pathophysiology of pulmonary diseases, as well as by the need for fast and robust clinically applicable imaging techniques as safe therapy monitoring tools. MRI can be considered a promising ionizing-free alternative to techniques like CT or nuclear medicine techniques for the evaluation of lung function. The goal of this article is to provide an overview on selected MRI techniques for the assessment of pulmonary ventilation and perfusion.

Doc 3 (Title: Comparison of dynamic flat-panel detector-based chest radiography with nuclear medicine ventilation-perfusion imaging for the evaluation of pulmonary function: A clinical validation study.)
Dynamic chest radiography (DCR) is a flat-panel detector (FPD)-based functional lung imaging technique capable of measuring temporal changes in radiographic lung density due to ventilation and perfusion. The aim of this study was to determine the diagnostic performance of DCR in the evaluation of pulmonary function based on changes in radiographic lung density compared to nuclear medicine lung scans. This study included 53 patients with pulmonary disease who underwent DCR and nuclear medicine imaging at our institution. Dynamic chest radiography was conducted using a dynamic FPD system to obtain sequential chest radiographs during one breathing cycle. The maximum change in the average pixel value (Δ<sub>max</sub>) was measured, and the percentage of Δ<sub>max</sub> in each lung region, calculated relative to the sum of all lung regions (Δ<sub>max</sub>%), was calculated for each factor (ventilation and perfusion). The Δ<sub>max</sub>% was compared with the accumulation of radioactive agents (radioactive agents%) on ventilation and perfusion scans in each lung and lung region using correlation coefficients and scatter plots. The ratio of ventilation to perfusion Δ<sub>max</sub>% was calculated and compared with nuclear medicine ventilation-perfusion (V/Q) findings in terms of sensitivity and specificity for V/Q mismatch in each lung region. There was a high correlation between Δ<sub>max</sub>% and radioactive agents% for each lung (Ventilation: r = 0.81, perfusion: r = 0.87). However, correlation coefficients were lower (0.37 to 0.80) when comparing individual lung regions, with the upper lung regions showing the lowest correlation coefficients. The sensitivity and specificity of DCR for V/Q mismatch were 63.3% (19/30) and 60.1% (173/288), respectively. Motion artifacts occasionally increased Δ<sub>max</sub>%, resulting in false negatives. Ventilation and perfusion Δ<sub>max</sub>% correlated reasonably with radioactive agents% on ventilation and perfusion scans. Although the regional correlations were lower and the detection performance for V/Q mismatch was not enough for clinical use at the moment, these results suggest the potential for DCR to be used as a functional imaging modality that can be performed without the use of radioactive contrast agents. Further technical improvement is required for the implementation of DCR-based V/Q studies."}
</tool_response>

assistant

<think>
Okay, let me process this. The user is asking about a nuclear medicine imaging modality used to evaluate lung perfusion in Behçet's disease. 

From the search results, Doc 1 mentions that lung scintigraphy using 99Tcm-MAA (technetium-99m macroaggregated albumin) was used in patients with Behçet's disease. The study found that perfusion lung scintigraphy identified 13 out of 14 patients with lung involvement, with a high diagnostic accuracy. This suggests that lung scintigraphy is a method used here.

Doc 2 talks about ventilation and perfusion MRI, but the question specifies nuclear medicine, so MRI might not be the answer. Doc 3 discusses dynamic chest radiography (DCR) compared to nuclear medicine scans, but again, DCR isn't nuclear medicine.

So the key point is that lung scintigraphy, specifically with 99Tcm-MAA, is the nuclear medicine modality mentioned in the context of Behçet's disease for assessing lung perfusion. The other documents don't provide a direct answer related to nuclear medicine. Therefore, the answer should be lung scintigraphy using 99Tcm-MAA.
</think>

<answer> The nuclear medicine imaging modality used to evaluate lung perfusion abnormalities in patients with Behçet's disease is lung scintigraphy with 99Tcm-MAA (technetium-99m macroaggregated albumin). This technique was found to have high diagnostic accuracy in detecting pulmonary involvement in Behçet's disease, as demonstrated in a study where it identified 13 out of 14 patients with lung involvement. </answer>
"""
    titles = get_all_titles(response_str)
    print(titles)

if __name__ == "__main__":
    test_get_all_titles()