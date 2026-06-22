import os
import sys
import json
import random
import re
import requests

# 配置 sys.path
current_project_path = "E:/project/advanced-rag"
if current_project_path not in sys.path:
    sys.path.insert(0, current_project_path)

# Windows 下 utf-8 配置
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

OUTPUT_PATH = "tests/temp_data/test_sanguo_dataset.json"
LLM_API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "qwen3.6-35b-a3b-opus-nothink"

BLACKLIST_ENTITIES = [
    "刘备", "关羽", "张飞", "曹操", "孙权", "诸葛亮", "周瑜", "吕布", "赵云", "司马懿",
    "袁绍", "董卓", "鲁肃", "魏延", "黄忠", "马超", "庞统", "陆逊", "邓艾", "姜维",
    "司马昭", "司马师", "曹丕", "袁术", "刘表", "刘璋", "王允", "蒋干", "孟获", "张松",
    "夏侯惇", "夏侯渊", "张辽", "徐晃", "张郃", "太史慈", "甘宁", "吕蒙", "马岱", "典韦",
    "荆州", "益州", "徐州", "兖州", "扬州", "冀州", "幽州", "并州", "凉州", "交州",
    "洛阳", "建业", "成都", "长安", "汉中", "许昌", "新野", "樊城", "赤壁", "官渡", "白帝城", "托孤",
    "鞭打", "督邮", "大怒", "怒", "羞辱", "结义", "结盟", "盟约", "结拜"
]


def get_semantic_paragraph_chunks(txt_path, target_length=800):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"未找到文档文件: '{txt_path}'。")
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    for p in paragraphs:
        p_text = p.strip()
        if not p_text:
            continue
        current_chunk.append(p_text)
        current_length += len(p_text)
        if current_length >= target_length:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def extract_json(text: str):
    text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', text).strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
    match_braces = re.search(r'(\{[\s\S]*\})', text)
    if match_braces:
        text = match_braces.group(1).strip()
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return json.loads(text)

def generate_graph_based_qa(node_a_text, node_b_text, idx):
    prompt = f"""你是一个极其优秀的、专门用于测试 RAG 检索器图拓扑扩散能力的学术评测出题官。
以下是两段从完全脱敏的三国小说中抽取的文本片段（在物理上它们并不相邻，但逻辑上它们通过某个角色或地点相互关联）：

【片段 A】（包含提问背景和线索）：
{node_a_text}

【片段 B】（包含逻辑链条终点和答案）：
{node_b_text}

请结合这两个断裂片段中的事实，设计一个必须同时参考这两个片段才能正确回答的【跨章节多跳隐式推理问题】，并给出标准答案。

【设计准则】：
1. 答案（ground_truth）必须且只能是片段 B 中出现的某个伪装代号，形如 `[角色_X]` 或 `[地点_Y]`。绝对不能是任何历史上真实的中文人名（如“刘备”、“曹操”等）或地名！
2. 问题本身中绝对不能出现任何形如 `[角色_X]`、`[地点_Y]` 等代号，必须通过片段 A 中角色的外貌特征、动作、担任官职、所乘坐骑等隐式指代。
3. 问题中绝对严禁提及【片段 A】中任何可辨识的具体情节动作或核心名词词汇（例如“鞭打”、“大怒”、“督邮”、“结义”、“怒”等）。
4. 必须将具体的情节、动作、原因和结果，翻译为高度抽象的隐式代称（如将“结盟/结义”化为“先前确立的人物关系/盟约”；将“张飞怒鞭督邮/羞辱”化为“后续发生的突发性肢体冲突/决策”等）。
5. 必须是“关键词绝杀”全隐式出题。问题规范示例：“前一章中确立的核心人物关系，对后一章里发生的突发性肢体冲突起到了怎样的催化作用？”。
6. 问题本身也严禁使用任何著名的历史事件专有名词（如“赤壁之战”、“桃园结义”、“凤仪亭”、“连环计”、“白帝城”等）。
7. 问题的设计机制：问题提及的实体和起因在【片段 A】，但提问指向的终点人物/地点，其代号必须位于【片段 B】。只检索到片段 A 或片段 B 之一是无法得出此答案的。

请严格按照以下 JSON 格式输出，不要包含任何多余解释、Markdown 代码标记或思考过程：
{{
  "question": "隐式多跳推理问题（不得出现任何代号，也不得出现任何真实历史人名/地名/典故名称/具体情节词汇）",
  "ground_truth": "标准答案（必须是类似于 `[角色_X]` 或 `[地点_Y]` 的规范代号，不能包含其他字）"
}}"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs raw JSON content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    for attempt in range(1, 4):
        try:
            print(f"  正在请求 API (尝试 {attempt}/3)...")
            resp = requests.post(LLM_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            
            try:
                response_data = resp.json()
                content = response_data['choices'][0]['message']['content'].strip()
            except (ValueError, KeyError, IndexError) as err:
                print(f"  ⚠️ 尝试 {attempt} 接口响应数据结构解析失败: {err}")
                continue
                
            try:
                qa = extract_json(content)
                if "question" in qa and "ground_truth" in qa:
                    return {
                        "id": idx,
                        "node_a_context": node_a_text,
                        "node_b_context": node_b_text,
                        "question": qa["question"],
                        "ground_truth": qa["ground_truth"].strip()
                    }
                else:
                    print(f"  ⚠️ 尝试 {attempt} JSON 缺少必要的键: {qa}")
            except Exception as e:
                print(f"  ⚠️ 尝试 {attempt} JSON 解析或过滤失败: {e}. 原始内容摘要: {content[:200]}...")
                
        except requests.exceptions.RequestException as req_err:
            print(f"  ⚠️ 尝试 {attempt} 网络或 API 请求错误: {req_err}")
            
    return None

def main():
    disguised_path = "tests/temp_data/三国演义白话文_disguised.txt"
    if not os.path.exists(disguised_path):
        print(f"❌ 未找到换皮小说文件: {disguised_path}。请先生成它。")
        sys.exit(1)

    print("Step 1: Parsing text and extracting semantic relation chunks (Zero CPU overhead)...")
    chunks = get_semantic_paragraph_chunks(disguised_path, target_length=800)
    print(f"Generated {len(chunks)} chunks from disguised book.")
    
    # 提取每个 chunk 里的代号集合，用于判定实体共现
    code_pattern = re.compile(r'\[(?:角色|地点)_\d+\]')
    chunk_entities = []
    for idx, c in enumerate(chunks):
        entities = set(code_pattern.findall(c))
        chunk_entities.append((idx, c, entities))
        
    # Step 2: 匹配“物理断裂（索引差大于2）但实体共现（至少共享一个代号）”的对
    relation_pairs = []
    n = len(chunk_entities)
    for i in range(n):
        idx_i, text_i, ents_i = chunk_entities[i]
        if not ents_i:
            continue
        for j in range(i + 3, n): # 保证跨度大于 2 个物理块
            idx_j, text_j, ents_j = chunk_entities[j]
            # 找交集实体
            common = ents_i & ents_j
            if common:
                # 我们找到了一个物理断裂但逻辑共现的对
                relation_pairs.append((text_i, text_j, common))
                
    print(f"Found {len(relation_pairs)} non-adjacent entity-co-occurrence chunk pairs.")
    if not relation_pairs:
        print("⚠️ 没找到物理断裂实体共现对，退而求其次使用所有不相邻块对。")
        for i in range(n):
            for j in range(i + 3, n):
                relation_pairs.append((chunks[i], chunks[j], set()))

    # 随机打乱以增加采样随机度
    random.seed(42)
    random.shuffle(relation_pairs)
    
    dataset = []
    idx = 1
    
    # 遍历这些连边对应的节点对，让大模型跨这两个断裂文本块出题
    for node_a_text, node_b_text, common_entities in relation_pairs:
        if len(dataset) >= 10:
            break
            
        print(f"🤖 Generating Graph-based Q&A [{len(dataset)+1}/10] on entity common {common_entities}...")
        qa_pair = generate_graph_based_qa(node_a_text, node_b_text, idx)
        if qa_pair:
            q_text = qa_pair.get("question", "")
            gt_text = qa_pair.get("ground_truth", "")
            
            # 严格过滤
            if "[角色_" in q_text or "角色_" in q_text or "[地点_" in q_text or "地点_" in q_text:
                print("  ⚠️ 提问本身泄露了代号，废弃。")
                continue
                
            has_blacklist = False
            for black_word in BLACKLIST_ENTITIES:
                if black_word in q_text or black_word in gt_text:
                    print(f"  ⚠️ 问题或答案中包含黑名单词汇 '{black_word}'，废弃。")
                    has_blacklist = True
                    break
            if has_blacklist:
                continue
                
            if not re.match(r'^\[(角色|地点)_\d+\]$', gt_text):
                print(f"  ⚠️ 答案格式不合规 (必须仅为 [角色_X] 或 [地点_Y]): '{gt_text}'，废弃。")
                continue
                
            dataset.append(qa_pair)
            print(f"  ✅ Generated successfully! Q: {q_text[:40]}... A: {gt_text}")
            idx += 1
            
    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 Graph-based Q&A Generation finished! Saved 10 pairs to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
