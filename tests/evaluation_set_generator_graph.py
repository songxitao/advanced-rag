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
    
    # 优先匹配并提取 JSON 数组 [ ... ] 结构
    match_bracket = re.search(r'(\[[\s\S]*\])', text)
    if match_bracket:
        try:
            cleaned = re.sub(r',\s*([\]}])', r'\1', match_bracket.group(1).strip())
            return json.loads(cleaned)
        except Exception:
            pass

    # 其次匹配并提取 JSON 字典 { ... } 结构
    match_braces = re.search(r'(\{[\s\S]*\})', text)
    if match_braces:
        try:
            cleaned = re.sub(r',\s*([\]}])', r'\1', match_braces.group(1).strip())
            return json.loads(cleaned)
        except Exception:
            pass

    text = re.sub(r',\s*([\]}])', r'\1', text)
    return json.loads(text)

def generate_graph_based_qa(node_a_text, node_c_text, node_b_text, start_idx):
    prompt = f"""你是一个极其优秀的、专门用于测试 RAG 检索器图拓扑扩散能力的学术评测出题官。
以下是三段从三国小说中抽取的文本片段（在物理上它们并不相邻，但逻辑上它们通过特定的实体和因果逻辑关联，形成 A - C - B 的推理链条）：

【片段 A】（逻辑链条起点/提问背景和线索）：
{node_a_text}

【片段 C】（逻辑链条过渡桥梁/包含关键的推理衔接事实）：
{node_c_text}

【片段 B】（逻辑链条终点/包含答案）：
{node_b_text}

请结合这三个片段中的事实，设计 2 道【必须跨越这三个片段】才能回答的跨章节多跳隐式推理问题，并给出标准答案。这 2 道题应该从不同的逻辑角度或细节发问。

【设计准则】：
1. 答案（ground_truth）必须是片段 B 中出现的某个真实的中文人名或地名（如“关羽”、“许昌”等）。绝对不能包含任何伪装代号或多余的修饰字，必须仅为真实的人名或地名本身！
2. 问题本身中绝对不能出现任何真实的历史人名或地名，必须通过片段 A 中角色的外貌特征、动作、担任官职、所乘坐骑等隐式指代。
3. 必须设计成【三节点多跳推理】：问题的起因或线索在【片段 A】，但为了推导出最终答案，必须依赖【片段 C】中提供的衔接事实或逻辑过渡，最后指向位于【片段 B】的答案。如果只参考【片段 A】和【片段 B】，而丢失了【片段 C】作为逻辑桥梁，该问题是绝对无法被回答的。
4. 问题中绝对严禁提及【片段 A】、【片段 C】和【片段 B】中任何可辨识的具体情节动作或核心名词词汇（例如“鞭打”、“大怒”、“督邮”、“结义”、“怒”等）。
5. 必须将具体的情节、动作、原因和结果，翻译为高度抽象的隐式代称（如将“结盟/结义”化为“先前确立的人物关系/盟约”；将“张飞怒鞭督邮/羞辱”化为“后续发生的突发性肢体冲突/决策”等）。
6. 问题本身也严禁使用任何著名的历史事件专有名词（如“赤壁之战”、“桃园结义”、“凤仪亭”、“连环计”、“白帝城”等）。

请严格按照以下 JSON 格式输出包含 2 个问题字典的数组，不要包含任何多余解释、Markdown 代码标记或思考过程：
[
  {{
    "question": "第 1 个多跳推理问题（必须依赖 A-C-B 链条，不得出现任何真实历史人名/地名/典故名称/具体情节词汇）",
    "ground_truth": "标准答案（必须是片段 B 中出现的真实中文人名或地名，如“张飞”、“荆州”，不能包含其他字）"
  }},
  {{
    "question": "第 2 个多跳推理问题（必须依赖 A-C-B 链条，与第 1 个角度不同，不得出现任何真实历史人名/地名/典故名称/具体情节词汇）",
    "ground_truth": "标准答案（必须是片段 B 中出现的真实中文人名或地名，如“张飞”、“荆州”，不能包含其他字）"
  }}
]"""

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
                raw_data = extract_json(content)
                qa_list = []
                if isinstance(raw_data, list):
                    qa_list = raw_data
                elif isinstance(raw_data, dict):
                    if "questions" in raw_data:
                        qa_list = raw_data["questions"]
                    elif "items" in raw_data:
                        qa_list = raw_data["items"]
                    else:
                        qa_list = [raw_data]
                
                if isinstance(qa_list, list) and len(qa_list) > 0:
                    valid_pairs = []
                    for i, qa in enumerate(qa_list):
                        if "question" in qa and "ground_truth" in qa:
                            valid_pairs.append({
                                "id": start_idx + i,
                                "node_a_context": node_a_text,
                                "node_c_context": node_c_text,
                                "node_b_context": node_b_text,
                                "question": qa["question"],
                                "ground_truth": qa["ground_truth"].strip()
                            })
                    if len(valid_pairs) > 0:
                        return valid_pairs
                else:
                    print(f"  ⚠️ 尝试 {attempt} JSON 缺少/无法提取必要的键，原始解析为: {raw_data}")
            except Exception as e:
                print(f"  ⚠️ 尝试 {attempt} JSON 解析或过滤失败: {e}. 原始内容摘要: {content[:200]}...")
                
        except requests.exceptions.RequestException as req_err:
            print(f"  ⚠️ 尝试 {attempt} 网络或 API 请求错误: {req_err}")
            
    return None

def main():
    disguised_path = "tests/temp_data/三国演义白话文.txt"
    if not os.path.exists(disguised_path):
        print(f"❌ 未找到小说原文文件: {disguised_path}。")
        sys.exit(1)

    print("Step 1: Initializing RAG components and building NetworkX graph...")
    from src.loader import DocumentLoader
    from src.splitter import SemanticParentChildSplitter
    from src.embedding import LocalEmbeddingService
    from src.database import ChromaAdapter
    from src.coordinator import RAGCoordinator
    from src.graph_search import run_personalized_pagerank
    import networkx as nx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_dir = "E:/project/advanced-rag/vector_db"
    
    embedding_service = LocalEmbeddingService(device=device)
    splitter = SemanticParentChildSplitter(embedding_service=embedding_service, threshold=None, child_size=50, min_parent_size=300, max_parent_size=800)
    db_adapter = ChromaAdapter(db_dir=db_dir)

    # 干净的重置，以便出题前重建最新的图谱
    try:
        db_adapter.client.delete_collection(db_adapter.collection.name)
    except Exception:
        pass
    db_adapter.collection = db_adapter.client.get_or_create_collection(
        name=db_adapter.collection.name,
        metadata={"hnsw:space": "cosine"}
    )
    db_adapter.bm25 = None
    db_adapter.bm25_docs = []
    db_adapter.graph = nx.Graph()

    loader = DocumentLoader()
    coordinator = RAGCoordinator(
        loader=loader,
        splitter=splitter,
        embedding_service=embedding_service,
        db_adapter=db_adapter,
        reranker=None
    )

    print("Adding original book to build NetworkX graph...")
    coordinator.add_file(disguised_path)
    
    graph = db_adapter.graph
    print(f"Graph nodes: {len(graph.nodes)}, Graph edges: {len(graph.edges)}")

    # 计算物理相对位置
    with open(disguised_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    node_positions = []
    for node in graph.nodes:
        parent_text = graph.nodes[node].get("parent_text", "")
        pos = full_text.find(parent_text) if parent_text else -1
        if pos == -1:
            pos = 999999
        node_positions.append((node, pos))
    
    node_positions.sort(key=lambda x: x[1])
    node_index = {node: idx for idx, (node, _) in enumerate(node_positions)}

    # 计算核心实体关联
    core_ents = [ent for ent in BLACKLIST_ENTITIES if ent not in ["鞭打", "督邮", "大怒", "怒", "羞辱", "结义", "结盟", "盟约", "结拜", "托孤"]]
    
    SUPER_GLOBAL_ENTS = {"刘备", "曹操", "曹丕", "孙权", "诸葛亮"}
    
    def has_entity_relation(u, v):
        if graph.has_edge(u, v):
            text_u = graph.nodes[u].get("parent_text", "")
            text_v = graph.nodes[v].get("parent_text", "")
            shared = [ent for ent in core_ents if ent in text_u and ent in text_v]
            meaningful_shared = [ent for ent in shared if ent not in SUPER_GLOBAL_ENTS]
            return len(meaningful_shared) > 0
        return False

    print("Searching for 3-node logic chain A - (entity) - C - (entity) - B...")
    triplets = []
    seen_triplets = set()
    
    for C in graph.nodes:
        neighbors = list(graph.neighbors(C))
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                A = neighbors[i]
                B = neighbors[j]
                
                node_A, node_B = (A, B) if A < B else (B, A)
                triplet_key = (node_A, C, node_B)
                if triplet_key in seen_triplets:
                    continue
                seen_triplets.add(triplet_key)
                
                if has_entity_relation(node_A, C) and has_entity_relation(C, node_B):
                    dist = abs(node_index[node_A] - node_index[node_B])
                    if dist >= 8:
                        text_A = graph.nodes[node_A].get("parent_text", "")
                        text_C = graph.nodes[C].get("parent_text", "")
                        text_B = graph.nodes[node_B].get("parent_text", "")
                        if len(text_A) >= 150 and len(text_C) >= 150 and len(text_B) >= 150:
                            triplets.append((node_A, C, node_B, dist, text_A, text_C, text_B))

    print(f"Candidates satisfying A-C-B conditions: {len(triplets)}")
    
    # 按照 PPR 得分对三元组排序并选取前 30 对
    ppr_cache = {}
    def get_ppr_scores(seed):
        if seed not in ppr_cache:
            ppr_results = run_personalized_pagerank(graph, seed_node_id=seed, top_k=len(graph.nodes))
            ppr_cache[seed] = dict(ppr_results)
        return ppr_cache[seed]

    scored_triplets = []
    for node_A, C, node_B, dist, text_A, text_C, text_B in triplets:
        scores_A = get_ppr_scores(node_A)
        score = scores_A.get(C, 0.0) * scores_A.get(node_B, 0.0)
        scored_triplets.append((node_A, C, node_B, text_A, text_C, text_B, score))
        
    scored_triplets.sort(key=lambda x: x[6], reverse=True)

    relation_pairs = []
    a_used = {}
    c_used = {}
    b_used = {}
    for node_A, C, node_B, text_A, text_C, text_B, score in scored_triplets:
        if len(relation_pairs) >= 30:
            break
        if a_used.get(node_A, 0) < 2 and c_used.get(C, 0) < 2 and b_used.get(node_B, 0) < 2:
            relation_pairs.append((text_A, text_C, text_B, score))
            a_used[node_A] = a_used.get(node_A, 0) + 1
            c_used[C] = c_used.get(C, 0) + 1
            b_used[node_B] = b_used.get(node_B, 0) + 1

    print(f"Found {len(relation_pairs)} qualifying gold 3-node logic chains.")
    
    if not relation_pairs:
        print("⚠️ 未找到任何符合条件的三节点逻辑链，降级使用物理相邻节点对...")
        for u, v, d in graph.edges(data=True):
            text_u = graph.nodes[u].get("parent_text", "")
            text_v = graph.nodes[v].get("parent_text", "")
            if text_u and text_v:
                relation_pairs.append((text_u, text_u, text_v, 1.0))

    dataset = []
    idx = 1
    
    # 遍历这些三元组，让大模型跨这三个文本块出题
    for node_a_text, node_c_text, node_b_text, _score in relation_pairs:
        if len(dataset) >= 10:
            break
            
        print(f"🤖 Generating Graph-based Q&A (Current dataset size: {len(dataset)}/10)...")
        qa_pairs = generate_graph_based_qa(node_a_text, node_c_text, node_b_text, idx)
        if qa_pairs:
            for qa_pair in qa_pairs:
                if len(dataset) >= 10:
                    break
                q_text = qa_pair.get("question", "")
                gt_text = qa_pair.get("ground_truth", "")
                
                # 严格过滤
                has_blacklist = False
                for black_word in BLACKLIST_ENTITIES:
                    if black_word in q_text:
                        print(f"  ⚠️ 问题中包含黑名单词汇 '{black_word}'，废弃。")
                        has_blacklist = True
                        break
                if has_blacklist:
                    continue
                    
                if not re.match(r'^[\u4e00-\u9fa5]{2,6}$', gt_text):
                    print(f"  ⚠️ 答案格式不合规 (必须仅为 2-6 字的汉字实体): '{gt_text}'，废弃。")
                    continue
                    
                qa_pair["id"] = idx
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
