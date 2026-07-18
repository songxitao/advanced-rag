import os
import sys
import json
import networkx as nx
import numpy as np

# 配置 sys.path
current_project_path = "E:/project/advanced-rag"
if current_project_path not in sys.path:
    sys.path.insert(0, current_project_path)

from src.loader import DocumentLoader
from src.splitter import SemanticParentChildSplitter
from src.embedding import LocalEmbeddingService
from src.database import ChromaAdapter
from src.coordinator import RAGCoordinator
from src.graph_search import run_personalized_pagerank
import torch

def main():
    disguised_path = "tests/temp_data/三国演义白话文.txt"
    if not os.path.exists(disguised_path):
        print(f"❌ 未找到文件: {disguised_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_dir = "E:/project/advanced-rag/vector_db"
    
    embedding_service = LocalEmbeddingService(device=device)
    splitter = SemanticParentChildSplitter(embedding_service=embedding_service, threshold=None, child_size=50, min_parent_size=300, max_parent_size=800)
    db_adapter = ChromaAdapter(db_dir=db_dir)
    
    graph = db_adapter.graph
    print(f"Loaded graph. Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

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

    from tests.evaluation_set_generator_graph import BLACKLIST_ENTITIES
    core_ents = [ent for ent in BLACKLIST_ENTITIES if ent not in ["鞭打", "督邮", "大怒", "怒", "羞辱", "结义", "结盟", "盟约", "结拜", "托孤"]]

    SUPER_GLOBAL_ENTS = {"刘备", "曹操", "曹丕", "孙权", "诸葛亮"}

    def get_shared_entities(text_u, text_v):
        return [ent for ent in core_ents if ent in text_u and ent in text_v]

    def has_entity_relation(u, v):
        if graph.has_edge(u, v):
            text_u = graph.nodes[u].get("parent_text", "")
            text_v = graph.nodes[v].get("parent_text", "")
            shared = get_shared_entities(text_u, text_v)
            meaningful_shared = [ent for ent in shared if ent not in SUPER_GLOBAL_ENTS]
            return len(meaningful_shared) > 0
        return False

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

    print(f"Total triplets satisfying new rules: {len(triplets)}")

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
        scored_triplets.append((node_A, C, node_B, dist, text_A, text_C, text_B, score))

    scored_triplets.sort(key=lambda x: x[7], reverse=True)

    # 贪心去重挑选前 5 对
    sample_pairs = []
    a_used = {}
    c_used = {}
    b_used = {}
    for node_A, C, node_B, dist, text_A, text_C, text_B, score in scored_triplets:
        if len(sample_pairs) >= 5:
            break
        if a_used.get(node_A, 0) < 2 and c_used.get(C, 0) < 2 and b_used.get(node_B, 0) < 2:
            shared_AC = get_shared_entities(text_A, text_C)
            shared_CB = get_shared_entities(text_C, text_B)
            sample_pairs.append({
                "rank": len(sample_pairs) + 1,
                "score": score,
                "physical_dist": dist,
                "node_A_id": node_A,
                "node_C_id": C,
                "node_B_id": node_B,
                "shared_AC_entities": [e for e in shared_AC if e not in SUPER_GLOBAL_ENTS],
                "shared_CB_entities": [e for e in shared_CB if e not in SUPER_GLOBAL_ENTS],
                "node_A_text": text_A,
                "node_C_text": text_C,
                "node_B_text": text_B
            })
            a_used[node_A] = a_used.get(node_A, 0) + 1
            c_used[C] = c_used.get(C, 0) + 1
            b_used[node_B] = b_used.get(node_B, 0) + 1

    output_file = "tests/temp_data/sample_triplets.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_pairs, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved 5 samples to {output_file}")

if __name__ == "__main__":
    main()
