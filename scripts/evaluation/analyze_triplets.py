import os
import sys
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
    
    # 重新加载或重建
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

    import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.evaluation_set_generator_graph import BLACKLIST_ENTITIES
    core_ents = [ent for ent in BLACKLIST_ENTITIES if ent not in ["鞭打", "督邮", "大怒", "怒", "羞辱", "结义", "结盟", "盟约", "结拜", "托孤"]]

    def has_entity_relation(u, v):
        if graph.has_edge(u, v):
            text_u = graph.nodes[u].get("parent_text", "")
            text_v = graph.nodes[v].get("parent_text", "")
            shared = [ent for ent in core_ents if ent in text_u and ent in text_v]
            return len(shared) > 0
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

    print(f"Total triplets found: {len(triplets)}")

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
        # 记录节点 ID 以便分析重复
        scored_triplets.append((node_A, C, node_B, score, text_A[:40], text_C[:40], text_B[:40]))

    scored_triplets.sort(key=lambda x: x[3], reverse=True)

    print("\n--- TOP 30 Scored Triplets ---")
    for idx, (nA, nC, nB, score, tA, tC, tB) in enumerate(scored_triplets[:30]):
        print(f"Rank {idx+1}: Score={score:.6f} | A={nA} ({tA}...) | C={nC} ({tC}...) | B={nB} ({tB}...)")

    print("\n--- Node A Frequencies in Top 30 ---")
    from collections import Counter
    a_nodes = [x[0] for x in scored_triplets[:30]]
    c_nodes = [x[1] for x in scored_triplets[:30]]
    b_nodes = [x[2] for x in scored_triplets[:30]]
    print("A nodes count:", Counter(a_nodes))
    print("C nodes count:", Counter(c_nodes))
    print("B nodes count:", Counter(b_nodes))

if __name__ == "__main__":
    main()
