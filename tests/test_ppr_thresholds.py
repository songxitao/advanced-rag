import os
import sys
import numpy as np
import networkx as nx

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
        print(f"❌ 未找到文档: {disguised_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_dir = "E:/project/advanced-rag/vector_db"
    
    embedding_service = LocalEmbeddingService(device=device)
    splitter = SemanticParentChildSplitter(embedding_service=embedding_service, threshold=None, child_size=50, min_parent_size=300, max_parent_size=800)
    db_adapter = ChromaAdapter(db_dir=db_dir)
    
    graph = db_adapter.graph
    n_nodes = len(graph.nodes)
    print(f"Graph loaded. Total Nodes (N): {n_nodes}, Total Edges: {len(graph.edges)}")
    
    if n_nodes == 0:
        print("❌ 图为空！请确认是否已经构建过数据库。")
        sys.exit(1)

    # 随机选择 20 个节点来进行统计分析
    import random
    random.seed(42)
    nodes_sample = random.sample(list(graph.nodes), min(20, n_nodes))
    
    # 统计指标容器
    all_scores = []
    direct_neighbor_scores = []
    hop2_neighbor_scores = []
    hop3_plus_scores = []
    non_reachable_scores = []

    print("\n--- 正在分析 20 个随机种子节点的 PPR 扩散分数分布 ---")
    for seed in nodes_sample:
        # 计算该 seed 的全图 PPR 分数 (top_k=len(graph.nodes) 取得全图数据)
        ppr_results = run_personalized_pagerank(graph, seed_node_id=seed, top_k=n_nodes)
        ppr_dict = dict(ppr_results)
        
        # 1. 记录全图除 seed 外所有分数
        all_scores.extend(ppr_dict.values())
        
        # 2. 计算邻近度跳数（使用 BFS 计算物理图上的最短路径跳数）
        for node, score in ppr_dict.items():
            if node == seed:
                continue
            try:
                # 寻找最短路径
                path_len = nx.shortest_path_length(graph, source=seed, target=node)
                if path_len == 1:
                    direct_neighbor_scores.append(score)
                elif path_len == 2:
                    hop2_neighbor_scores.append(score)
                else:
                    hop3_plus_scores.append(score)
            except nx.NetworkXNoPath:
                non_reachable_scores.append(score)

    # 打印全局分布情况
    all_scores = np.array(all_scores)
    print("\n[📊 全局 PPR 分数分布统计]")
    print(f"平均分数 (Mean): {np.mean(all_scores):.6f}")
    print(f"最大值 (Max): {np.max(all_scores):.6f}")
    print(f"中位数 (Median): {np.median(all_scores):.6f}")
    print(f"标准差 (Std Dev): {np.std(all_scores):.6f}")
    print(f"90% 分位数 (P90): {np.percentile(all_scores, 90):.6f}")
    print(f"95% 分位数 (P95): {np.percentile(all_scores, 95):.6f}")
    print(f"99% 分位数 (P99): {np.percentile(all_scores, 99):.6f}")
    print(f"均匀随机基准期望 (1/N): {1.0 / n_nodes:.6f}")

    # 打印按跳数划分的分布
    print("\n[🗺️ 拓扑距离(跳数)与 PPR 分数的对应分布]")
    
    if direct_neighbor_scores:
        dns = np.array(direct_neighbor_scores)
        print(f"1 跳直接邻居 (共 {len(dns)} 个):")
        print(f"  均值: {np.mean(dns):.6f} | 区间: [{np.min(dns):.6f}, {np.max(dns):.6f}]")
    else:
        print("1 跳直接邻居: 无数据")
        
    if hop2_neighbor_scores:
        h2s = np.array(hop2_neighbor_scores)
        print(f"2 跳衔接邻居 (共 {len(h2s)} 个):")
        print(f"  均值: {np.mean(h2s):.6f} | 区间: [{np.min(h2s):.6f}, {np.max(h2s):.6f}]")
    else:
        print("2 跳衔接邻居: 无数据")
        
    if hop3_plus_scores:
        h3s = np.array(hop3_plus_scores)
        print(f"3跳及更远节点 (共 {len(h3s)} 个):")
        print(f"  均值: {np.mean(h3s):.6f} | 区间: [{np.min(h3s):.6f}, {np.max(h3s):.6f}]")
    else:
        print("3跳及更远节点: 无数据")

    if non_reachable_scores:
        nrs = np.array(non_reachable_scores)
        print(f"逻辑不可达节点 (共 {len(nrs)} 个):")
        print(f"  均值: {np.mean(nrs):.6f} | 区间: [{np.min(nrs):.6f}, {np.max(nrs):.6f}]")
    else:
        print("逻辑不可达节点: 无数据")

if __name__ == "__main__":
    main()
