import networkx as nx
import numpy as np

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    计算两个 Dense 向量的余弦相似度。
    """
    arr1 = np.array(v1, dtype=np.float32)
    arr2 = np.array(v2, dtype=np.float32)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))

def run_personalized_pagerank(graph: nx.Graph, seed_node_id: str, top_k: int = 5) -> list[str]:
    """
    以 seed_node_id 设为唯一能量源计算全图 Personalized PageRank 分数。
    :param graph: 内存 NetworkX 图
    :param seed_node_id: 起点（种子）节点 ID
    :param top_k: 捞回的 Top-K 节点数
    :return: 捞回的父块 parent_id 列表（已过滤掉起点本身）
    """
    if seed_node_id not in graph:
        return []
    if len(graph.nodes) <= 1:
        return []
    
    # 构造 personalization 字典，仅 seed_node_id 为 1.0，其余为 0.0
    personalization = {node: 0.0 for node in graph.nodes}
    personalization[seed_node_id] = 1.0
    
    try:
        # 调用 nx.pagerank 计算全图节点分值
        scores = nx.pagerank(graph, alpha=0.85, personalization=personalization, max_iter=100)
    except Exception:
        # 若因图未连通或计算不收敛抛出异常，做防御返回空
        return []
    
    # 根据分值降序排序，过滤掉 seed_node_id 自身后返回 Top-K 的 parent_id 列表
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = [node for node, score in sorted_nodes if node != seed_node_id]
    
    return result[:top_k]

def run_semantic_random_walk(graph: nx.Graph, seed_node_id: str, query_vector: list[float], top_k: int = 5) -> list[str]:
    """
    语义引导的 2跳随机游走：
    1. 第 1 跳：获取 Seed Node 的直接邻居，计算与 Query 的相似度，取 Top-3 个 1 跳节点。
    2. 第 2 跳：分别获取上述选中的 1 跳节点的直接邻居（排除起点），计算相似度并取 Top-2 个。
    3. 合并并去重所有选中的 1 跳与 2 跳邻居，取前 Top-K 返回。
    :param graph: 内存 NetworkX 图
    :param seed_node_id: 起点（种子）节点 ID
    :param query_vector: Query Dense 向量
    :param top_k: 返回的 Top-K 节点数
    :return: 融合去重后按与 Query 的相似度降序排列的 parent_id 列表
    """
    if seed_node_id not in graph:
        return []
    
    # 1. 第 1 跳：获取 Seed Node 的直接邻居
    neighbors_1st = list(graph.neighbors(seed_node_id))
    if not neighbors_1st:
        return []
    
    sims_1st = []
    for n in neighbors_1st:
        emb = graph.nodes[n].get("embedding")
        if emb is not None:
            sim = cosine_similarity(query_vector, emb)
            sims_1st.append((n, sim))
            
    # 按相似度降序排序，取 Top-3 个 1 跳节点
    sims_1st.sort(key=lambda x: x[1], reverse=True)
    top_1st = [n for n, _ in sims_1st[:3]]
    
    # 2. 第 2 跳：分别获取上述选中的 1 跳节点的直接邻居
    candidates_2nd = set()
    for n1 in top_1st:
        neighbors_2nd = list(graph.neighbors(n1))
        # 排除起点 seed_node_id
        neighbors_2nd_filtered = [n for n in neighbors_2nd if n != seed_node_id]
        
        sims_2nd = []
        for n2 in neighbors_2nd_filtered:
            emb = graph.nodes[n2].get("embedding")
            if emb is not None:
                sim = cosine_similarity(query_vector, emb)
                sims_2nd.append((n2, sim))
                
        # 降序排序，取 Top-2 个
        sims_2nd.sort(key=lambda x: x[1], reverse=True)
        for n2, _ in sims_2nd[:2]:
            candidates_2nd.add(n2)
            
    # 3. 合并并去重所有选中的 1 跳与 2 跳邻居
    all_selected = set(top_1st) | candidates_2nd
    
    # 按照与 query_vector 的相似度降序排序后，截取前 Top-K 返回
    final_list = []
    for node in all_selected:
        emb = graph.nodes[node].get("embedding")
        sim = cosine_similarity(query_vector, emb) if emb is not None else -1.0
        final_list.append((node, sim))
        
    final_list.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in final_list[:top_k]]
