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

def run_personalized_pagerank(
    graph: nx.Graph, 
    seed_node_id: str, 
    top_k: int = 5, 
    edge_threshold: float = 0.0
) -> list[tuple[str, float]]:
    """
    以 seed_node_id 设为唯一能量源计算 2-Hop 剪枝子图的 Personalized PageRank 分数。
    :param graph: 内存 NetworkX 图
    :param seed_node_id: 起点（种子）节点 ID
    :param top_k: 捞回的 Top-K 节点数
    :param edge_threshold: 边权重过滤阈值
    :return: 捞回的父块 parent_id 和分数的元组列表（已过滤掉起点本身）
    """
    if seed_node_id not in graph:
        return []
    if len(graph.nodes) <= 1:
        return []
    
    # 提取强 1 跳邻居
    neighbors_1st = set()
    for n in graph.neighbors(seed_node_id):
        edge_data = graph.get_edge_data(seed_node_id, n)
        # 防御性：若旧测试图无 weight，取无穷大确保其不被过滤
        weight = edge_data.get("weight", float('inf'))
        if weight >= edge_threshold:
            neighbors_1st.add(n)
            
    # 提取强 2 跳邻居
    neighbors_2nd = set()
    for n1 in neighbors_1st:
        for n in graph.neighbors(n1):
            edge_data = graph.get_edge_data(n1, n)
            weight = edge_data.get("weight", float('inf'))
            if weight >= edge_threshold and n != seed_node_id:
                neighbors_2nd.add(n)
                
    # 组装节点集合并提取子图
    target_nodes = {seed_node_id} | neighbors_1st | neighbors_2nd
    if len(target_nodes) <= 1:  # 节点太少，不足以构成合理的转移链条
        return []
        
    sub_graph = graph.subgraph(target_nodes).copy()
    
    # 过滤子图中的弱边
    edges_to_remove = []
    for u, v, data in sub_graph.edges(data=True):
        weight = data.get("weight", float('inf'))
        if weight < edge_threshold:
            edges_to_remove.append((u, v))
    sub_graph.remove_edges_from(edges_to_remove)
    
    # 构造 personalization 字典，仅 seed_node_id 为 1.0，其余为 0.0
    personalization = {node: 0.0 for node in sub_graph.nodes}
    personalization[seed_node_id] = 1.0
    
    try:
        # 调用 nx.pagerank 计算子图节点分值
        scores = nx.pagerank(sub_graph, alpha=0.85, personalization=personalization, max_iter=100, weight='weight')
    except Exception:
        # 若因图未连通或计算不收敛抛出异常，做防御返回空
        return []
    
    # 根据分值降序排序，过滤掉 seed_node_id 自身后返回 Top-K 的 (parent_id, score) 元组列表
    # 以分数降序、节点 ID 字母序升序排序
    sorted_nodes = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    result = [(node, score) for node, score in sorted_nodes if node != seed_node_id]
    
    return result[:top_k]


def run_semantic_random_walk(graph: nx.Graph, seed_node_id: str, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
    """
    语义引导的 2跳随机游走：
    1. 第 1 跳：获取 Seed Node 的直接邻居，计算与 Query 的相似度，取 Top-3 个 1 跳节点。
    2. 第 2 跳：分别获取上述选中的 1 跳节点的直接邻居（排除起点），计算相似度并取 Top-2 个。
    3. 合并并去重所有选中的 1 跳与 2 跳邻居，取前 Top-K 返回。
    :param graph: 内存 NetworkX 图
    :param seed_node_id: 起点（种子）节点 ID
    :param query_vector: Query Dense 向量
    :param top_k: 返回的 Top-K 节点数
    :return: 融合去重后按与 Query 的相似度降序排列的 (parent_id, similarity) 元组列表
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
            
    # 按相似度降序排序（相似度相同时按节点 ID 字母序升序），取 Top-3 个 1 跳节点
    sims_1st.sort(key=lambda x: (-x[1], x[0]))
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
                
        # 降序排序（相似度相同时按节点 ID 字母序升序），取 Top-2 个
        sims_2nd.sort(key=lambda x: (-x[1], x[0]))
        for n2, _ in sims_2nd[:2]:
            candidates_2nd.add(n2)
            
    # 3. 合并并去重所有选中的 1 跳与 2 跳邻居
    all_selected = set(top_1st) | candidates_2nd
    
    # 按照与 query_vector 的相似度降序排序后（相似度相同时按节点 ID 字母序升序），截取前 Top-K 返回
    final_list = []
    for node in all_selected:
        emb = graph.nodes[node].get("embedding")
        sim = cosine_similarity(query_vector, emb) if emb is not None else -1.0
        final_list.append((node, sim))
        
    final_list.sort(key=lambda x: (-x[1], x[0]))
    return final_list[:top_k]


class GraphPostRetriever:
    def __init__(self, embedding_service, db_adapter, reranker):
        """
        初始化 GraphPostRetriever
        :param embedding_service: 向量生成服务实例 LocalEmbeddingService
        :param db_adapter: 向量数据库适配器实例 ChromaAdapter
        :param reranker: 重排服务实例 RerankerService
        """
        self.embedding_service = embedding_service
        self.db_adapter = db_adapter
        self.reranker = reranker

    def query_graph_enhanced(self, user_question: str, graph_search_mode: str = "heuristic_walk") -> str:
        """
        非对称 3+2 通道配额、免检图谱及 0.5 熔断门控。
        :param user_question: 用户提问
        :param graph_search_mode: 图检索模式，可选 "heuristic_walk"、"ppr" 或 "none"
        :return: 拼接格式化后的上下文字符串
        """
        # 1. 计算提问的 Dense 和 Sparse 向量
        dense_vec = self.embedding_service.get_dense_embedding(user_question)
        sparse_vec = self.embedding_service.get_sparse_embedding(user_question)

        # 2. 调用 db_adapter.hybrid_search 初筛去重召回 (Top 15)
        candidates = self.db_adapter.hybrid_search(dense_vec, sparse_vec, top_k=15)
        if not candidates:
            return ""

        # 3. 一阶段 Rerank 对初筛结果进行重排，不限制截断，传递大 threshold 防止内部截断
        try:
            first_rerank = self.reranker.rerank(user_question, candidates, top_k=len(candidates), cliff_threshold=999.0)
        except TypeError:
            first_rerank = self.reranker.rerank(user_question, candidates, top_k=len(candidates))
        if not first_rerank:
            return ""

        # 4. 向量自适应断崖检测：对前3个向量块(依据实际长度)做 1.5 得分差断崖检测，落差大于 1.5 则截断后续所有向量候选
        vector_results = first_rerank[:3]
        cutoff_idx = -1
        for i in range(len(vector_results) - 1):
            diff = vector_results[i]["rerank_score"] - vector_results[i+1]["rerank_score"]
            if diff > 1.5:
                cutoff_idx = i + 1
                break
        if cutoff_idx != -1:
            vector_results = vector_results[:cutoff_idx]

        # 5. 熔断门控：首位向量分数 S_seed < 0.5 时，熔断图谱，只返回上述截断后的向量候选块
        S_seed = first_rerank[0]["rerank_score"]
        if S_seed < 0.5 or graph_search_mode == "none":
            selected = vector_results
        else:
            # 未熔断
            # 以 V_1 的 parent_id 作为唯一种子节点 Seed
            seed_node_id = first_rerank[0]["metadata"].get("parent_id")
            
            graph_scores = []
            if seed_node_id:
                if graph_search_mode == "heuristic_walk":
                    graph_scores = run_semantic_random_walk(self.db_adapter.graph, seed_node_id, dense_vec, top_k=5)
                elif graph_search_mode == "ppr":
                    import numpy as np
                    edge_thr = float(np.exp(4 * 0.3))
                    graph_scores = run_personalized_pagerank(
                        self.db_adapter.graph, 
                        seed_node_id, 
                        top_k=5, 
                        edge_threshold=edge_thr
                    )
            
            # 解耦独立打分：
            # 若 ppr：Score_PPR(v) = S_seed * PPR(v)
            # 若 heuristic_walk：Score_HW(v) = S_seed * Sim(v, Q)
            calculated_graph_scores = []
            for pid, val in graph_scores:
                score = S_seed * val
                calculated_graph_scores.append((pid, score))
            
            # 图谱去重与比例断崖：
            # 去除已经被截断后保留的向量块占用的 parent_id
            vector_pids = {c["metadata"].get("parent_id") for c in vector_results}
            filtered_graph_scores = [item for item in calculated_graph_scores if item[0] not in vector_pids]
            
            # 排序与比例断崖截取：
            # 排序后的前 2 个图谱节点 G_1, G_2，如果 Score(G_2) < 0.4 * Score(G_1)，触发比例断崖，抛弃 G_2 仅保留 G_1。
            filtered_graph_scores.sort(key=lambda x: x[1], reverse=True)
            top_graph_results = []
            if len(filtered_graph_scores) == 1:
                top_graph_results = [filtered_graph_scores[0]]
            elif len(filtered_graph_scores) >= 2:
                g1 = filtered_graph_scores[0]
                g2 = filtered_graph_scores[1]
                if g2[1] < 0.4 * g1[1]:
                    top_graph_results = [g1]
                else:
                    top_graph_results = [g1, g2]
            
            top_graph_pids = [pid for pid, score in top_graph_results]
            
            # 补齐图谱节点信息
            candidates_dict = {c["metadata"]["parent_id"]: c for c in candidates}
            graph_vector_results = []
            for pid in top_graph_pids:
                if pid in candidates_dict:
                    graph_vector_results.append(candidates_dict[pid])
                elif pid in self.db_adapter.graph:
                    node_data = self.db_adapter.graph.nodes[pid]
                    graph_vector_results.append({
                        "content": node_data.get("parent_text", ""),
                        "metadata": {
                            "parent_id": pid,
                            "parent_text": node_data.get("parent_text", ""),
                            "source_path": node_data.get("source_path", ""),
                            "filename": node_data.get("filename", ""),
                            "char_start": node_data.get("char_start", 0),
                            "char_end": 0
                        }
                    })
            
            # GNN 空间紧邻拼接：拼接后返回的格式顺序确保为：[V_1, G_1, G_2, V_2, V_3]（或断崖裁剪后的实际子集）
            selected = []
            if len(vector_results) > 0:
                selected.append(vector_results[0])
            selected.extend(graph_vector_results)
            if len(vector_results) > 1:
                selected.extend(vector_results[1:])

        if not selected:
            return ""

        # 做题阶段：统一将选出的上下文片段按照在原著中的物理先后顺序（时间线）进行重排序，提高大模型理解的丝滑度
        selected = sorted(selected, key=lambda x: x["metadata"].get("char_start", 0))

        # 6. 执行父块替换并拼接格式化后的上下文字符串
        formatted_parts = []
        for idx, candidate in enumerate(selected, 1):
            filename = candidate["metadata"].get("filename", "未知文件")
            parent_text = candidate["metadata"].get("parent_text", "")
            part_str = f"[片段{idx}] (来源: {filename})\n{parent_text}"
            formatted_parts.append(part_str)

        return "\n\n".join(formatted_parts)

