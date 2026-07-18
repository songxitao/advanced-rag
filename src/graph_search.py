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
    top_k: int = 5
) -> list[tuple[str, float]]:
    """
    以 seed_node_id 设为唯一能量源计算 2-Hop 剪枝子图的 Personalized PageRank 分数。
    :param graph: 内存 NetworkX 图
    :param seed_node_id: 起点（种子）节点 ID
    :param top_k: 捞回的 Top-K 节点数
    :return: 捞回的父块 parent_id 和分数的元组列表（已过滤掉起点本身）
    """
    if seed_node_id not in graph:
        return []
    if len(graph.nodes) <= 1:
        return []
    
    # 提取所有 1 跳邻居
    neighbors_1st = set(graph.neighbors(seed_node_id))
            
    # 提取所有 2 跳邻居
    neighbors_2nd = set()
    for n1 in neighbors_1st:
        for n in graph.neighbors(n1):
            if n != seed_node_id:
                neighbors_2nd.add(n)
                
    # 组装节点集合并提取子图
    target_nodes = {seed_node_id} | neighbors_1st | neighbors_2nd
    if len(target_nodes) <= 1:  # 节点太少，不足以构成合理的转移链条
        return []
        
    sub_graph = graph.subgraph(target_nodes).copy()
    
    # 构造 personalization 字典，仅 seed_node_id 为 1.0，其余为 0.0
    personalization = {node: 0.0 for node in sub_graph.nodes}
    personalization[seed_node_id] = 1.0
    
    try:
        # 检测子图是否有 weight 属性（生产图有，测试图可能没有），避免新版 NetworkX 抛异常
        has_weight = any('weight' in d for _, _, d in sub_graph.edges(data=True))
        weight_param = 'weight' if has_weight else None
        # 调用 nx.pagerank 计算子图节点分值
        scores = nx.pagerank(sub_graph, alpha=0.85, personalization=personalization, max_iter=100, weight=weight_param)
    except Exception as e:
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
        混合了向量候选与图游走候选的全局混合二次 Rerank。
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

        # 4. 熔断门控：首位向量分数 S_seed < 0.5 时，熔断图谱，只返回向量候选（走旧的 1.5 得分差断崖检测和 Top 3）
        S_seed = first_rerank[0]["rerank_score"]
        if S_seed < 0.5 or graph_search_mode == "none":
            # 沿用之前的逻辑：对前3个向量块做 1.5 得分差断崖检测，落差大于 1.5 则截断
            vector_results = first_rerank[:3]
            cutoff_idx = -1
            for i in range(len(vector_results) - 1):
                diff = vector_results[i]["rerank_score"] - vector_results[i+1]["rerank_score"]
                if diff > 1.5:
                    cutoff_idx = i + 1
                    break
            if cutoff_idx != -1:
                selected = vector_results[:cutoff_idx]
            else:
                selected = vector_results
        else:
            # 未熔断：
            # 以一阶段 Rerank 的 Top 1 名向量的 parent_id 作为唯一种子节点 Seed
            seed_node_id = first_rerank[0]["metadata"].get("parent_id")
            
            graph_scores = []
            if seed_node_id:
                if graph_search_mode == "heuristic_walk":
                    graph_scores = run_semantic_random_walk(self.db_adapter.graph, seed_node_id, dense_vec, top_k=5)
                elif graph_search_mode == "ppr":
                    graph_scores = run_personalized_pagerank(
                        self.db_adapter.graph, 
                        seed_node_id, 
                        top_k=5
                    )
            
            # 将图候选转为 candidate dict（使用 parent_text 作为 content）
            graph_candidates = []
            for pid, _ in graph_scores:
                if pid in self.db_adapter.graph:
                    node_data = self.db_adapter.graph.nodes[pid]
                    graph_candidates.append({
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
            
            # 去重合并：向量 Top5 + 图候选 Top5 -> 统一候选池
            vector_candidates = first_rerank[:5]
            vector_pids = {c["metadata"].get("parent_id") for c in vector_candidates if c.get("metadata")}
            
            filtered_graph_candidates = []
            for gc in graph_candidates:
                g_pid = gc["metadata"].get("parent_id")
                if g_pid not in vector_pids:
                    filtered_graph_candidates.append(gc)
            
            combined_candidates = vector_candidates + filtered_graph_candidates
            
            # 全局二次 Rerank（CrossEncoder 打分）
            try:
                second_rerank = self.reranker.rerank(user_question, combined_candidates, top_k=len(combined_candidates), cliff_threshold=999.0)
            except TypeError:
                second_rerank = self.reranker.rerank(user_question, combined_candidates, top_k=len(combined_candidates))
            
            # 自适应断崖截断
            cutoff_idx = -1
            for i in range(len(second_rerank) - 1):
                diff = second_rerank[i]["rerank_score"] - second_rerank[i+1]["rerank_score"]
                if diff > 1.5:
                    cutoff_idx = i + 1
                    break
            if cutoff_idx != -1:
                second_rerank = second_rerank[:cutoff_idx]
            
            # 取 Top5
            selected = second_rerank[:5]
            
        if not selected:
            return ""

        # 统一将选出的上下文片段按照在原著中的物理先后顺序（时间线）进行重排序，提高大模型理解的丝滑度
        selected = sorted(selected, key=lambda x: x["metadata"].get("char_start", 0))

        # 6. 执行父块替换并拼接格式化后的上下文字符串
        formatted_parts = []
        for idx, candidate in enumerate(selected, 1):
            filename = candidate["metadata"].get("filename", "未知文件")
            parent_text = candidate["metadata"].get("parent_text", "")
            part_str = f"[片段{idx}] (来源: {filename})\n{parent_text}"
            formatted_parts.append(part_str)

        return "\n\n".join(formatted_parts)

