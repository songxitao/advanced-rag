import pytest
import networkx as nx
from src.graph_search import run_personalized_pagerank, run_semantic_random_walk, cosine_similarity, GraphPostRetriever

def test_cosine_similarity():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    v3 = [1.0, 0.0]
    assert cosine_similarity(v1, v2) == 0.0
    assert cosine_similarity(v1, v3) == 1.0
    # 测试全零向量防御
    assert cosine_similarity(v1, [0.0, 0.0]) == 0.0

def test_run_personalized_pagerank():
    # 构造内存图
    g = nx.Graph()
    # 节点
    g.add_node("p1", embedding=[1.0, 0.0])
    g.add_node("p2", embedding=[0.9, 0.1])
    g.add_node("p3", embedding=[0.0, 1.0])
    g.add_node("p4", embedding=[0.1, 0.9])
    
    # 建立一条链式连边
    g.add_edge("p1", "p2")
    g.add_edge("p2", "p3")
    g.add_edge("p3", "p4")
    
    # 从 p1 出发进行 PPR
    res = run_personalized_pagerank(g, "p1", top_k=3)
    assert len(res) > 0
    # 此时返回 list[tuple[str, float]]，提取其 pid
    pids = [node for node, score in res]
    # 种子节点 p1 必须被排除在外
    assert "p1" not in pids
    # 距离 p1 较近的 p2 应排在前面
    assert pids[0] == "p2"

    # 防御性测试：不存在的节点
    assert run_personalized_pagerank(g, "non_existent", top_k=3) == []
    # 防御性测试：空图
    assert run_personalized_pagerank(nx.Graph(), "p1", top_k=3) == []

def test_run_semantic_random_walk():
    # 构造内存图
    g = nx.Graph()
    g.add_node("p1", embedding=[1.0, 0.0])
    g.add_node("p2", embedding=[0.8, 0.6]) # 1跳
    g.add_node("p3", embedding=[0.1, 0.9]) # 2跳
    g.add_node("p4", embedding=[0.0, 1.0]) # 3跳
    
    g.add_edge("p1", "p2")
    g.add_edge("p2", "p3")
    g.add_edge("p3", "p4")
    
    # 设定 Query 向量
    query_vector = [0.85, 0.52]
    # 从 p1 出发，1跳节点有 p2；2跳节点有 p3（排除起点 p1）
    res = run_semantic_random_walk(g, "p1", query_vector, top_k=2)
    assert len(res) > 0
    # 此时返回 list[tuple[str, float]]，提取其 pid
    pids = [node for node, score in res]
    assert "p1" not in pids
    assert "p2" in pids
    assert "p3" in pids

    # 防御性测试：不存在的节点
    assert run_semantic_random_walk(g, "non_existent", query_vector, top_k=2) == []
    # 防御性测试：无邻居节点的孤立节点
    g.add_node("p_isolated", embedding=[0.5, 0.5])
    assert run_semantic_random_walk(g, "p_isolated", query_vector, top_k=2) == []

def test_coordinator_graph_retrieval_and_cliff(tmp_path):
    from src.loader import DocumentLoader
    from src.splitter import SemanticParentChildSplitter
    from src.embedding import LocalEmbeddingService
    from src.database import ChromaAdapter
    from src.reranker import RerankerService
    from src.coordinator import RAGCoordinator
    
    db_dir = tmp_path / "test_coordinator_graph_db"
    loader = DocumentLoader()
    emb = LocalEmbeddingService(device="cpu")
    # 设置较大的阈值，以确保不同主题的句子被明确切分为不同的父块
    splitter = SemanticParentChildSplitter(embedding_service=emb, threshold=0.9, child_size=15)
    db = ChromaAdapter(db_dir=str(db_dir))
    reranker = RerankerService(device="cpu")
    
    coordinator = RAGCoordinator(loader, splitter, emb, db, reranker)
    retriever = GraphPostRetriever(emb, db, reranker)
    
    # 写入同一个文件，用换行符分开，生成物理相邻的三段：
    # 1. 特工A01持有关于绝密计划的核心图纸。 (特工核心)
    # 2. 早稻田大学的大楼就在图纸上标明的位置附近。 (1跳物理相邻，弱相关)
    # 3. 今天天气真好，小红在家里开心地吃苹果和梨子。 (2跳物理相邻，完全不相关)
    test_file = tmp_path / "test_cliff.txt"
    test_file.write_text(
        "特工A01持有关于绝密计划的核心图纸。\n\n"
        "早稻田大学的大楼就在图纸上标明的位置附近。\n\n"
        "今天天气真好，小红在家里开心地吃苹果和梨子。", 
        encoding="utf-8"
    )
    
    coordinator.add_file(str(test_file))
    
    # 调试打印 chunks
    print("\n--- DEBUG CHUNKS ---")
    count = db.collection.count()
    print("Chroma total count:", count)
    all_data = db.collection.get(include=["metadatas"])
    for m in all_data.get("metadatas", []):
        print("Meta parent_id:", m.get("parent_id"), "parent_text:", m.get("parent_text"))
    
    # 1. 验证以 heuristic_walk 模式能够把 1跳的“早稻田大学”捞出
    context_walk = retriever.query_graph_enhanced("特工A01的图纸在哪里？", graph_search_mode="heuristic_walk")
    print("\n--- DEBUG context_walk ---")
    print(context_walk)
    assert "特工A01" in context_walk
    assert "早稻田大学" in context_walk

    # 2. 验证以 ppr 模式也能够把“早稻田大学”捞出
    context_ppr = retriever.query_graph_enhanced("特工A01的图纸在哪里？", graph_search_mode="ppr")
    assert "特工A01" in context_ppr
    assert "早稻田大学" in context_ppr

def test_asymmetric_quota_and_fuse():
    # 模拟 EmbeddingService
    class MockEmbeddingService:
        def get_dense_embedding(self, query):
            return [0.1, 0.2]
        def get_sparse_embedding(self, query):
            return {}

    # 模拟 db_adapter
    class MockDBAdapter:
        def __init__(self):
            self.graph = nx.Graph()
            # 添加节点和节点属性，以便能补齐信息
            self.graph.add_node("vector_1", parent_text="Vector Context 1", filename="doc1.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("vector_2", parent_text="Vector Context 2", filename="doc1.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("vector_3", parent_text="Vector Context 3", filename="doc1.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("vector_4", parent_text="Vector Context 4", filename="doc1.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("graph_1", parent_text="Graph Context 1", filename="doc1.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("graph_2", parent_text="Graph Context 2", filename="doc1.txt", source_path="", embedding=[0.1, 0.2])
            
            # 建图谱邻边，让 seeds node 能游走到图谱节点
            self.graph.add_edge("vector_1", "graph_1")
            self.graph.add_edge("vector_1", "graph_2")
            self.graph.add_edge("vector_1", "vector_2") # 用于去重测试

        def hybrid_search(self, dense_vec, sparse_vec, top_k):
            # 初筛召回 candidate 列表
            return [
                {"metadata": {"parent_id": "vector_1", "filename": "doc1.txt", "parent_text": "Vector Context 1"}, "content": "Vector Context 1"},
                {"metadata": {"parent_id": "vector_2", "filename": "doc1.txt", "parent_text": "Vector Context 2"}, "content": "Vector Context 2"},
                {"metadata": {"parent_id": "vector_3", "filename": "doc1.txt", "parent_text": "Vector Context 3"}, "content": "Vector Context 3"},
                {"metadata": {"parent_id": "vector_4", "filename": "doc1.txt", "parent_text": "Vector Context 4"}, "content": "Vector Context 4"},
            ]

    # 模拟 RerankerService
    class MockRerankerService:
        def __init__(self, score_override=None):
            self.score_override = score_override

        def rerank(self, query, candidates, top_k):
            results = []
            for i, c in enumerate(candidates):
                score = self.score_override[i] if self.score_override and i < len(self.score_override) else 0.8
                results.append({
                    "content": c["content"],
                    "metadata": c["metadata"],
                    "rerank_score": score
                })
            # 排序
            return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    emb = MockEmbeddingService()
    db = MockDBAdapter()

    # Case 1: 首位得分 < 0.5，触发熔断，只返回 3 个向量块，不进行图游走
    reranker_fuse = MockRerankerService(score_override=[0.4, 0.3, 0.2, 0.1])
    retriever_fuse = GraphPostRetriever(emb, db, reranker_fuse)
    
    # 熔断时，即使 graph_search_mode="heuristic_walk"，也不应去游走图
    context_fuse = retriever_fuse.query_graph_enhanced("test query", graph_search_mode="heuristic_walk")
    # 应当只包含 vector_1, vector_2, vector_3，且不包含任何 graph 节点
    assert "Vector Context 1" in context_fuse
    assert "Vector Context 2" in context_fuse
    assert "Vector Context 3" in context_fuse
    assert "Vector Context 4" not in context_fuse
    assert "Graph Context" not in context_fuse
    # 片段数量应该正好是 3 个
    assert "[片段1]" in context_fuse
    assert "[片段2]" in context_fuse
    assert "[片段3]" in context_fuse
    assert "[片段4]" not in context_fuse

    # Case 2: 正常情况（第一名分数 >= 0.5），非对称配额（向量 3 + 图谱 2），包含去重和强行拼接
    # 第一名是 vector_1。
    # 图检索 ppr 或 walk 应该捞出 vector_2 (已在向量通道前3中), graph_1, graph_2
    # 去重后剩余 graph_1, graph_2，作为图谱免检通道的 2 个强行拼在向量通道后
    reranker_normal = MockRerankerService(score_override=[0.9, 0.8, 0.7, 0.6])
    retriever_normal = GraphPostRetriever(emb, db, reranker_normal)
    
    # 我们调用 query_graph_enhanced，在图里 vector_1 的邻居有 vector_2 (相似度/PPR得分), graph_1, graph_2
    context_normal = retriever_normal.query_graph_enhanced("test query", graph_search_mode="heuristic_walk")
    
    # 结果应为 vector_1, vector_2, vector_3 + graph_1, graph_2
    assert "Vector Context 1" in context_normal
    assert "Vector Context 2" in context_normal
    assert "Vector Context 3" in context_normal
    assert "Graph Context 1" in context_normal
    assert "Graph Context 2" in context_normal
    assert "Vector Context 4" not in context_normal
    
    # 共 5 个片段
    assert "[片段1]" in context_normal
    assert "[片段2]" in context_normal
    assert "[片段3]" in context_normal
    assert "[片段4]" in context_normal
    assert "[片段5]" in context_normal
    assert "[片段6]" not in context_normal
