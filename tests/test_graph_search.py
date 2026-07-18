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

@pytest.mark.slow
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
        def __init__(self, score_map):
            self.score_map = score_map

        def rerank(self, query, candidates, top_k, cliff_threshold=999.0):
            results = []
            for c in candidates:
                pid = c["metadata"].get("parent_id")
                score = self.score_map.get(pid, 0.8)
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
    score_map_fuse = {"vector_1": 0.4, "vector_2": 0.3, "vector_3": 0.2, "vector_4": 0.1}
    reranker_fuse = MockRerankerService(score_map_fuse)
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

    # Case 2: 正常情况（第一名分数 >= 0.5），合并去重后进行全局二次重排打分
    # 设定分数：V1=0.9, V2=0.8, V3=0.7, V4=0.6, G1=0.85, G2=0.85
    # 去重后候选池：[vector_1, vector_2, vector_3, vector_4] + [graph_1, graph_2]
    # 二阶段打分排序后：vector_1 (0.9), graph_1 (0.85), graph_2 (0.85), vector_2 (0.8), vector_3 (0.7), vector_4 (0.6)
    # 无自适应断崖截断，最终取 Top 5: vector_1, graph_1, graph_2, vector_2, vector_3
    score_map_normal = {
        "vector_1": 0.9,
        "vector_2": 0.8,
        "vector_3": 0.7,
        "vector_4": 0.6,
        "graph_1": 0.85,
        "graph_2": 0.85
    }
    reranker_normal = MockRerankerService(score_map_normal)
    retriever_normal = GraphPostRetriever(emb, db, reranker_normal)
    
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


def test_asymmetric_quota_cliff_and_reorder():
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
            # 添加节点属性
            self.graph.add_node("V_1", parent_text="Vector Context 1", filename="doc.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("V_2", parent_text="Vector Context 2", filename="doc.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("V_3", parent_text="Vector Context 3", filename="doc.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("V_4", parent_text="Vector Context 4", filename="doc.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("G_1", parent_text="Graph Context 1", filename="doc.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("G_2", parent_text="Graph Context 2", filename="doc.txt", source_path="", embedding=[0.1, 0.2])
            self.graph.add_node("G_3", parent_text="Graph Context 3", filename="doc.txt", source_path="", embedding=[0.1, 0.2])
            
            # 建立图谱连边，用于 walk 和 ppr 游走
            self.graph.add_edge("V_1", "G_1")
            self.graph.add_edge("V_1", "G_2")
            self.graph.add_edge("V_1", "G_3")

        def hybrid_search(self, dense_vec, sparse_vec, top_k):
            return [
                {"metadata": {"parent_id": "V_1", "filename": "doc.txt", "parent_text": "Vector Context 1"}, "content": "Vector Context 1"},
                {"metadata": {"parent_id": "V_2", "filename": "doc.txt", "parent_text": "Vector Context 2"}, "content": "Vector Context 2"},
                {"metadata": {"parent_id": "V_3", "filename": "doc.txt", "parent_text": "Vector Context 3"}, "content": "Vector Context 3"},
                {"metadata": {"parent_id": "V_4", "filename": "doc.txt", "parent_text": "Vector Context 4"}, "content": "Vector Context 4"},
            ]

    # 模拟 RerankerService
    class MockRerankerService:
        def __init__(self, score_map):
            self.score_map = score_map

        def rerank(self, query, candidates, top_k, cliff_threshold=999.0):
            results = []
            for c in candidates:
                pid = c["metadata"].get("parent_id")
                score = self.score_map.get(pid, 0.8)
                results.append({
                    "content": c["content"],
                    "metadata": c["metadata"],
                    "rerank_score": score
                })
            # 降序排序
            return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    emb = MockEmbeddingService()
    db = MockDBAdapter()

    # 1. 全局混合二次 Rerank 下的自适应断崖截断
    # 设分数 V_1=3.0, V_2=1.0, V_3=0.8, V_4=0.6, G_1=2.9, G_2=2.8, G_3=0.5
    # 排序为: V_1(3.0), G_1(2.9), G_2(2.8), V_2(1.0), V_3(0.8), V_4(0.6), G_3(0.5)
    # G_2 到 V_2 得分落差 2.8 - 1.0 = 1.8 > 1.5，触发差值断崖，从 V_2 处截断。
    # 最终保留: V_1, G_1, G_2
    score_map_cliff1 = {
        "V_1": 3.0,
        "V_2": 1.0,
        "V_3": 0.8,
        "V_4": 0.6,
        "G_1": 2.9,
        "G_2": 2.8,
        "G_3": 0.5
    }
    reranker_cliff1 = MockRerankerService(score_map_cliff1)
    retriever_cliff1 = GraphPostRetriever(emb, db, reranker_cliff1)
    context_cliff1 = retriever_cliff1.query_graph_enhanced("test", graph_search_mode="heuristic_walk")
    assert "Vector Context 1" in context_cliff1
    assert "Vector Context 2" not in context_cliff1
    assert "Vector Context 3" not in context_cliff1
    assert "Graph Context 1" in context_cliff1
    assert "Graph Context 2" in context_cliff1
    
    # 验证排序与输出行数
    lines = [line for line in context_cliff1.split("\n\n") if line.strip()]
    assert len(lines) == 3
    assert "Vector Context 1" in lines[0]
    assert "Graph Context" in lines[1]
    assert "Graph Context" in lines[2]

    # 2. 首位分数 < 0.5 时的熔断测试
    # 设分数为 V_1=0.49, V_2=0.48, V_3=0.47, V_4=0.46。不触发向量得分差断崖，但由于首位分数 0.49 < 0.5 触发熔断。
    # 应只返回 V_1, V_2, V_3
    score_map_fuse = {
        "V_1": 0.49,
        "V_2": 0.48,
        "V_3": 0.47,
        "V_4": 0.46,
        "G_1": 0.9,
        "G_2": 0.9
    }
    reranker_fuse = MockRerankerService(score_map_fuse)
    retriever_fuse = GraphPostRetriever(emb, db, reranker_fuse)
    context_fuse = retriever_fuse.query_graph_enhanced("test", graph_search_mode="heuristic_walk")
    assert "Vector Context 1" in context_fuse
    assert "Vector Context 2" in context_fuse
    assert "Vector Context 3" in context_fuse
    assert "Graph Context" not in context_fuse

    # 3 & 4. 验证全局重排的差值过滤代替旧比例断崖
    # 重新定义 Mock 结构，以便更精准地控制相似度
    class PreciseEmbeddingService:
        def get_dense_embedding(self, query):
            return [1.0, 0.0]
        def get_sparse_embedding(self, query):
            return {}

    class PreciseDBAdapter:
        def __init__(self):
            self.graph = nx.Graph()
            # V_1, V_2, V_3 作为向量通道
            self.graph.add_node("V_1", parent_text="V1 Text", filename="doc.txt", source_path="", embedding=[1.0, 0.0])
            self.graph.add_node("V_2", parent_text="V2 Text", filename="doc.txt", source_path="", embedding=[1.0, 0.0])
            self.graph.add_node("V_3", parent_text="V3 Text", filename="doc.txt", source_path="", embedding=[1.0, 0.0])
            
            # G_1, G_2 作为图谱邻居
            # G_1 的相似度为 1.0
            self.graph.add_node("G_1", parent_text="G1 Text", filename="doc.txt", source_path="", embedding=[1.0, 0.0])
            # G_2 的相似度为 0.3
            self.graph.add_node("G_2", parent_text="G2 Text", filename="doc.txt", source_path="", embedding=[0.3, 0.9539])
            
            self.graph.add_edge("V_1", "G_1")
            self.graph.add_edge("V_1", "G_2")

        def hybrid_search(self, dense_vec, sparse_vec, top_k):
            return [
                {"metadata": {"parent_id": "V_1", "filename": "doc.txt", "parent_text": "V1 Text"}, "content": "V1 Text"},
                {"metadata": {"parent_id": "V_2", "filename": "doc.txt", "parent_text": "V2 Text"}, "content": "V2 Text"},
                {"metadata": {"parent_id": "V_3", "filename": "doc.txt", "parent_text": "V3 Text"}, "content": "V3 Text"},
            ]

    precise_emb = PreciseEmbeddingService()
    precise_db = PreciseDBAdapter()
    
    # 设二次 Rerank 分数：V_1=3.0, G_1=2.9, V_2=2.8, V_3=2.6, G_2=0.5 (无前4断崖，G_2低分被1.5差值断崖过滤)
    # 二次 Rerank 降序: V_1(3.0), G_1(2.9), V_2(2.8), V_3(2.6), G_2(0.5)
    # 2.6 - 0.5 = 2.1 > 1.5，触发差值断崖，从 G_2 截断，保留 V_1, G_1, V_2, V_3
    score_map_cliff_g = {
        "V_1": 3.0,
        "V_2": 2.8,
        "V_3": 2.6,
        "G_1": 2.9,
        "G_2": 0.5
    }
    reranker_cliff_g = MockRerankerService(score_map_cliff_g)
    retriever_cliff_g = GraphPostRetriever(precise_emb, precise_db, reranker_cliff_g)
    context_cliff_g = retriever_cliff_g.query_graph_enhanced("test", graph_search_mode="heuristic_walk")
    
    assert "V1 Text" in context_cliff_g
    assert "G1 Text" in context_cliff_g
    assert "V2 Text" in context_cliff_g
    assert "V3 Text" in context_cliff_g
    assert "G2 Text" not in context_cliff_g
    
    # 5. 验证排序与输出行数
    parts = [p for p in context_cliff_g.split("\n\n") if p.strip()]
    assert len(parts) == 4
    assert "V1 Text" in parts[0]
    assert "G1 Text" in parts[1]
    assert "V2 Text" in parts[2]
    assert "V3 Text" in parts[3]


def test_weighted_subgraph_pagerank():
    g = nx.Graph()
    # 节点
    g.add_node("seed", embedding=[1.0, 0.0])
    g.add_node("strong_1hop", embedding=[0.9, 0.1])
    g.add_node("weak_1hop", embedding=[0.1, 0.9])
    
    # 边，一个强，一个弱
    g.add_edge("seed", "strong_1hop", weight=10.0)
    g.add_edge("seed", "weak_1hop", weight=2.0)
    
    # 从 seed 出发进行带权剪枝 PPR
    res = run_personalized_pagerank(g, "seed", top_k=5)
    
    # 转换为字典以验证得分大小
    scores = dict(res)
    assert "strong_1hop" in scores
    assert "weak_1hop" in scores
    # 强权重的边应该传递更多的 PageRank 能量，因而得分更高
    assert scores["strong_1hop"] > scores["weak_1hop"]


