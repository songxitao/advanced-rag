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
    # 种子节点 p1 必须被排除在外
    assert "p1" not in res
    # 距离 p1 较近的 p2 应排在前面
    assert res[0] == "p2"

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
    assert "p1" not in res
    assert "p2" in res
    assert "p3" in res

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

    # 3. 验证断崖截断
    # 物理邻居 2跳为“吃苹果”。我们通过 Mock Reranker 强行降低“吃苹果”的得分至 -2.0，
    # 而将特工/早稻田分值设为 1.0。这样落差为 1.0 - (-2.0) = 3.0 > 1.5，必定触发截断。
    original_rerank = retriever.reranker.rerank
    def mock_rerank(query, candidates, top_k):
        print("\n--- DEBUG mock_rerank call ---")
        for idx, c in enumerate(candidates):
            print(f"candidate {idx}: parent_id: {c['metadata']['parent_id']}, parent_text: {c['metadata']['parent_text']}, content: {c['content']}")
        results = original_rerank(query, candidates, top_k)
        for r in results:
            if "苹果" in r["content"] or "苹果" in r.get("metadata", {}).get("parent_text", ""):
                r["rerank_score"] = -2.0
            else:
                r["rerank_score"] = 1.0
        print("mock_rerank results:")
        for r in results:
            print("score:", r["rerank_score"], "content:", r["content"])
        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        
    retriever.reranker.rerank = mock_rerank

    context_cliff = retriever.query_graph_enhanced("特工A01的图纸在哪里？", graph_search_mode="heuristic_walk")
    print("\n--- DEBUG context_cliff ---")
    print(context_cliff)
    # 特工A01依然存在
    assert "特工A01" in context_cliff
    # 吃苹果段落被断崖截断过滤
    assert "苹果" not in context_cliff



