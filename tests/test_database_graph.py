import pytest
import numpy as np
from src.database import ChromaAdapter

def test_database_graph_linking(tmp_path):
    db_dir = tmp_path / "test_chroma_db"
    adapter = ChromaAdapter(db_dir=str(db_dir))
    
    # 模拟 5 个 parent 块以实现三轨独立连边且互不干扰覆盖：
    # p1, p2, p3, p4 在 test_doc1.txt 中，按 char_start 排序为 p1, p2, p3, p4
    # p5 在 test_doc2.txt 中
    #
    # 轨 1：物理相邻边 (p1, p2), (p2, p3), (p3, p4) 将建立 physical 类型的边
    # 轨 2：实体共现边 (p1, p3) 在同一个文档内且共享特征词 "特工A01" (不与相邻或语义重合)
    # 轨 3：语义关联边
    #    - 局域 (p1, p4) 在同一个文档内且余弦相似度 >= 0.82
    #    - 跨文档 (p1, p5) 在不同文档且余弦相似度 >= 0.85 (利用 ANN 优化)
    
    chunks = [
        {
            "child_text": "特工A01正在执行秘密任务。",
            "parent_text": "特工A01正在执行秘密任务。特工A01的代号是机密。",
            "parent_id": "p1",
            "source_path": "test_doc1.txt",
            "filename": "test_doc1.txt",
            "char_start": 0,
            "char_end": 100
        },
        {
            "child_text": "早稻田大学信息生产系统工程系招收硕士研究生。",
            "parent_text": "早稻田大学信息生产系统工程系招收硕士研究生。",
            "parent_id": "p2",
            "source_path": "test_doc1.txt",
            "filename": "test_doc1.txt",
            "char_start": 100,
            "char_end": 200
        },
        {
            "child_text": "特工A01已经与总部取得了联系。",
            "parent_text": "特工A01已经与总部取得了联系。总部对特工A01的安全表示关注。",
            "parent_id": "p3",
            "source_path": "test_doc1.txt",
            "filename": "test_doc1.txt",
            "char_start": 200,
            "char_end": 300
        },
        {
            "child_text": "特工C03正在执行C计划。",
            "parent_text": "特工C03正在执行C计划。特工C03已经就位。",
            "parent_id": "p4",
            "source_path": "test_doc1.txt",
            "filename": "test_doc1.txt",
            "char_start": 300,
            "char_end": 400
        },
        {
            "child_text": "特工D04的秘密行动开始。",
            "parent_text": "特工D04的秘密行动开始。这是一个需要高度警惕的任务。",
            "parent_id": "p5",
            "source_path": "test_doc2.txt",
            "filename": "test_doc2.txt",
            "char_start": 0,
            "char_end": 100
        }
    ]
    
    # 语义向量设置
    emb_p1 = [1.0] + [0.0] * 1023
    emb_p2 = [0.0, 1.0] + [0.0] * 1022
    emb_p3 = [0.0, 0.0, 1.0] + [0.0] * 1021
    emb_p4 = [0.9, 0.1] + [0.0] * 1022
    emb_p5 = [0.95, 0.05] + [0.0] * 1022
    
    dense_embeddings = [emb_p1, emb_p2, emb_p3, emb_p4, emb_p5]
    
    # 写入数据并触发连边重建
    adapter.add_chunks(chunks, dense_embeddings)
    
    # 验证图的节点是否正确加入
    assert "p1" in adapter.graph
    assert "p2" in adapter.graph
    assert "p3" in adapter.graph
    assert "p4" in adapter.graph
    assert "p5" in adapter.graph
    
    # 验证物理相邻边 (p1-p2, p2-p3, p3-p4)
    # p1 和 p2 相邻且不共享高相似度和实体关键字，所以连边类型仅为 physical
    assert adapter.graph.has_edge("p1", "p2")
    assert adapter.graph.get_edge_data("p1", "p2")["type"] == "physical"
    
    # 验证无监督实体共现边 (p1, p3)
    # p1 和 p3 不相邻且余弦相似度极低(0)，但共享名词 "特工A01"
    assert adapter.graph.has_edge("p1", "p3")
    assert adapter.graph.get_edge_data("p1", "p3")["type"] == "entity"
    
    # 验证局域语义关联边 (p1, p4)
    # p1 和 p4 在同一个文档，不相邻，且余弦相似度为 0.9 / sqrt(0.82) ≈ 0.99 >= 0.82
    assert adapter.graph.has_edge("p1", "p4")
    assert adapter.graph.get_edge_data("p1", "p4")["type"] == "semantic"
    
    # 验证跨文档语义关联边 (p1, p5)
    # p1 在 test_doc1.txt，p5 在 test_doc2.txt，余弦相似度 ≈ 0.998 >= 0.85
    assert adapter.graph.get_edge_data("p1", "p5")["type"] == "semantic"

def test_database_graph_empty_and_robustness(tmp_path):
    db_dir = tmp_path / "test_chroma_db_robust"
    adapter = ChromaAdapter(db_dir=str(db_dir))
    
    # 1. 库为空时 rebuild_graph 应安全返回
    adapter.rebuild_graph()
    assert len(adapter.graph.nodes) == 0
    
    # 2. 库为空时 _rebuild_bm25 应安全返回
    adapter._rebuild_bm25()
    assert adapter.bm25 is None
    
    # 3. 传入空 chunks 时 add_chunks 应安全处理
    adapter.add_chunks([], [])
    assert len(adapter.graph.nodes) == 0


def test_graph_edge_weights_and_idf():
    import shutil
    import tempfile
    import os
    from src.loader import DocumentLoader
    from src.splitter import SemanticParentChildSplitter
    from src.embedding import LocalEmbeddingService
    from src.database import ChromaAdapter
    from src.coordinator import RAGCoordinator

    tmpdir = tempfile.mkdtemp()
    try:
        loader = DocumentLoader()
        emb = LocalEmbeddingService(device="cpu")
        splitter = SemanticParentChildSplitter(embedding_service=emb, threshold=2.0, child_size=30, min_parent_size=5)
        db = ChromaAdapter(db_dir=tmpdir)
        coordinator = RAGCoordinator(loader, splitter, emb, db, None)

        # 写入包含高频词"刘备"和罕见词"督邮"的测试文档
        # 块1：刘备 督邮 怒鞭
        # 块2：刘备 督邮 刁难
        # 块3：刘备 娶妻 孙尚香
        test_file = os.path.join(tmpdir, "test_weight_doc.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("角色 刘备 和 角色 督邮 在这里，怒鞭督邮。\n\n"
                    "角色 刘备 被 角色 督邮 刁难了。\n\n"
                    "角色 刘备 娶了 角色 孙尚香。")
        
        coordinator.add_file(test_file)
        graph = db.graph
        
        # 验证实体边权重
        # 寻找仅通过常见词"刘备"相连的实体边（块3和块1之间仅有"刘备"）
        # 寻找通过罕见词"督邮"相连的实体边（块1和块2之间有"督邮"和"刘备"）
        node_1_id = None
        node_2_id = None
        node_3_id = None
        for node_id, data in graph.nodes(data=True):
            text = data.get("parent_text", "")
            if "怒鞭督邮" in text:
                node_1_id = node_id
            elif "刁难" in text:
                node_2_id = node_id
            elif "娶了" in text:
                node_3_id = node_id

        assert node_1_id is not None
        assert node_2_id is not None
        assert node_3_id is not None

        # 块1与块2包含"督邮"（罕见），块1与块3仅包含"刘备"（高频）
        edge_1_2 = graph.get_edge_data(node_1_id, node_2_id)
        edge_1_3 = graph.get_edge_data(node_1_id, node_3_id)

        # 必须包含带有指数拉伸的 weight 属性
        assert "weight" in edge_1_2
        assert "weight" in edge_1_3
        assert edge_1_2["weight"] > edge_1_3["weight"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


