import pytest
import numpy as np
from src.database import ChromaAdapter

def test_database_graph_linking(tmp_path):
    db_dir = tmp_path / "test_chroma_db"
    adapter = ChromaAdapter(db_dir=str(db_dir))
    
    # 模拟 6 个 parent 块以实现重构后的连边验证：
    # p1, p2, p3, p4 在 test_doc1.txt 中，按 char_start 排序为 p1, p2, p3, p4
    # p5 在 test_doc2.txt 中
    # p6 在 test_doc2.txt 中，包含核心词 "特工A01" 用于测试跨文档实体边
    #
    # 轨 1：物理相邻边 (p1, p2), (p2, p3), (p3, p4) 将建立 physical 类型的边
    # 轨 2：跨文档实体共现边 (p1, p6) 因为它们共享名词 "特工A01"，即使在不同文档也会建立 entity 连边
    # 轨 3：原语义相似边已废除，不应建立任何 semantic 类型的边
    
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
        },
        {
            "child_text": "特工A01遇到了新的敌人。",
            "parent_text": "特工A01遇到了新的敌人。情况对特工A01不太妙。",
            "parent_id": "p6",
            "source_path": "test_doc2.txt",
            "filename": "test_doc2.txt",
            "char_start": 100,
            "char_end": 200
        }
    ]
    
    # 语义向量设置
    emb_p1 = [1.0] + [0.0] * 1023
    emb_p2 = [0.0, 1.0] + [0.0] * 1022
    emb_p3 = [0.0, 0.0, 1.0] + [0.0] * 1021
    emb_p4 = [0.9, 0.1] + [0.0] * 1022
    emb_p5 = [0.95, 0.05] + [0.0] * 1022
    emb_p6 = [0.8, 0.2] + [0.0] * 1022
    
    dense_embeddings = [emb_p1, emb_p2, emb_p3, emb_p4, emb_p5, emb_p6]
    
    # 写入数据并触发连边重建
    adapter.add_chunks(chunks, dense_embeddings)
    
    # 验证图的节点是否正确加入
    assert "p1" in adapter.graph
    assert "p2" in adapter.graph
    assert "p3" in adapter.graph
    assert "p4" in adapter.graph
    assert "p5" in adapter.graph
    assert "p6" in adapter.graph
    
    # 验证物理相邻边 (p1-p2, p2-p3, p3-p4)
    assert adapter.graph.has_edge("p1", "p2")
    assert adapter.graph.get_edge_data("p1", "p2")["type"] == "physical"
    
    # 验证同文档实体边 (p1, p3)
    assert adapter.graph.has_edge("p1", "p3")
    assert adapter.graph.get_edge_data("p1", "p3")["type"] == "entity"
    
    # 验证跨文档实体边 (p1, p6)
    # p1 在 test_doc1.txt，p6 在 test_doc2.txt，但它们共享 "特工A01"
    assert adapter.graph.has_edge("p1", "p6")
    assert adapter.graph.get_edge_data("p1", "p6")["type"] == "entity"
    
    # 验证语义边已完全被废除
    # 以前 p1-p4 会有 semantic 边，现在应该没有了
    if adapter.graph.has_edge("p1", "p4"):
        assert adapter.graph.get_edge_data("p1", "p4")["type"] != "semantic"
    if adapter.graph.has_edge("p1", "p5"):
        assert adapter.graph.get_edge_data("p1", "p5")["type"] != "semantic"

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


@pytest.mark.slow
def test_graph_edge_weights_and_idf():
    import shutil
    import tempfile
    import os
    import jieba
    from src.loader import DocumentLoader
    from src.splitter import SemanticParentChildSplitter
    from src.embedding import LocalEmbeddingService
    from src.database import ChromaAdapter
    from src.coordinator import RAGCoordinator

    # 注册测试涉及的中文实体名词，防止被 jieba 切分为单字而被长度过滤机制忽略
    jieba.add_word("刘备", tag="nr")
    jieba.add_word("督邮", tag="nz")
    jieba.add_word("关羽", tag="nr")
    jieba.add_word("张飞", tag="nr")

    tmpdir = tempfile.mkdtemp()
    try:
        loader = DocumentLoader()
        emb = LocalEmbeddingService(device="cpu")
        splitter = SemanticParentChildSplitter(embedding_service=emb, threshold=2.0, child_size=30, min_parent_size=5)
        db = ChromaAdapter(db_dir=tmpdir)
        coordinator = RAGCoordinator(loader, splitter, emb, db, None)

        # 写入 10 个测试 Chunk，其中 6 个 Chunk 包含高频词 "刘备" (超过 20% 节点占比)
        # 2 个 Chunk 包含稀缺词 "督邮"
        test_file = os.path.join(tmpdir, "test_weight_doc.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(
                "角色 刘备 一号出现。\n\n"
                "角色 刘备 二号出现。\n\n"
                "角色 刘备 三号出现。\n\n"
                "角色 刘备 四号出现。\n\n"
                "角色 刘备 五号出现。\n\n"
                "角色 刘备 六号出现，且这里有 角色 督邮。\n\n"
                "角色 督邮 正在处理公务。\n\n"
                "其他角色 关羽 正在看书。\n\n"
                "其他角色 张飞 正在喝酒。\n\n"
                "其他角色 赵云 正在练武。"
            )
        
        coordinator.add_file(test_file)
        graph = db.graph
        
        # 查找对应节点
        node_liubei_ids = []
        node_duyou_ids = []
        for node_id, data in graph.nodes(data=True):
            text = data.get("parent_text", "")
            if "刘备" in text:
                node_liubei_ids.append(node_id)
            if "督邮" in text:
                node_duyou_ids.append(node_id)

        # 包含"刘备"的节点超过全局 20% (6/10 = 60%)，应触发 Hub 剔除，它们之间绝不应建立基于"刘备"的 entity 边
        assert len(node_liubei_ids) >= 6
        for i in range(len(node_liubei_ids)):
            for j in range(i + 1, len(node_liubei_ids)):
                u = node_liubei_ids[i]
                v = node_liubei_ids[j]
                if graph.has_edge(u, v):
                    assert graph.get_edge_data(u, v)["type"] != "entity"

        # 包含"督邮"的节点数未超标 (2/10 = 20%，且未超 5 个节点的 Hub 豁免下限)
        # 它们之间应当通过实体 "督邮" 建立连边
        assert len(node_duyou_ids) == 2
        assert graph.has_edge(node_duyou_ids[0], node_duyou_ids[1])
        assert graph.get_edge_data(node_duyou_ids[0], node_duyou_ids[1])["type"] == "entity"
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


