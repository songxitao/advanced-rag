import pytest
import shutil
from pathlib import Path
from src.coordinator import RAGCoordinator
from src.loader import DocumentLoader
from src.splitter import SemanticParentChildSplitter
from src.embedding import LocalEmbeddingService
from src.database import ChromaAdapter
from src.reranker import RerankerService

@pytest.mark.slow
def test_rag_pipeline_integration(tmp_path):
    db_dir = tmp_path / "test_integration_db"
    loader = DocumentLoader()
    
    # 初始化 Embedding，为测试快速运行，在构造函数中指定设备为 cpu
    emb = LocalEmbeddingService(device="cpu")
    splitter = SemanticParentChildSplitter(embedding_service=emb, threshold=0.5, child_size=30)
    db = ChromaAdapter(db_dir=str(db_dir))
    
    # 初始化 Reranker，为测试快速运行，在构造函数中指定设备为 cpu
    reranker = RerankerService(device="cpu")
    
    coordinator = RAGCoordinator(loader, splitter, emb, db, reranker)
    
    # 建立临时测试文件
    test_file = tmp_path / "test_doc.txt"
    # 确保两句话有较弱的语义关联，但我们只想根据提问召回最相关的一句话对应的大段落
    test_file.write_text("东南大学本科的主修课程有图像处理。早稻田大学IPS的硕士课程有机器学习与大数据。", encoding="utf-8")
    
    # 文件入库
    coordinator.add_file(str(test_file))
    
    # 检索
    context = coordinator.query("早稻田大学IPS有些什么课程？")
    
    # 验证返回的结果是完整的父块段落，且追加了 (来源: test_doc.txt) 标记
    assert "机器学习" in context
    assert "(来源: test_doc.txt)" in context
