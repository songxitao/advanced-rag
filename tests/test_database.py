import pytest
from src.database import ChromaAdapter
from pathlib import Path

def test_chroma_adapter(tmp_path):
    db_dir = tmp_path / "test_chroma_db"
    adapter = ChromaAdapter(db_dir=str(db_dir))
    
    chunks = [
        {
            "child_text": "早稻田大学信息生产系统工程系",
            "parent_text": "早稻田大学位于日本，信息生产系统工程系招收硕士研究生。",
            "parent_id": "p1",
            "source_path": "test.txt",
            "filename": "test.txt",
            "char_start": 0,
            "char_end": 100
        }
    ]
    
    # 模拟一个 1024 维的 Dense 向量和稀疏词汇字典
    dense_vec = [0.1] * 1024
    sparse_vec = {"早稻田": 1.0, "硕士": 1.0}
    
    # 写入数据
    adapter.add_chunks(chunks, [dense_vec])
    
    # 检索数据
    results = adapter.hybrid_search(dense_vec, sparse_vec, top_k=1)
    
    assert len(results) > 0
    assert results[0]["metadata"]["parent_id"] == "p1"
    assert results[0]["metadata"]["filename"] == "test.txt"
    assert results[0]["content"] == "早稻田大学信息生产系统工程系"
