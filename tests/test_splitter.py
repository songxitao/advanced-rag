import pytest
from src.splitter import SemanticParentChildSplitter

class MockEmbeddingService:
    def get_dense_embedding(self, text: str):
        # 根据文本内容的不同，返回完全不同的两组向量
        if "第一部分" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

def test_semantic_parent_child_splitter():
    mock_embedding = MockEmbeddingService()
    # 设定较小的 child_size 以测试切割
    splitter = SemanticParentChildSplitter(embedding_service=mock_embedding, threshold=0.5, child_size=30)
    
    text = "第一部分。这一句在聊第一部分的内容。换行\n第二部分。第二句开始了新的概念。"
    chunks = splitter.create_parent_child_chunks(text)
    
    assert len(chunks) > 0
    assert "child_text" in chunks[0]
    assert "parent_text" in chunks[0]
    assert "parent_id" in chunks[0]
    assert chunks[0]["child_text"] != ""

def test_empty_input():
    mock_embedding = MockEmbeddingService()
    splitter = SemanticParentChildSplitter(embedding_service=mock_embedding)
    assert splitter.create_parent_child_chunks("") == []
    assert splitter.create_parent_child_chunks(None) == []

def test_no_punctuation_truncation():
    mock_embedding = MockEmbeddingService()
    text = "这绝对是一个没有任何标点符号的长字符串它需要被暴力截断因为在它的字数范围内没有任何符号"
    splitter = SemanticParentChildSplitter(embedding_service=mock_embedding, child_size=10)
    chunks = splitter.create_parent_child_chunks(text)
    
    assert len(chunks) == 5
    assert chunks[0]["child_text"] == "这绝对是一个没有任何"
    assert chunks[1]["child_text"] == "标点符号的长字符串它"

def test_dynamic_threshold():
    mock_embedding = MockEmbeddingService()
    splitter = SemanticParentChildSplitter(embedding_service=mock_embedding, threshold=None, child_size=100)
    text = "第一部分。第一部分。第一部分。第一部分。第二部分。"
    chunks = splitter.create_parent_child_chunks(text)
    
    parents = list(set(chunk["parent_text"] for chunk in chunks))
    assert len(parents) == 2
    assert "第一部分。第一部分。第一部分。第一部分。" in parents
    assert "第二部分。" in parents
