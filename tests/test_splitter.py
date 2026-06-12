import pytest
from src.splitter import SemanticParentChildSplitter

class MockEmbeddingService:
    def __init__(self):
        self.batch_called = False

    def get_dense_embedding(self, text: str):
        # 根据文本内容的不同，返回完全不同的两组向量
        if "第一部分" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    def get_dense_embeddings_batch(self, texts: list[str]):
        self.batch_called = True
        res = []
        for text in texts:
            if "第一部分" in text:
                res.append([1.0, 0.0, 0.0])
            else:
                res.append([0.0, 1.0, 0.0])
        return res

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

def test_splitter_uses_batch_embeddings():
    mock_embedding = MockEmbeddingService()
    splitter = SemanticParentChildSplitter(embedding_service=mock_embedding, threshold=0.5, child_size=30)
    text = "第一部分。这一句在聊第一部分的内容。换行\n第二部分。第二句开始了新的概念。"
    chunks = splitter.create_parent_child_chunks(text)
    
    assert mock_embedding.batch_called is True

def test_markdown_splitter_specialized():
    mock_embedding = MockEmbeddingService()
    splitter = SemanticParentChildSplitter(embedding_service=mock_embedding, threshold=0.5, child_size=150)
    
    markdown_text = """---
title: 测试Markdown
date: 2026-06-13
---

# 这是一个标题

- 列表第1项
- 列表第2项

这是一个普通段落，包含了连续的两句话。这里是第二句话。

```python
def test():
    # 代码块内部不应该被拆分
    print("hello world!")
```

这是代码块后面的普通结尾。
"""
    
    chunks = splitter.create_parent_child_chunks(markdown_text, is_markdown=True)
    
    assert len(chunks) > 0
    code_found = False
    for c in chunks:
        parent = c["parent_text"]
        if "def test():" in parent:
            code_found = True
            assert 'print("hello world!")' in parent
            
    assert code_found is True


def test_markdown_ast_blocks_extraction():
    mock_embedding = MockEmbeddingService()
    # 设定 child_size = 50
    splitter = SemanticParentChildSplitter(embedding_service=mock_embedding, threshold=0.5, child_size=50)
    
    markdown_text = """# 标题

这是一个段落。

| Name | Age | Role |
|:---|:---:|---:|
| Alice | 24 | Engineer |
| Bob | 30 | Designer |

下面是代码块。

```python
def my_complex_function(x, y):
    res = x + y
    return res
```

结束。
"""
    chunks = splitter.create_parent_child_chunks(markdown_text, is_markdown=True)
    
    assert len(chunks) > 0
    
    # 验证表格作为整体保留，没有被切碎
    table_found = False
    for chunk in chunks:
        child = chunk["child_text"]
        if "Alice" in child:
            table_found = True
            # 表格的结构应该被完整保留在同一个 child chunk 内
            assert "Bob" in child
            assert "Name" in child
            assert "|:---" in child or "| :---" in child or "| ---" in child
            
    assert table_found is True

    # 验证代码块作为整体保留，没有被切碎
    code_found = False
    for chunk in chunks:
        child = chunk["child_text"]
        if "my_complex_function" in child:
            code_found = True
            # 代码块的结构应该完整保留在同一个 child chunk 内
            assert "return res" in child
            assert "def my_complex_function" in child
            assert "```python" in child
            
    assert code_found is True


def test_markdown_parent_size_guarantee():
    mock_embedding = MockEmbeddingService()
    splitter = SemanticParentChildSplitter(embedding_service=mock_embedding, threshold=0.5, child_size=100)
    
    # 这里的文本很短，总长度 < 150 字符
    markdown_text = "# 第一部分\n\n这是第二部分内容。"
    
    chunks = splitter.create_parent_child_chunks(markdown_text, is_markdown=True)
    parents = list(set(chunk["parent_text"] for chunk in chunks))
    
    # 断言超短 parent 块会被合并，使得最后只有一个 parent chunk
    assert len(parents) == 1

