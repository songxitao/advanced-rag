# Task 4: 抽象数据库适配器与 ChromaDB / BM25 混合检索实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 开发抽象数据库适配器与基于 ChromaDB 和 BM25 的双通道混合检索实现（database.py），并跑通单元测试。

**Architecture:** 定义 `VectorStoreAdapter` 抽象基类，并实现继承它的 `ChromaAdapter`。使用 `chromadb` 持久化向量并进行余弦空间 Dense 检索，使用 `rank_bm25` (基于 `jieba` 分词后的语料库) 进行本地内存的 Sparse 检索，在 `hybrid_search` 方法中进行双路召回、合并并根据 `parent_id` 去重。

**Tech Stack:** Python 3.10+, chromadb, rank_bm25, jieba, pytest

---

### Task 4.1: 依赖环境安装与环境确认

**Files:**
- Modify: 无

- [ ] **Step 1: 安装依赖库**
  
  运行：`cmd /c 'call "D:\program files\Miniconda\Scripts\activate.bat" & conda activate deepseek-ocr & pip install rank_bm25 jieba chromadb'`
  预期：依赖库顺利安装完成。

---

### Task 4.2: 编写测试用例并确认失败

**Files:**
- Create: `E:/project/advanced-rag/tests/test_database.py`

- [ ] **Step 1: 编写测试文件 `tests/test_database.py`**
  
  代码内容：
  ```python
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
  ```

- [ ] **Step 2: 运行测试以确信其失败**
  
  运行：`cmd /c 'call "D:\program files\Miniconda\Scripts\activate.bat" & conda activate deepseek-ocr & pytest E:/project/advanced-rag/tests/test_database.py -v'`
  预期：FAIL，提示 `ModuleNotFoundError: No module named 'src.database'`。

---

### Task 4.3: 编写数据库适配器与混合检索逻辑

**Files:**
- Create: `E:/project/advanced-rag/src/database.py`

- [ ] **Step 1: 编写 `src/database.py` 代码**
  
  代码内容：
  ```python
  from abc import ABC, abstractmethod
  from typing import List, Dict, Any
  import chromadb
  from rank_bm25 import BM25Okapi
  import jieba

  class VectorStoreAdapter(ABC):
      @abstractmethod
      def add_chunks(self, chunks_data: List[Dict[str, Any]], dense_embeddings: List[List[float]]) -> None:
          pass

      @abstractmethod
      def hybrid_search(self, dense_vec: List[float], sparse_vec: Dict[str, float], top_k: int) -> List[Dict[str, Any]]:
          pass

  class ChromaAdapter(VectorStoreAdapter):
      def __init__(self, db_dir: str = "./vector_db", collection_name: str = "advanced_rag_collection"):
          self.client = chromadb.PersistentClient(path=db_dir)
          self.collection = self.client.get_or_create_collection(
              name=collection_name,
              metadata={"hnsw:space": "cosine"}
          )
          self.bm25 = None
          self.bm25_docs = []  # 备份内存倒排索引所对应的数据列表，每个元素包含 {"document": str, "metadata": dict}
          self._rebuild_bm25()

      def _rebuild_bm25(self) -> None:
          """从已存的 Chroma 中提取文本和元数据，重建 BM25 词频索引"""
          count = self.collection.count()
          self.bm25_docs = []
          if count > 0:
              all_data = self.collection.get(include=["documents", "metadatas"])
              corpus = []
              for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                  self.bm25_docs.append({"document": doc, "metadata": meta})
                  # 使用 jieba 对文档进行分词并转为 list
                  words = list(jieba.cut(doc))
                  corpus.append(words)
              self.bm25 = BM25Okapi(corpus)
          else:
              self.bm25 = None

      def add_chunks(self, chunks_data: List[Dict[str, Any]], dense_embeddings: List[List[float]]) -> None:
          """在 Chroma 里存储 child_text 作为 document，并同步重建 BM25"""
          ids = [f"{c['parent_id']}_{i}" for i, c in enumerate(chunks_data)]
          documents = [c["child_text"] for c in chunks_data]
          metadatas = []
          for c in chunks_data:
              metadatas.append({
                  "parent_id": c["parent_id"],
                  "parent_text": c["parent_text"],
                  "source_path": c["source_path"],
                  "filename": c["filename"],
                  "char_start": c["char_start"],
                  "char_end": c["char_end"]
              })
          
          self.collection.add(
              ids=ids,
              embeddings=dense_embeddings,
              documents=documents,
              metadatas=metadatas
          )
          self._rebuild_bm25()

      def hybrid_search(self, dense_vec: List[float], sparse_vec: Dict[str, float], top_k: int) -> List[Dict[str, Any]]:
          """双通道混合检索与去重"""
          # 1. Dense 检索
          dense_candidates = []
          count = self.collection.count()
          if count > 0:
              n_results = min(top_k * 2, count)
              chroma_results = self.collection.query(
                  query_embeddings=[dense_vec],
                  n_results=n_results,
                  include=["documents", "metadatas", "distances"]
              )
              if chroma_results["documents"] and chroma_results["documents"][0]:
                  for i, doc in enumerate(chroma_results["documents"][0]):
                      dense_candidates.append({
                          "content": doc,
                          "metadata": chroma_results["metadatas"][0][i],
                          "score": 1.0 - chroma_results["distances"][0][i]  # cosine 余弦距离转化为相似度分数
                      })

          # 2. Sparse 检索
          sparse_candidates = []
          if self.bm25 and len(sparse_vec) > 0:
              query_tokens = list(sparse_vec.keys())
              scores = self.bm25.get_scores(query_tokens)
              
              # 按 BM25 分数排序，选出分数前 top_k * 2 的子块
              top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]
              for idx in top_indices:
                  # 只收集分数大于 0 的或者所有前几名的结果
                  if scores[idx] > 0:
                      doc_info = self.bm25_docs[idx]
                      sparse_candidates.append({
                          "content": doc_info["document"],
                          "metadata": doc_info["metadata"],
                          "score": float(scores[idx])
                      })

          # 3. 去重与合并：根据 parent_id 过滤去重，最终返回候选子块列表
          merged = {}
          # 先合并 Dense
          for c in dense_candidates:
              p_id = c["metadata"]["parent_id"]
              if p_id not in merged:
                  merged[p_id] = c
          # 再合并 Sparse，如已存在则忽略
          for c in sparse_candidates:
              p_id = c["metadata"]["parent_id"]
              if p_id not in merged:
                  merged[p_id] = c
                  
          # 最终返回一个去重后的候选子块列表
          return list(merged.values())
  ```

---

### Task 4.4: 运行并验证测试

**Files:**
- Modify: 无

- [ ] **Step 1: 运行单元测试**
  
  运行：`cmd /c 'call "D:\program files\Miniconda\Scripts\activate.bat" & conda activate deepseek-ocr & pytest E:/project/advanced-rag/tests/test_database.py -v'`
  预期：测试通过 (PASS)。

---

### Task 4.5: Git 提交代码

**Files:**
- Modify: 无

- [ ] **Step 1: 提交重构开发文件**
  
  运行：
  ```bash
  git add E:/project/advanced-rag/tests/test_database.py E:/project/advanced-rag/src/database.py
  git commit -m "feat: 实现抽象数据库适配器与ChromaDB/BM25混合检索(Task 4)并跑通测试"
  ```
  预期：代码成功提交。
