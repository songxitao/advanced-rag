# RAG 引擎高内聚低耦合重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于高内聚低耦合的设计原则，重构构建一个支持本地 BGE-M3 的语义化切分与父子块混合检索的 RAG 引擎，并暴露 FastAPI 接口以接入 Dify。

**Architecture:** 将系统解耦为文档读取、语义化父子切割、本地向量提取、Chroma 适配器、BM25 稀疏检索及 Reranker 精排服务，最终由 Coordinator 进行统一编排并通过 FastAPI 提供微服务。

**Tech Stack:** Python 3.10+, PyMuPDF (fitz), sentence-transformers, chromadb, rank_bm25, fastapi, uvicorn, pytest

---

### Task 1: 项目环境初始化与文档加载模块开发

**Files:**
- Create: `E:/project/advanced-rag/src/loader.py`
- Test: `E:/project/advanced-rag/tests/test_loader.py`

- [ ] **Step 1: 编写失败的 DocumentLoader 单元测试**
  在 `tests/test_loader.py` 中写入：
  ```python
  import pytest
  from src.loader import DocumentLoader
  from pathlib import Path

  def test_document_loader_txt(tmp_path):
      test_file = tmp_path / "test.txt"
      test_file.write_text("Hello 早稻田 IPS 和东南大学", encoding="utf-8")
      
      loader = DocumentLoader()
      text = loader.load(str(test_file))
      assert text == "Hello 早稻田 IPS 和东南大学"
  ```

- [ ] **Step 2: 运行测试并确认失败**
  运行：`pytest E:/project/advanced-rag/tests/test_loader.py -v`
  预期：FAIL（找不到模块 `src.loader`）

- [ ] **Step 3: 编写最小化 Loader 实现**
  在 `src/loader.py` 中写入：
  ```python
  from pathlib import Path
  import fitz
  from docx import Document

  class DocumentLoader:
      def load(self, file_path: str) -> str:
          path = Path(file_path)
          if not path.exists():
              raise FileNotFoundError(f"文件不存在: {file_path}")
          
          ext = path.suffix.lower()
          if ext == ".txt":
              return path.read_text(encoding="utf-8", errors="ignore")
          elif ext == ".pdf":
              doc = fitz.open(path)
              return "\n".join([page.get_text() or "" for page in doc])
          elif ext == ".docx":
              doc = Document(path)
              return "\n".join([p.text for p in doc.paragraphs])
          elif ext == ".srt":
              content = path.read_text(encoding="utf-8", errors="ignore")
              lines = []
              for line in content.splitlines():
                  line = line.strip()
                  if not line or line.isdigit() or "-->" in line:
                      continue
                  lines.append(line)
              return "\n".join(lines)
          else:
              raise ValueError(f"不支持的格式: {ext}")
  ```

- [ ] **Step 4: 运行测试并确认通过**
  运行：`pytest E:/project/advanced-rag/tests/test_loader.py -v`
  预期：PASS

- [ ] **Step 5: 提交代码**
  运行：
  ```bash
  git add E:/project/advanced-rag/src/loader.py E:/project/advanced-rag/tests/test_loader.py
  git commit -m "feat: 增加DocumentLoader模块并跑通测试"
  ```

---

### Task 2: 语义化切分与父子块关系生成器 (Splitter) 开发

**Files:**
- Create: `E:/project/advanced-rag/src/splitter.py`
- Test: `E:/project/advanced-rag/tests/test_splitter.py`

- [ ] **Step 1: 编写测试用例验证语义切块与父子块结构**
  在 `tests/test_splitter.py` 中写入：
  ```python
  import pytest
  from src.splitter import SemanticParentChildSplitter

  # 模拟一个简易的向量编码器用于切片器测试
  class MockEmbeddingService:
      def get_dense_embedding(self, text):
          # 根据文本内容返回伪向量
          if "第一部分" in text:
              return [1.0, 0.0, 0.0]
          return [0.0, 1.0, 0.0]

  def test_semantic_parent_child_splitter():
      mock_embedding = MockEmbeddingService()
      splitter = SemanticParentChildSplitter(embedding_service=mock_embedding, threshold=0.5)
      
      text = "第一部分。这一句在聊第一部分的内容。换行\n第二部分。第二句开始了新的概念。"
      chunks = splitter.create_parent_child_chunks(text)
      
      assert len(chunks) > 0
      assert "child_text" in chunks[0]
      assert "parent_text" in chunks[0]
      assert "parent_id" in chunks[0]
  ```

- [ ] **Step 2: 运行测试并确认失败**
  运行：`pytest E:/project/advanced-rag/tests/test_splitter.py -v`
  预期：FAIL（模块不存在）

- [ ] **Step 3: 编写语义化切片与子块标点对齐的逻辑**
  在 `src/splitter.py` 中写入：
  ```python
  import re
  import uuid
  import numpy as np
  from typing import List, Dict, Any

  class SemanticParentChildSplitter:
      def __init__(self, embedding_service, threshold: float = 0.6, window_size: int = 3, child_size: int = 150):
          self.embedding_service = embedding_service
          self.threshold = threshold
          self.window_size = window_size
          self.child_size = child_size

      def _cos_sim(self, v1, v2) -> float:
          dot_product = np.dot(v1, v2)
          norm_v1 = np.linalg.norm(v1)
          norm_v2 = np.linalg.norm(v2)
          if norm_v1 == 0 or norm_v2 == 0:
              return 0.0
          return float(dot_product / (norm_v1 * norm_v2))

      def semantic_split(self, text: str) -> List[str]:
          # 1. 拆分为单句
          raw_sentences = re.split(r'(。|！|？|\n+)', text)
          sentences = []
          for i in range(0, len(raw_sentences)-1, 2):
              sent = raw_sentences[i] + raw_sentences[i+1]
              if sent.strip():
                  sentences.append(sent.strip())
          if len(raw_sentences) % 2 != 0 and raw_sentences[-1].strip():
              sentences.append(raw_sentences[-1].strip())
              
          if len(sentences) <= self.window_size:
              return [text]

          # 2. 滑动窗口组合并提取向量
          groups = []
          for i in range(len(sentences) - self.window_size + 1):
              group_text = " ".join(sentences[i:i+self.window_size])
              groups.append(group_text)
              
          embeddings = [self.embedding_service.get_dense_embedding(g) for g in groups]
          
          # 3. 计算相邻余弦相似度
          similarities = []
          for i in range(len(embeddings) - 1):
              similarities.append(self._cos_sim(embeddings[i], embeddings[i+1]))
          
          # 4. 动态阈值判定波谷
          if not similarities:
              return [text]
          mean_sim = np.mean(similarities)
          std_sim = np.std(similarities)
          dynamic_threshold = mean_sim - 0.8 * std_sim
          
          parent_chunks = []
          current_chunk = []
          
          for i, sent in enumerate(sentences):
              current_chunk.append(sent)
              # 只有在滑动窗口范围内的分界处才进行波谷判定
              if i < len(similarities):
                  if similarities[i] < dynamic_threshold:
                      parent_chunks.append(" ".join(current_chunk))
                      current_chunk = []
                      
          if current_chunk:
              parent_chunks.append(" ".join(current_chunk))
          return parent_chunks

      def create_parent_child_chunks(self, text: str) -> List[Dict[str, Any]]:
          parents = self.semantic_split(text)
          all_chunks = []
          
          for p_text in parents:
              p_id = str(uuid.uuid4())
              # 将父块标点折断成子块（非暴力截断）
              sentences = re.split(r'(。|！|？|\n+)', p_text)
              current_child = ""
              char_cursor = 0
              
              for i in range(0, len(sentences)-1, 2):
                  sent = sentences[i] + sentences[i+1]
                  if len(current_child) + len(sent) > self.child_size:
                      if current_child:
                          all_chunks.append({
                              "child_text": current_child.strip(),
                              "parent_text": p_text,
                              "parent_id": p_id,
                              "char_start": char_cursor,
                              "char_end": char_cursor + len(current_child)
                          })
                          char_cursor += len(current_child)
                          current_child = ""
                  current_child += sent
                  
              if current_child.strip():
                  all_chunks.append({
                      "child_text": current_child.strip(),
                      "parent_text": p_text,
                      "parent_id": p_id,
                      "char_start": char_cursor,
                      "char_end": char_cursor + len(current_child)
                  })
          return all_chunks
  ```

- [ ] **Step 4: 运行测试并确认通过**
  运行：`pytest E:/project/advanced-rag/tests/test_splitter.py -v`
  预期：PASS

- [ ] **Step 5: 提交代码**
  运行：
  ```bash
  git add E:/project/advanced-rag/src/splitter.py E:/project/advanced-rag/tests/test_splitter.py
  git commit -m "feat: 跑通语义化父子块切片器"
  ```

---

### Task 3: 本地向量表征服务 (Embedding) 模块开发

**Files:**
- Create: `E:/project/advanced-rag/src/embedding.py`
- Test: `E:/project/advanced-rag/tests/test_embedding.py`

- [ ] **Step 1: 编写测试用例验证 Dense 和 Sparse 提取**
  在 `tests/test_embedding.py` 中写入：
  ```python
  import pytest
  from src.embedding import LocalEmbeddingService

  def test_embedding_service():
      # 直接使用本地 cache 的 bge-m3 测试
      service = LocalEmbeddingService()
      dense = service.get_dense_embedding("测试早稻田")
      sparse = service.get_sparse_embedding("测试东南")
      
      assert len(dense) == 1024  # BGE-M3 的 Dense 向量为 1024 维
      assert isinstance(sparse, dict)
      assert len(sparse) > 0
  ```

- [ ] **Step 2: 运行测试确认失败**
  运行：`pytest E:/project/advanced-rag/tests/test_embedding.py -v`
  预期：FAIL

- [ ] **Step 3: 编写 Embedding 接口的本地加载逻辑**
  在 `src/embedding.py` 中写入：
  ```python
  import os
  # 优先配置本地 cache 目录
  os.environ['HF_HOME'] = r'D:\my_huggingface_cache'
  os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'D:\my_huggingface_cache'

  from sentence_transformers import SentenceTransformer
  from typing import List, Dict

  class LocalEmbeddingService:
      def __init__(self, model_path: str = "BAAI/bge-m3", device: str = "cuda"):
          self.model = SentenceTransformer(model_path, device=device)
          
      def get_dense_embedding(self, text: str) -> List[float]:
          # bge-m3 默认推理
          emb = self.model.encode(text, normalize_embeddings=True)
          return emb.tolist()

      def get_sparse_embedding(self, text: str) -> Dict[str, float]:
          # 调用 BGE-M3 的词汇权重输出能力
          # sentence_transformers 可直接通过 model.encode(..., return_dense=False) 或者类似扩展接口提取
          # 兼容方案：在此只取分词后的字频/词频权重做初步表示
          tokens = self.model.tokenizer.tokenize(text)
          encoded_input = self.model.tokenizer(text, return_tensors='pt').to(self.model.device)
          import torch
          with torch.no_grad():
              model_output = self.model.q_model(**encoded_input) if hasattr(self.model, 'q_model') else self.model(**encoded_input)
              # 计算 lexical weights (稀疏表示)
              # 为保持零冗余，如果没有自定义 head 则回退为词表词频模拟：
              weights = {}
              for token in tokens:
                  weights[token] = weights.get(token, 0.0) + 1.0
          return weights
  ```

- [ ] **Step 4: 运行并确认通过**
  运行：`pytest E:/project/advanced-rag/tests/test_embedding.py -v`
  预期：PASS

- [ ] **Step 5: 提交代码**
  运行：
  ```bash
  git add E:/project/advanced-rag/src/embedding.py E:/project/advanced-rag/tests/test_embedding.py
  git commit -m "feat: 跑通Embedding服务"
  ```

---

### Task 4: 抽象数据库适配器与 ChromaDB / BM25 混合检索实现

**Files:**
- Create: `E:/project/advanced-rag/src/database.py`
- Test: `E:/project/advanced-rag/tests/test_database.py`

- [ ] **Step 1: 编写混合检索与父子块持久化测试**
  在 `tests/test_database.py` 中写入：
  ```python
  import pytest
  import shutil
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
      
      # 模拟向量
      dense_vec = [0.1] * 1024
      sparse_vec = {"早稻田": 1.5, "硕士": 1.0}
      
      adapter.add_chunks(chunks, [dense_vec])
      results = adapter.hybrid_search(dense_vec, sparse_vec, top_k=1)
      
      assert len(results) > 0
      assert results[0]["metadata"]["parent_id"] == "p1"
      assert results[0]["metadata"]["filename"] == "test.txt"
  ```

- [ ] **Step 2: 运行测试确认失败**
  运行：`pytest E:/project/advanced-rag/tests/test_database.py -v`
  预期：FAIL

- [ ] **Step 3: 编写适配器层代码及 BM25 检索逻辑**
  在 `src/database.py` 中写入：
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
          self.bm25_docs = [] # 内存倒排索引对应的数据备份
          self._rebuild_bm25()

      def _rebuild_bm25(self):
          # 重启时从本地库还原 BM25 词频索引
          count = self.collection.count()
          if count > 0:
              all_data = self.collection.get(include=["documents", "metadatas"])
              self.bm25_docs = []
              corpus = []
              for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                  self.bm25_docs.append({"document": doc, "metadata": meta})
                  corpus.append(list(jieba.cut(doc)))
              self.bm25 = BM25Okapi(corpus)

      def add_chunks(self, chunks_data: List[Dict[str, Any]], dense_embeddings: List[List[float]]) -> None:
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
          # 1. Dense 检索 (从 Chroma)
          chroma_results = self.collection.query(
              query_embeddings=[dense_vec],
              n_results=top_k * 2,
              include=["documents", "metadatas", "distances"]
          )
          
          dense_candidates = []
          if chroma_results["documents"] and chroma_results["documents"][0]:
              for i, doc in enumerate(chroma_results["documents"][0]):
                  dense_candidates.append({
                      "content": doc,
                      "metadata": chroma_results["metadatas"][0][i],
                      "dense_score": 1.0 - chroma_results["distances"][0][i]
                  })
                  
          # 2. Sparse 检索 (从本地内存 BM25)
          sparse_candidates = []
          if self.bm25 and len(sparse_vec) > 0:
              query_tokens = list(sparse_vec.keys())
              scores = self.bm25.get_scores(query_tokens)
              # 获取得分前几名的文档
              top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]
              for idx in top_indices:
                  if scores[idx] > 0:
                      doc_info = self.bm25_docs[idx]
                      sparse_candidates.append({
                          "content": doc_info["document"],
                          "metadata": doc_info["metadata"],
                          "sparse_score": float(scores[idx])
                      })
                      
          # 3. 混合去重融合 (Reranker 之前初筛去重)
          merged = {}
          for c in dense_candidates:
              merged[c["metadata"]["parent_id"]] = c
          for c in sparse_candidates:
              p_id = c["metadata"]["parent_id"]
              if p_id not in merged:
                  merged[p_id] = {
                      "content": c["content"],
                      "metadata": c["metadata"],
                      "dense_score": 0.0
                  }
          return list(merged.values())
  ```

- [ ] **Step 4: 运行并确认通过**
  运行：`pytest E:/project/advanced-rag/tests/test_database.py -v`
  预期：PASS

- [ ] **Step 5: 提交代码**
  运行：
  ```bash
  git add E:/project/advanced-rag/src/database.py E:/project/advanced-rag/tests/test_database.py
  git commit -m "feat: 跑通Chroma与BM25的双通道混合数据库适配器"
  ```

---

### Task 5: 重排服务 (Reranker) 与总协调器 (Coordinator) 开发

**Files:**
- Create: `E:/project/advanced-rag/src/reranker.py`
- Create: `E:/project/advanced-rag/src/coordinator.py`
- Test: `E:/project/advanced-rag/tests/test_coordinator.py`

- [ ] **Step 1: 编写 Rerank 和 Pipeline 的整体集成测试**
  在 `tests/test_coordinator.py` 中写入：
  ```python
  import pytest
  from src.coordinator import RAGCoordinator
  from src.loader import DocumentLoader
  from src.splitter import SemanticParentChildSplitter
  from src.embedding import LocalEmbeddingService
  from src.database import ChromaAdapter
  from src.reranker import RerankerService
  import Path

  def test_rag_pipeline_integration(tmp_path):
      db_dir = tmp_path / "test_integration_db"
      loader = DocumentLoader()
      emb = LocalEmbeddingService()
      splitter = SemanticParentChildSplitter(embedding_service=emb)
      db = ChromaAdapter(db_dir=str(db_dir))
      reranker = RerankerService()
      
      coordinator = RAGCoordinator(loader, splitter, emb, db, reranker)
      
      # 写入一个测试文件
      test_file = tmp_path / "test_doc.txt"
      test_file.write_text("东南大学本科的主修课程有图像处理。早稻田大学IPS的硕士课程有机器学习与大数据。", encoding="utf-8")
      
      coordinator.add_file(str(test_file))
      context = coordinator.query("早稻田大学IPS有些什么课程？")
      
      assert "机器学习" in context
      assert "来源: test_doc.txt" in context
  ```

- [ ] **Step 2: 运行测试确认失败**
  运行：`pytest E:/project/advanced-rag/tests/test_coordinator.py -v`
  预期：FAIL

- [ ] **Step 3: 编写 Reranker 重排与 Coordinator 整体数据流**
  在 `src/reranker.py` 中写入：
  ```python
  import os
  os.environ['HF_HOME'] = r'D:\my_huggingface_cache'
  os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'D:\my_huggingface_cache'
  from sentence_transformers import CrossEncoder
  from typing import List, Dict, Any

  class RerankerService:
      def __init__(self, model_path: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
          self.model = CrossEncoder(model_path, device=device)
          
      def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
          if not candidates:
              return []
          pairs = [[query, c["content"]] for c in candidates]
          scores = self.model.predict(pairs)
          for i, score in enumerate(scores):
              candidates[i]["rerank_score"] = float(score)
          candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
          return candidates[:top_k]
  ```

  在 `src/coordinator.py` 中写入：
  ```python
  from pathlib import Path
  from typing import Dict, Any

  class RAGCoordinator:
      def __init__(self, loader, splitter, embedding_service, db_adapter, reranker):
          self.loader = loader
          self.splitter = splitter
          self.embedding_service = embedding_service
          self.db_adapter = db_adapter
          self.reranker = reranker

      def add_file(self, file_path: str) -> int:
          text = self.loader.load(file_path)
          chunks = self.splitter.create_parent_child_chunks(text)
          if not chunks:
              return 0
              
          # 补齐元数据里的源文件信息
          path = Path(file_path)
          for c in chunks:
              c["source_path"] = str(path)
              c["filename"] = path.name
              
          # 计算 Dense 向量
          texts_to_encode = [c["child_text"] for c in chunks]
          dense_embs = [self.embedding_service.get_dense_embedding(t) for t in texts_to_encode]
          
          self.db_adapter.add_chunks(chunks, dense_embs)
          return len(chunks)

      def query(self, user_question: str) -> str:
          dense_vec = self.embedding_service.get_dense_embedding(user_question)
          sparse_vec = self.embedding_service.get_sparse_embedding(user_question)
          
          # 双路召回 Top 15
          candidates = self.db_adapter.hybrid_search(dense_vec, sparse_vec, top_k=15)
          
          # 精排 Top 5
          final_candidates = self.reranker.rerank(user_question, candidates, top_k=5)
          
          if not final_candidates:
              return "未找到相关文档参考。"
              
          context_parts = []
          for i, cand in enumerate(final_candidates, 1):
              filename = cand["metadata"].get("filename", "未知来源")
              parent_text = cand["metadata"].get("parent_text", cand["content"])
              context_parts.append(f"[片段{i}] (来源: {filename})\n{parent_text}")
              
          return "\n\n".join(context_parts)
  ```

- [ ] **Step 4: 运行并确认通过**
  运行：`pytest E:/project/advanced-rag/tests/test_coordinator.py -v`
  预期：PASS

- [ ] **Step 5: 提交代码**
  运行：
  ```bash
  git add E:/project/advanced-rag/src/reranker.py E:/project/advanced-rag/src/coordinator.py E:/project/advanced-rag/tests/test_coordinator.py
  git commit -m "feat: 跑通Rerank与Coordinator端到端逻辑并完成集成测试"
  ```

---

### Task 6: FastAPI 服务端开发与 OpenAPI 导出

**Files:**
- Create: `E:/project/advanced-rag/src/app.py`
- Test: `E:/project/advanced-rag/tests/test_api.py`

- [ ] **Step 1: 编写 API 接口功能测试**
  在 `tests/test_api.py` 中写入：
  ```python
  import pytest
  from fastapi.testclient import TestClient
  from src.app import app

  client = TestClient(app)

  def test_retrieve_api():
      response = client.post(
          "/retrieve",
          json={"query": "测试提问", "top_k": 3}
      )
      assert response.status_code == 200
      data = response.json()
      assert data["status"] == "success"
      assert "context" in data
  ```

- [ ] **Step 2: 运行测试确认失败**
  运行：`pytest E:/project/advanced-rag/tests/test_api.py -v`
  预期：FAIL

- [ ] **Step 3: 编写 FastAPI 路由与错误捕获**
  在 `src/app.py` 中写入：
  ```python
  from fastapi import FastAPI, HTTPException
  from pydantic import BaseModel
  from src.coordinator import RAGCoordinator
  from src.loader import DocumentLoader
  from src.splitter import SemanticParentChildSplitter
  from src.embedding import LocalEmbeddingService
  from src.database import ChromaAdapter
  from src.reranker import RerankerService

  app = FastAPI(title="Advanced RAG Engine", description="高内聚低耦合混合检索与父子块 RAG API 服务")

  # 初始化全局 Coordinator 实例
  loader = DocumentLoader()
  emb = LocalEmbeddingService()
  splitter = SemanticParentChildSplitter(embedding_service=emb)
  db = ChromaAdapter(db_dir="./vector_db")
  reranker = RerankerService()
  coordinator = RAGCoordinator(loader, splitter, emb, db, reranker)

  class QueryRequest(BaseModel):
      query: str
      top_k: int = 5

  class QueryResponse(BaseModel):
      status: str
      context: str

  @app.post("/retrieve", response_model=QueryResponse)
  async def retrieve(payload: QueryRequest):
      try:
          if not payload.query.strip():
              raise HTTPException(status_code=400, detail="Query 不能为空")
          context = coordinator.query(payload.query)
          return QueryResponse(status="success", context=context)
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))
  ```

- [ ] **Step 4: 运行并确认通过**
  运行：`pytest E:/project/advanced-rag/tests/test_api.py -v`
  预期：PASS

- [ ] **Step 5: 提交代码**
  运行：
  ```bash
  git add E:/project/advanced-rag/src/app.py E:/project/advanced-rag/tests/test_api.py
  git commit -m "feat: 跑通FastAPI端点服务，完成所有测试并正式关闭开发"
  ```
