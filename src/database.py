from abc import ABC, abstractmethod
from typing import List, Dict, Any
import chromadb
from rank_bm25 import BM25Okapi
import jieba
import networkx as nx
import numpy as np
import re
import jieba.posseg as pseg
from collections import Counter

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
        self.graph = nx.Graph()
        self.rebuild_graph()

    def _rebuild_bm25(self) -> None:
        """从已存的 Chroma 中提取文本和元数据，重建 BM25 词频索引"""
        count = self.collection.count()
        self.bm25_docs = []
        if count > 0:
            all_data = self.collection.get(include=["documents", "metadatas"])
            docs = all_data.get("documents", [])
            metadatas = all_data.get("metadatas", [])
            if docs is None or metadatas is None or len(docs) == 0:
                self.bm25 = None
                return
            corpus = []
            for doc, meta in zip(docs, metadatas):
                if doc is None:
                    continue
                self.bm25_docs.append({"document": doc, "metadata": meta})
                # 使用 jieba 对文档进行分词并转为 list
                words = list(jieba.cut(doc))
                corpus.append(words)
            if corpus:
                self.bm25 = BM25Okapi(corpus)
            else:
                self.bm25 = None
        else:
            self.bm25 = None

    def rebuild_graph(self) -> None:
        """从已存的 Chroma 中提取所有 chunks 数据，重建 NetworkX 内存图并建立三轨连边"""
        self.graph = nx.Graph()
        count = self.collection.count()
        if count == 0:
            return

        # 1. 从 Chroma 中提取所有 chunks 数据（包含 doc, metadata, embeddings）
        all_data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])
        embeddings = all_data.get("embeddings", [])
        if ids is None or metadatas is None or embeddings is None:
            return
        n_chunks = len(ids)
        if n_chunks == 0 or len(metadatas) == 0 or len(embeddings) == 0:
            return

        # 2. 将独特的 parent_id 及其 parent_text、embedding、source_path、filename 作为节点加入到 self.graph 中
        parent_nodes_data = {}
        for i in range(n_chunks):
            if i >= len(metadatas) or i >= len(embeddings):
                break
            meta = metadatas[i]
            emb = embeddings[i]
            if meta is None or emb is None:
                continue
            pid = meta.get("parent_id")
            if not pid:
                continue
            if pid not in parent_nodes_data:
                parent_nodes_data[pid] = {
                    "parent_text": meta.get("parent_text", ""),
                    "source_path": meta.get("source_path", ""),
                    "filename": meta.get("filename", ""),
                    "char_start": meta.get("char_start", 0),
                    "embeddings": [emb]
                }
            else:
                parent_nodes_data[pid]["embeddings"].append(emb)
                if "char_start" in meta:
                    parent_nodes_data[pid]["char_start"] = min(parent_nodes_data[pid]["char_start"], meta["char_start"])

        # 计算 parent embedding 均值并添加到图中
        for pid, data in parent_nodes_data.items():
            mean_emb = np.mean(data["embeddings"], axis=0).tolist()
            self.graph.add_node(
                pid,
                parent_text=data["parent_text"],
                embedding=mean_emb,
                source_path=data["source_path"],
                filename=data["filename"],
                char_start=data.get("char_start", 0)
            )

        # 3. 物理相邻边：在同一个文档（相同 source_path）中，按 char_start 排序相邻的前后节点之间建立 physical 边
        doc_groups = {}
        for pid, data in parent_nodes_data.items():
            sp = data["source_path"]
            if sp not in doc_groups:
                doc_groups[sp] = []
            doc_groups[sp].append((pid, data["char_start"]))

        for sp, nodes in doc_groups.items():
            sorted_nodes = sorted(nodes, key=lambda x: x[1])
            for i in range(len(sorted_nodes) - 1):
                u = sorted_nodes[i][0]
                v = sorted_nodes[i+1][0]
                # 计算物理边权重并指数拉伸 (w_base = 0.3)
                weight = float(np.exp(4 * 0.3))
                self.graph.add_edge(u, v, type="physical", weight=weight)

        # 4. 无监督 TF-IDF 实体共现边
        # 4.1 动态注册脱敏代号到分词器中
        pattern = re.compile(r"(?:角色|代号|项目|特工)\s*[A-Za-z0-9_]+")
        for pid in self.graph.nodes:
            text = self.graph.nodes[pid].get("parent_text", "")
            if not text:
                continue
            for match in pattern.finditer(text):
                jieba.add_word(match.group(0), tag='n')

        # 4.2 用 jieba.posseg 对父块分词，仅保留名词和英文词性。统计特征词频（提取 Top-5 关键词）并统计全局名词 DF
        keywords = {}
        all_entity_words_list = []
        for pid in self.graph.nodes:
            text = self.graph.nodes[pid].get("parent_text", "")
            words = []
            if text:
                for word, flag in pseg.cut(text):
                    word_stripped = word.strip()
                    if not word_stripped:
                        continue
                    if flag.startswith('n') or flag == 'eng':
                        words.append(word_stripped)
            all_entity_words_list.append(set(words))
            
            counter = Counter(words)
            top_5 = [w for w, _ in counter.most_common(5)]
            keywords[pid] = top_5

        # 统计全局频次 DF
        df_counter = Counter()
        for word_set in all_entity_words_list:
            df_counter.update(word_set)

        # 计算 IDF 函数
        N_nodes = len(self.graph.nodes)
        def get_idf(word):
            df_val = df_counter.get(word, 0)
            if df_val == 0:
                return 0.0
            return float(np.log(1.0 + N_nodes / df_val))

        # 4.3 在同一个文档内部，若两个父块共享至少一个特征词，则建立实体共现边并赋予 IDF 拉伸权重
        for sp, nodes in doc_groups.items():
            node_ids = [n[0] for n in nodes]
            n_nodes = len(node_ids)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    u = node_ids[i]
                    v = node_ids[j]
                    shared = set(keywords[u]) & set(keywords[v])
                    if shared:
                        sum_idf = sum(get_idf(w) for w in shared)
                        w_base = min(1.0, sum_idf * 0.2)
                        weight = float(np.exp(4 * w_base))
                        self.graph.add_edge(u, v, type="entity", weight=weight)

        # 5. 局域与 ANN 语义关联边
        # 5.1 同一文档内部：使用 NumPy 矩阵乘法批量计算余弦相似度，避免 $O(N^2)$ 双重 Python 循环
        for sp, nodes in doc_groups.items():
            node_ids = [n[0] for n in nodes]
            n_nodes = len(node_ids)
            if n_nodes < 2:
                continue
            
            # 构建嵌入矩阵计算
            embs = np.array([self.graph.nodes[nid]["embedding"] for nid in node_ids], dtype=np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1e-9  # 除零保护
            embs_normed = embs / norms
            sim_matrix = np.dot(embs_normed, embs_normed.T)
            
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if sim_matrix[i, j] >= 0.82:
                        u = node_ids[i]
                        v = node_ids[j]
                        # 语义关联边：使用实际相似度并进行指数拉伸
                        w_base = float(sim_matrix[i, j])
                        weight = float(np.exp(4 * w_base))
                        self.graph.add_edge(u, v, type="semantic", weight=weight)

        # 5.2 跨文档：批量 Query 优化，减少在数据库规模较大时频繁单次 Query 导致的 I/O 与性能退化
        nodes_list = list(self.graph.nodes)
        if nodes_list:
            embs_list = [self.graph.nodes[u]["embedding"] for u in nodes_list]
            results = self.collection.query(
                query_embeddings=embs_list,
                n_results=min(6, count),
                include=["metadatas", "distances"]
            )
            
            if results.get("metadatas") and results.get("distances"):
                for i, u in enumerate(nodes_list):
                    src_u = self.graph.nodes[u]["source_path"]
                    metas = results["metadatas"][i]
                    dists = results["distances"][i]
                    if metas:
                        for idx, meta in enumerate(metas):
                            if meta is None:
                                continue
                            dist = dists[idx]
                            sim = 1.0 - dist
                            if sim >= 0.85:
                                v = meta.get("parent_id")
                                src_v = meta.get("source_path")
                                if v and src_u != src_v and v in self.graph:
                                    # 语义关联边：使用实际相似度并进行指数拉伸
                                    w_base = float(sim)
                                    weight = float(np.exp(4 * w_base))
                                    self.graph.add_edge(u, v, type="semantic", weight=weight)

    def add_chunks(self, chunks_data: List[Dict[str, Any]], dense_embeddings: List[List[float]]) -> None:
        """在 Chroma 里存储 child_text 作为 document，并同步重建 BM25 与 NetworkX 内存图"""
        if not chunks_data or not dense_embeddings:
            return
        # 生成唯一 ID，可以用 parent_id 拼上索引，确保在一批 add_chunks 里 ID 不冲突
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
        self.rebuild_graph()

    def hybrid_search(self, dense_vec: List[float], sparse_vec: Dict[str, float], top_k: int) -> List[Dict[str, Any]]:
        """双通道混合检索与去重 (使用 ThreadPoolExecutor 并行化检索)"""
        from concurrent.futures import ThreadPoolExecutor

        def _search_dense():
            candidates = []
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
                        candidates.append({
                            "content": doc,
                            "metadata": chroma_results["metadatas"][0][i],
                            "score": 1.0 - chroma_results["distances"][0][i]  # cosine 余弦距离转化为相似度分数
                        })
            return candidates

        def _search_sparse():
            candidates = []
            if self.bm25 and len(sparse_vec) > 0:
                query_tokens = list(sparse_vec.keys())
                scores = self.bm25.get_scores(query_tokens)
                
                # 按 BM25 分数排序，选出分数前 top_k * 2 的子块
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]
                for idx in top_indices:
                    if scores[idx] > 0:
                        doc_info = self.bm25_docs[idx]
                        candidates.append({
                            "content": doc_info["document"],
                            "metadata": doc_info["metadata"],
                            "score": float(scores[idx])
                        })
            return candidates

        # 并发启动双通道初筛检索
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(_search_dense)
            future_sparse = executor.submit(_search_sparse)
            dense_candidates = future_dense.result()
            sparse_candidates = future_sparse.result()

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
