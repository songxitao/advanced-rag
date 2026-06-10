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
