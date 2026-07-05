import os
import sys
from collections import Counter
from typing import List, Dict

# 配置 HuggingFace 相关的环境变量（确保直接使用下载好的本地缓存）
os.environ['HF_HOME'] = r'D:\my_huggingface_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'D:\my_huggingface_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 导入 sentence_transformers
from sentence_transformers import SentenceTransformer

class LocalEmbeddingService:
    def __init__(self, device: str = None, model_name: str = "BAAI/bge-m3"):
        """
        初始化 LocalEmbeddingService
        :param device: 运行设备, 如 "cuda" 或 "cpu"
        :param model_name: 模型名称或本地路径，默认为 "BAAI/bge-m3"
        """
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        
        # 兼容不同版本的 sentence-transformers 获取 tokenizer 的方式
        if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
            self.tokenizer = self.model.tokenizer
        elif hasattr(self.model, "submodules") and len(self.model.submodules) > 0 and hasattr(self.model.submodules[0], "tokenizer"):
            self.tokenizer = self.model.submodules[0].tokenizer
        elif len(self.model) > 0 and hasattr(self.model[0], "tokenizer"):
            self.tokenizer = self.model[0].tokenizer
        else:
            raise AttributeError("SentenceTransformer model does not have a tokenizer submodule.")

    def get_dense_embedding(self, text: str) -> List[float]:
        """
        获取 text 的 dense 向量 (1024维)
        :param text: 输入文本
        :return: 稠密向量
        """
        if not text:
            return [0.0] * 1024
        # 计算 dense 向量，并转换为 list[float]
        embedding = self.model.encode(text, convert_to_numpy=True)
        # BGE-M3 的标准稠密向量维度是 1024 维，如果是一维 ndarray，转换为 list 即可
        return embedding.tolist()

    def get_dense_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取 texts 的 dense 向量 (1024维)，分批打印进度防死锁怀疑
        """
        if not texts:
            return []
            
        batch_size = 8
        results = []
        total = len(texts)
        
        print(f"\n[LocalEmbeddingService] Start calculating dense embeddings for {total} chunks...")
        sys.stdout.flush()
        
        batch_idx = 0
        for i in range(0, total, batch_size):
            batch_idx += 1
            sub_texts = texts[i:i+batch_size]
            
            # 在开始计算前打印详细的 Chunk 序号、范围和每个 Chunk 的开头 30 个字符预览
            print(f"\n[LocalEmbeddingService] Computing Batch #{batch_idx} (Chunks {i} to {i + len(sub_texts) - 1}):")
            for idx, text in enumerate(sub_texts):
                chunk_num = i + idx
                preview = text[:30].replace('\n', ' ')
                # 安全兼容：替换掉当前终端编码（如 GBK）不支持的特殊连字符等
                encoding = sys.stdout.encoding or 'gbk'
                preview_safe = preview.encode(encoding, errors='replace').decode(encoding)
                print(f"  - Chunk #{chunk_num}: {preview_safe}...")
            sys.stdout.flush()
            
            embeddings = self.model.encode(sub_texts, convert_to_numpy=True, show_progress_bar=False)
            results.extend(embeddings.tolist())
            
            current_end = min(i + batch_size, total)
            print(f"[LocalEmbeddingService] Progress: {current_end}/{total} chunks computed.")
            sys.stdout.flush()
            
        print("[LocalEmbeddingService] All dense embeddings computed successfully!\n")
        sys.stdout.flush()
        return results

    def get_sparse_embedding(self, text: str) -> Dict[str, float]:
        """
        利用 self.model.tokenizer 提取 text 的分词，并自动计算这些 token 在当前短语中的词频分配
        :param text: 输入文本
        :return: 词频字典
        """
        if not text:
            return {}
        # 提取 tokens
        tokens = self.tokenizer.tokenize(text)
        # 统计频次
        counter = Counter(tokens)
        # 返回 Dict[str, float]
        return {str(k): float(v) for k, v in counter.items()}
