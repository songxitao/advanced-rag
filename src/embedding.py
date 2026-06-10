import os
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
    def __init__(self, device: str = "cuda", model_name: str = "BAAI/bge-m3"):
        """
        初始化 LocalEmbeddingService
        :param device: 运行设备, 如 "cuda" 或 "cpu"
        :param model_name: 模型名称或本地路径，默认为 "BAAI/bge-m3"
        """
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
