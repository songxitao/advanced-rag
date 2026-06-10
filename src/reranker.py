import os

# 导入必要的缓存环境变量
os.environ['HF_HOME'] = r'D:\my_huggingface_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'D:\my_huggingface_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from sentence_transformers import CrossEncoder

class RerankerService:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        """
        初始化 RerankerService
        :param model_name: 模型名称或本地路径，默认为 "BAAI/bge-reranker-v2-m3"
        :param device: 运行设备, 如 "cuda" 或 "cpu"
        """
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        使用 CrossEncoder 计算所有候选子块 (query, child_text) 的精排分数，并按分数倒序截取前 top_k
        :param query: 用户查询问题
        :param candidates: 初筛去重后的候选子块列表，其中子块文本存在 "content" 字段
        :param top_k: 重排截取前 top_k
        :return: 重排过滤后的前 top_k 个候选子块列表
        """
        if not candidates:
            return []

        # 构造句子对 (query, child_text)，其中 child_text 存储在 candidate["content"] 中
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.model.predict(pairs)

        # 兼容 predict 返回单个标量分数的情况
        if not hasattr(scores, "__len__"):
            scores = [scores]

        # 将精排分数保存至 candidates 的 rerank_score 字段中
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        # 按照精排分数从大到小倒序排序，并截取前 top_k 个子块返回
        sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_candidates[:top_k]
