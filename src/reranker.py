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

    def rerank(self, query: str, candidates: list[dict], top_k: int, cliff_threshold: float = 1.5) -> list[dict]:
        """
        使用 CrossEncoder 计算所有候选子块 (query, child_text) 的精排分数，
        并按照自适应语义断崖截断，返回不超过 top_k 的最相关子块列表。
        :param query: 用户查询问题
        :param candidates: 初筛去重后的候选子块列表，其中子块文本存在 "content" 字段
        :param top_k: 重排最大截取数量
        :param cliff_threshold: 语义断崖阈值，相邻得分落差大于该值时发生截断，默认 1.5
        :return: 重排过滤并截断后的候选子块列表
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

        # 按照精排分数从大到小倒序排序
        sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        truncated_candidates = sorted_candidates[:top_k]

        # 自适应语义断崖检测截断
        if len(truncated_candidates) > 1:
            cutoff_idx = -1
            for i in range(len(truncated_candidates) - 1):
                drop = truncated_candidates[i]["rerank_score"] - truncated_candidates[i+1]["rerank_score"]
                if drop > cliff_threshold:
                    cutoff_idx = i + 1  # 发生断崖，只保留索引 0 到 i 的元素 (共 i+1 个)
                    break
            if cutoff_idx != -1:
                truncated_candidates = truncated_candidates[:cutoff_idx]

        return truncated_candidates
