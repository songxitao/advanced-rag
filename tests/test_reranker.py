import pytest
from src.reranker import RerankerService

class MockRerankerModel:
    def predict(self, pairs, **kwargs):
        # 模拟重排分数输出：
        # 高相关1: 2.5
        # 高相关2: 2.3
        # -- 断崖 (落差 4.4) --
        # 低相关3: -2.1
        # 低相关4: -2.2
        # 低相关5: -2.3
        scores = []
        for pair in pairs:
            content = pair[1]
            if "高相关1" in content:
                scores.append(2.5)
            elif "高相关2" in content:
                scores.append(2.3)
            elif "低相关3" in content:
                scores.append(-2.1)
            elif "低相关4" in content:
                scores.append(-2.2)
            elif "低相关5" in content:
                scores.append(-2.3)
            else:
                scores.append(0.0)
        return scores

def test_reranker_service_cliff_cutoff():
    # 初始化 RerankerService (使用 CPU 模式以加载本地缓存模型)
    service = RerankerService(device="cpu")
    # 替换模型为 Mock
    service.model = MockRerankerModel()
    
    candidates = [
        {"content": "高相关1内容", "metadata": {"parent_id": "p1"}},
        {"content": "高相关2内容", "metadata": {"parent_id": "p2"}},
        {"content": "低相关3内容", "metadata": {"parent_id": "p3"}},
        {"content": "低相关4内容", "metadata": {"parent_id": "p4"}},
        {"content": "低相关5内容", "metadata": {"parent_id": "p5"}},
    ]
    
    # 运行重排
    # 期待结果：因为 2.3 和 -2.1 之间有 4.4 的大落差（大于默认阈值 1.5），
    # 应该在第 2 个片段后截断，仅返回前两个高相关片段。
    selected = service.rerank("用户提问", candidates, top_k=5)
    
    assert len(selected) == 2
    assert selected[0]["content"] == "高相关1内容"
    assert selected[1]["content"] == "高相关2内容"

def test_reranker_device_and_batch():
    import torch
    service = RerankerService()
    if torch.cuda.is_available():
        assert service.model.model.device.type == "cuda"
    else:
        assert service.model.model.device.type == "cpu"
